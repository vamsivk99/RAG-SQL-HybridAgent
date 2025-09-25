# Required imports
import os
import re
import textwrap
import uuid
import hashlib
from sqlalchemy import create_engine, inspect
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    SQLDatabase,
    PromptTemplate,
    StorageContext,
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusException

try:
    from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore
    HAS_FAISS = True
except ImportError:  # pragma: no cover - optional dependency
    FaissVectorStore = None  # type: ignore
    HAS_FAISS = False
from pymilvus import MilvusException
from cleanlab_codex.project import Project
from cleanlab_codex.client import Client


#####################################
# Define Tools for Router Agent
#####################################
def create_codex_project():
    """Create a Codex project for document validation."""
    try:
        # 1) If a project access key is provided, use it directly (no user-level key required)
        project_access_key = os.environ.get("CODEX_PROJECT_ACCESS_KEY") or os.environ.get("CODEX_ACCESS_KEY")
        if project_access_key:
            project = Project.from_access_key(project_access_key)
            return project, "external"

        # 2) Else require a USER-level API key to create a project on the fly
        user_key = os.environ.get("CODEX_API_KEY")
        if not user_key:
            print("Warning: CODEX_API_KEY not found. Codex validation will be disabled.")
            return None, None

        project_id = str(uuid.uuid4())[:8]
        codex_client = Client()
        project = codex_client.create_project(name=f"RAG + SQL Router {project_id}")
        access_key = project.create_access_key("default key")
        project = Project.from_access_key(access_key)
        return project, project_id
    except Exception as e:
        print(f"Error creating Codex project: {e}")
        return None, None


# Global variables for reuse - these will persist across function calls
docs_query_engine = None
codex_project = None
current_session_id = None
current_project_id = None

# Cache of previously built document tools keyed by file hash + storage backend
DOC_TOOL_CACHE: dict[str, FunctionTool] = {}


def _build_schema_primer(inspector, tables):
    if not inspector or not tables:
        return ""

    primer_lines = ["Schema overview (table → key columns):"]
    for table in tables:
        try:
            columns = inspector.get_columns(table)
        except Exception:
            continue
        col_names = [col["name"] for col in columns][:12]
        primer_lines.append(f"• {table}: {', '.join(col_names)}")

    # Add hand-crafted context for important tables
    highlights = []
    if "nfip_fact_claim" in tables:
        highlights.append(
            "nfip_fact_claim links FEMA claims to dimensions via state_id, zone_id, event_id, time_id and exposes payout metrics."
        )
    if "dim_time" in tables:
        highlights.append(
            "dim_time includes year, month, quarter to support seasonal trend questions."
        )
    if "quarterly_flood_zone_trend" in tables:
        highlights.append(
            "quarterly_flood_zone_trend is an aggregated view with one row per (year, quarter, rated_zone) containing pre-summed payouts."
        )

    if highlights:
        primer_lines.append("\nImportant relationships:")
        for line in highlights:
            primer_lines.append(f"- {line}")

    return "\n".join(primer_lines)


def _build_few_shot_examples():
    examples = [
        (
            "Which flood zones saw the highest building payouts last quarter?",
            "SELECT year, quarter, rated_zone, total_building_payout\nFROM quarterly_flood_zone_trend\nWHERE (year, quarter) = (2024, 4)\nORDER BY total_building_payout DESC\nLIMIT 5;",
        ),
        (
            "Show building vs contents payouts by state for 2020.",
            "SELECT ds.state,\n       SUM(nf.netBuildingPaymentAmount) AS total_building,\n       SUM(nf.netContentsPaymentAmount) AS total_contents\nFROM nfip_fact_claim nf\nJOIN dim_state ds ON nf.state_id = ds.state_id\nJOIN dim_time dt ON nf.time_id = dt.time_id\nWHERE dt.year = 2020\nGROUP BY ds.state\nORDER BY total_building DESC\nLIMIT 10;",
        ),
        (
            "Give me a quarterly trend for flood zone 'AE'.",
            "SELECT year, quarter, rated_zone, total_building_payout\nFROM quarterly_flood_zone_trend\nWHERE rated_zone = 'AE'\nORDER BY year, quarter;",
        ),
        (
            "Compare total building payouts between coastal and inland zones in 2021.",
            "SELECT CASE WHEN rated_zone LIKE 'V%' THEN 'Coastal' ELSE 'Inland' END AS zone_type,\n       SUM(total_building_payout) AS total_payout\nFROM quarterly_flood_zone_trend\nWHERE year = 2021\nGROUP BY zone_type\nORDER BY total_payout DESC;",
        ),
        (
            "Which states had more than $500M in total payouts over the last five years?",
            "SELECT ds.state,\n       SUM(nf.netBuildingPaymentAmount + nf.netContentsPaymentAmount + nf.netIccPaymentAmount) AS total_payout\nFROM nfip_fact_claim nf\nJOIN dim_state ds ON nf.state_id = ds.state_id\nJOIN dim_time dt ON nf.time_id = dt.time_id\nWHERE dt.year BETWEEN 2019 AND 2023\nGROUP BY ds.state\nHAVING total_payout > 500000000\nORDER BY total_payout DESC;",
        ),
        (
            "List the top 3 flood events by total building payouts and show their primary states.",
            "SELECT nf.floodEvent, ds.state,\n       SUM(nf.netBuildingPaymentAmount) AS total_building_payout\nFROM nfip_fact_claim nf\nJOIN dim_state ds ON nf.state_id = ds.state_id\nGROUP BY nf.floodEvent, ds.state\nORDER BY total_building_payout DESC\nLIMIT 3;",
        ),
    ]
    blocks = []
    for question, sql in examples:
        blocks.append(textwrap.dedent(f"""
        Question: {question}
        SQL:
        {sql}
        """).strip())
    return "\n\n".join(blocks)


_FEW_SHOT_PROMPT = _build_few_shot_examples()


def _normalize_semantic_entities(query: str) -> str:
    normalized = query

    replacements = {
        r"\bQ([1-4])\b": r"Quarter \1",
        r"\bquarterly trend(?:\s+table)?\b": "quarterly_flood_zone_trend",
        r"\bflood zone trend\b": "quarterly_flood_zone_trend",
        r"\bfema\s+claims\b": "nfip_fact_claim",
        r"\bstar\s+schema\b": "nfip_fact_claim and dimensions",
    }

    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    return normalized


def get_or_create_codex_project(session_id):
    """Get existing Codex project or create a new one for the session."""
    global codex_project, current_session_id, current_project_id
    
    # If we have a project and it's for the same session, reuse it
    if codex_project is not None and current_session_id == session_id:
        print(f"Reusing existing Codex project for session {session_id}")
        return codex_project
    
    # Create a new project for this session
    print(f"Creating new Codex project for session {session_id}")
    codex_project, project_id = create_codex_project()
    current_session_id = session_id
    current_project_id = project_id
    
    return codex_project


def get_codex_project_info():
    """Get information about the current Codex project for debugging."""
    global codex_project, current_session_id, current_project_id
    
    if codex_project is None:
        return {
            "status": "No project created",
            "session_id": current_session_id,
            "project_id": None
        }
    
    try:
        # Get the actual project name using the stored project ID
        if current_project_id:
            project_name = f"RAG + SQL Router {current_project_id}"
        else:
            project_name = "RAG + SQL Router Project"
            
        return {
            "status": "Active",
            "session_id": current_session_id,
            "project_id": "Available",
            "project_name": project_name
        }
    except Exception as e:
        return {
            "status": f"Error getting info: {str(e)}",
            "session_id": current_session_id,
            "project_id": "Unknown"
        }


def setup_sql_tool(db_path="city_database.sqlite", tables=("city_stats",), engine_url: str | None = None, schema: str | None = None):
    """Setup SQL query tool for querying a SQLite database.

    Returns a FunctionTool that exposes both the natural language answer and the
    exact SQL used (when available) so the UI can display the query for transparency.
    """
    # Build engine
    try:
        if engine_url:
            engine = create_engine(engine_url)
        else:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found: {db_path}")
            engine = create_engine(f"sqlite:///{db_path}")
        sql_database = SQLDatabase(engine)
    except Exception as e:
        print(f"Error setting up SQL database: {e}")
        raise

    # Determine tables: reflect if not provided
    if not tables:
        try:
            inspector = inspect(engine)
            tables_list = inspector.get_table_names(schema=schema)
        except Exception:
            tables_list = []
    else:
        if isinstance(tables, str):
            tables_list = [tables]
        else:
            tables_list = list(tables)

    # Determine dialect name for prompt hinting
    dialect = "sqlite"
    if engine_url:
        if engine_url.startswith("postgresql"):
            dialect = "postgresql"
        elif engine_url.startswith("mysql"):
            dialect = "mysql"
        elif engine_url.startswith("duckdb"):
            dialect = "duckdb"
        elif engine_url.startswith("sqlite"):
            dialect = "sqlite"

    inspector = None
    try:
        inspector = inspect(engine)
    except Exception:
        inspector = None

    # Ensure the quarterly trend summary table is always available if present
    if inspector is not None:
        try:
            available_tables = set(inspector.get_table_names())
        except Exception:
            available_tables = set()
        else:
            if "quarterly_flood_zone_trend" in available_tables:
                if tables_list:
                    if "quarterly_flood_zone_trend" not in tables_list:
                        tables_list = list(tables_list) + ["quarterly_flood_zone_trend"]
                else:
                    # Leave tables_list as None to allow introspection of all tables
                    pass

    schema_primer = _build_schema_primer(inspector, tables_list or (available_tables if inspector else []))

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=tables_list if tables_list else None,
    )

    # Wrap the query engine so we can return the generated SQL alongside the answer
    def sql_query_tool(query: str):
        normalized_query = _normalize_semantic_entities(query)

        tables_str = ", ".join(tables_list) if tables_list else "(introspected at runtime)"
        hint_sections = [
            f"You are writing {dialect.upper()} SQL. Use the correct syntax for this dialect.",
            f"Available tables: {tables_str}. Prefer explicit joins, standard aliases, and fully qualified column names when ambiguous.",
        ]
        if schema_primer:
            hint_sections.append(schema_primer)
        if _FEW_SHOT_PROMPT:
            hint_sections.append(
                "Reference these examples when helpful to choose the best tables and grouping logic:\n"
                f"{_FEW_SHOT_PROMPT}"
            )
        hint_sections.append("Return only a single SQL query that answers the question.")
        hint = "\n\n".join(hint_sections)

        hinted_query = f"{hint}\n\nQuestion: {normalized_query}"
        response_obj = sql_query_engine.query(hinted_query)
        sql_text = None
        try:
            meta = getattr(response_obj, "metadata", None)
            if isinstance(meta, dict):
                sql_text = meta.get("sql_query") or meta.get("sql_query_raw")
        except Exception:
            sql_text = None
        if not sql_text:
            try:
                extra = getattr(response_obj, "extra_info", None)
                if isinstance(extra, dict):
                    sql_text = extra.get("sql_query") or extra.get("sql")
            except Exception:
                sql_text = None
        response_text = None
        try:
            response_text = getattr(response_obj, "response", None)
        except Exception:
            response_text = None
        if not response_text:
            response_text = str(response_obj)
        return {
            "response": response_text,
            "sql": sql_text,
        }

    # Create tool for SQL querying as a function tool
    sql_tool = FunctionTool.from_defaults(
        sql_query_tool,
        name="sql_tool",
        description=(
            "Translate a natural language question into SQL over the configured tables"
            " and return both the answer and the exact SQL used."
        ),
    )

    # Return the SQL tool
    return sql_tool


def setup_document_tool(file_dir, session_id=None, use_milvus=True, milvus_uri="http://localhost:19530", progress_callback=None):
    """Setup document query tool from uploaded documents with Codex validation."""
    global docs_query_engine

    # Create a reader and load the data
    reader, node_parser = DoclingReader(), MarkdownNodeParser()
    loader = SimpleDirectoryReader(
        input_dir=file_dir,
        file_extractor={
            ".pdf": reader,
            ".docx": reader,
            ".pptx": reader,
            ".txt": reader,
        },
    )
    docs = loader.load_data()
    if progress_callback:
        progress_callback("Stage 1/3: Document parsing complete. Chunking and embedding next…")

    # Compute cache key for current file set and backend selection
    cache_hasher = hashlib.sha256()
    for name in sorted(os.listdir(file_dir)):
        full_path = os.path.join(file_dir, name)
        if os.path.isfile(full_path):
            cache_hasher.update(name.encode("utf-8"))
            with open(full_path, "rb") as f_in:
                for chunk in iter(lambda: f_in.read(8192), b""):
                    cache_hasher.update(chunk)
    cache_hasher.update(str(bool(use_milvus)).encode("utf-8"))
    cache_key = cache_hasher.hexdigest()

    if cache_key in DOC_TOOL_CACHE:
        print("Reusing cached document tool for uploaded files")
        return DOC_TOOL_CACHE[cache_key]

    # Creating a vector index over loaded data
    vector_index = None
    milvus_error = None
    if use_milvus:
        unique_collection_id = uuid.uuid4().hex
        collection_name = f"rag_with_sql_{unique_collection_id}"
        try:
            vector_store = MilvusVectorStore(
                uri=milvus_uri,
                dim=384,
                overwrite=True,
                collection_name=collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex.from_documents(
                docs,
                show_progress=True,
                transformations=[node_parser],
                storage_context=storage_context,
            )
            if progress_callback:
                progress_callback("Stage 2/3: Embeddings created. Indexing into Milvus…")
        except MilvusException as e:
            milvus_error = str(e)
        except Exception as e:
            milvus_error = str(e)

    if vector_index is None:
        fallback_message = "Stage 2/3: Embeddings created."
        if not HAS_FAISS:
            raise ImportError(
                "FAISS fallback requested but llama_index.vector_stores.faiss is not installed. "
                "Install it with `pip install llama-index-vector-stores-faiss` or keep Milvus enabled."
            )
        fallback_store = FaissVectorStore(dim=384)
        storage_context = StorageContext.from_defaults(vector_store=fallback_store)
        vector_index = VectorStoreIndex.from_documents(
            docs,
            show_progress=True,
            transformations=[node_parser],
            storage_context=storage_context,
        )
        if progress_callback:
            progress_callback(f"{fallback_message} Using FAISS in-memory index (Milvus unavailable).")
    elif progress_callback:
        progress_callback("Stage 3/3: Milvus index ready. Building query engine…")

    if milvus_error:
        print(f"Milvus unavailable, fell back to FAISS: {milvus_error}")

    # Custom prompt template
    template = (
        "You are a meticulous and accurate document analyst. Your task is to answer the user's question based exclusively on the provided context. "
        "Follow these rules strictly:\n"
        "1. Your entire response must be grounded in the facts provided in the 'Context' section. Do not use any prior knowledge.\n"
        "2. If multiple parts of the context are relevant, synthesize them into a single, coherent answer.\n"
        "3. If the context does not contain the information needed to answer the question, you must state only: 'The provided context does not contain enough information to answer this question.'\n"
        "-----------------------------------------\n"
        "Context: {context_str}\n"
        "-----------------------------------------\n"
        "Question: {query_str}\n\n"
        "Answer:"
    )
    qa_template = PromptTemplate(template)

    # Create a query engine for the vector index
    docs_query_engine = vector_index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=8,
        response_mode="tree_summarize",
    )

    # Get or create Codex project for this session
    codex_project = get_or_create_codex_project(session_id)

    # Define the document query function with Codex validation
    def document_query_tool(query: str):
        """Query documents with Codex validation for enhanced accuracy."""
        # Optional per-file filtering if user mentions a specific filename
        filtered_engine = None
        mentioned_file = None
        for ext in [".pdf", ".docx", ".pptx", ".txt"]:
            if ext.lower() in query.lower():
                try:
                    import re
                    m = re.search(r"([\w\-\.]+\%s)" % re.escape(ext), query, flags=re.IGNORECASE)
                    if m:
                        mentioned_file = m.group(1)
                        break
                except Exception:
                    mentioned_file = None

        if mentioned_file:
            try:
                filters = MetadataFilters(filters=[MetadataFilter(key="file_name", value=mentioned_file)])
                filtered_engine = vector_index.as_query_engine(
                    text_qa_template=qa_template,
                    similarity_top_k=8,
                    response_mode="tree_summarize",
                    filters=filters,
                )
            except Exception:
                filtered_engine = None

        # Step 1: Query the engine (filtered if possible)
        engine_to_use = filtered_engine or docs_query_engine
        response_obj = engine_to_use.query(query)
        # If filtered query returns no context, fall back to unfiltered
        try:
            src = getattr(response_obj, "source_nodes", [])
            if filtered_engine is not None and (src is None or len(src) == 0):
                response_obj = docs_query_engine.query(query)
        except Exception:
            pass
        initial_response = str(response_obj)

        # Step 2: Gather source context
        context = response_obj.source_nodes
        context_str = "\n".join([n.node.text for n in context])

        # Step 3: Prepare prompt for Codex validation
        prompt_template = (
            "You are a meticulous and accurate document analyst. Your task is to answer the user's question based exclusively on the provided context. "
            "Follow these rules strictly:\n"
            "1. Your entire response must be grounded in the facts provided in the 'Context' section. Do not use any prior knowledge.\n"
            "2. If multiple parts of the context are relevant, synthesize them into a single, coherent answer.\n"
            "3. If the context does not contain the information needed to answer the question, you must state only: 'The provided context does not contain enough information to answer this question.'\n"
            "-----------------------------------------\n"
            "Context: {context}\n"
            "-----------------------------------------\n"
            "Question: {query}\n\n"
            "Answer:"
        )
        user_prompt = prompt_template.format(context=context_str, query=query)
        messages = [{"role": "user", "content": user_prompt}]

        # Step 4: Validate with Codex (if available)
        if codex_project:
            try:
                print(f"Validating query with Codex: '{query[:50]}...'")
                result = codex_project.validate(
                    messages=messages,
                    query=query,
                    context=context_str,
                    response=initial_response,
                )
                print("Codex validation completed successfully")

                # Step 5: Final response selection
                fallback_response = "I'm sorry, I couldn't find an answer — can I help with something else?"
                final_response = (
                    result.expert_answer
                    if result.expert_answer and result.escalated_to_sme
                    else (
                        fallback_response
                        if result.should_guardrail
                        else initial_response
                    )
                )
                trust_score = result.model_dump()["eval_scores"]["trustworthiness"]["score"]

                # Return a dictionary to avoid tuple handling issues
                return {
                    "response": str(final_response),
                    "trust_score": float(trust_score)
                }
            except Exception as e:
                # If Codex validation fails, return the initial response
                print(f"Codex validation failed: {e}")
                return {
                    "response": str(initial_response),
                    "trust_score": None
                }
        else:
            # If Codex is not available, return the initial response
            print("Codex not available, using basic RAG response")
            return {
                "response": str(initial_response),
                "trust_score": None
            }

    # Create tool for document querying using FunctionTool
    docs_tool = FunctionTool.from_defaults(
        document_query_tool,
        name="document_tool",
        description=(
            "Useful for answering a natural language question by performing a semantic search over "
            "a collection of documents. These documents may contain general knowledge, reports, "
            "or domain-specific content. Returns the most relevant passages or synthesized answers. "
            "If the user query does not relate to US city statistics (population and state), use this document search tool."
        ),
    )

    # Cache and return the document tool
    DOC_TOOL_CACHE[cache_key] = docs_tool
    return docs_tool
