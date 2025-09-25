# Required imports
import torch
torch.classes.__path__ = []

import os
import uuid
import random
import time
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text, inspect

# Import necessary components from llama_index
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import custom tools and workflow
from tools import setup_document_tool, setup_sql_tool, get_codex_project_info
from workflow import RouterOutputAgentWorkflow


from llama_index.llms.openai import OpenAI


# Apply nest_asyncio to allow running asyncio in Streamlit
import asyncio
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Text2SQL + RAG hybrid query engine ⚙️ ",
    page_icon="🏙️",
    layout="wide",
)

# Initialize session state
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "workflow_needs_update" not in st.session_state:
    st.session_state.workflow_needs_update = False

if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False

if "codex_api_key" not in st.session_state:
    st.session_state.codex_api_key = ""

if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = ""

# Bootstrap keys/models from environment or Streamlit secrets (optional)
def _get_env_or_secret(key: str, default: str | None = None):
    val = os.environ.get(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# Pre-populate Codex key if provided via env/secrets
if not st.session_state.codex_api_key:
    _codex = _get_env_or_secret("CODEX_API_KEY")
    if _codex:
        st.session_state.codex_api_key = _codex
        os.environ["CODEX_API_KEY"] = _codex
        st.session_state.workflow_needs_update = True

# Also propagate a Codex Project Access Key from secrets/env into environment
_codex_project_key = _get_env_or_secret("CODEX_PROJECT_ACCESS_KEY") or _get_env_or_secret("CODEX_ACCESS_KEY")
if _codex_project_key and not os.environ.get("CODEX_PROJECT_ACCESS_KEY") and not os.environ.get("CODEX_ACCESS_KEY"):
    os.environ["CODEX_PROJECT_ACCESS_KEY"] = _codex_project_key

# Pre-populate OpenRouter key if provided via env/secrets
if not st.session_state.openrouter_api_key:
    _openrouter = _get_env_or_secret("OPENROUTER_API_KEY")
    if _openrouter:
        st.session_state.openrouter_api_key = _openrouter
        st.session_state.llm_initialized = False
        st.session_state.workflow_needs_update = True

# Cache model name from env/secrets or default
if "openrouter_model" not in st.session_state:
    st.session_state.openrouter_model = (
        _get_env_or_secret("OPENROUTER_MODEL", "qwen/qwen-turbo")
        or "qwen/qwen-turbo"
    )


#####################################
# Helper Functions
#####################################
def reset_chat():
    """Reset the chat history and clear context"""
    # Clear messages immediately
    if "messages" in st.session_state:
        st.session_state.messages = []

    if "workflow" in st.session_state and st.session_state.workflow:
        st.session_state.workflow = None
        st.session_state.workflow_needs_update = True


#####################################
# Database Visualization Helpers
#####################################


def set_active_db_config(db_type, sqlite_path=None, engine_url=None, tables=None):
    """Persist the active database configuration and dispose old engines."""
    if "active_db_engine" in st.session_state:
        try:
            st.session_state["active_db_engine"].dispose()
        except Exception:
            pass
        del st.session_state["active_db_engine"]

    st.session_state["active_db_config"] = {
        "db_type": db_type,
        "sqlite_path": sqlite_path,
        "engine_url": engine_url,
        "tables": tuple(tables) if tables else None,
    }


def get_active_db_config():
    return st.session_state.get("active_db_config")


def get_active_engine():
    cfg = get_active_db_config()
    if not cfg:
        return None, []

    engine = st.session_state.get("active_db_engine")
    if engine is None:
        try:
            if cfg["db_type"] == "SQLite (file)":
                path = cfg.get("sqlite_path")
                if not path or not os.path.exists(path):
                    st.error(f"SQLite database not found: {path}")
                    return None, []
                engine = create_engine(f"sqlite:///{path}")
            else:
                engine_url = cfg.get("engine_url")
                if not engine_url:
                    st.error("No SQLAlchemy URL configured.")
                    return None, []
                engine = create_engine(engine_url)
            st.session_state["active_db_engine"] = engine
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            return None, []

    tables = cfg.get("tables")
    try:
        inspector = inspect(engine)
        if not tables:
            tables = inspector.get_table_names()
            cfg["tables"] = tuple(tables)
            st.session_state["active_db_config"] = cfg
    except Exception as e:
        st.error(f"Failed to inspect database: {e}")
        return None, []

    return engine, list(tables)


def fetch_dataframe(query, engine):
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn)


def fetch_scalar(query, engine):
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()


def render_database_tab():
    """Render an updated database explorer for the active FEMA dataset."""
    st.header("🗄️ Database Visualization")
    
    engine, tables = get_active_engine()
    if not engine or not tables:
        st.error("No database is configured. Use the sidebar to select a database.")
        return
    
    table_name = st.selectbox("Select table", tables, index=0, key="db_table_select")

    try:
        inspector = inspect(engine)
        columns_info = inspector.get_columns(table_name)
        column_names = {col["name"] for col in columns_info}
    except Exception as e:
        st.error(f"Unable to read table schema: {e}")
        return

    st.subheader("📊 Quick Metrics")
    metrics_cols = st.columns(4)

    try:
        total_records = fetch_scalar(f"SELECT COUNT(*) FROM {table_name}", engine)
    except Exception:
        total_records = None
    with metrics_cols[0]:
        st.metric("Records", f"{total_records:,}" if total_records is not None else "—")

    payout_terms = []
    if "amountPaidOnBuildingClaim" in column_names:
        payout_terms.append("COALESCE(amountPaidOnBuildingClaim,0)")
    if "amountPaidOnContentsClaim" in column_names:
        payout_terms.append("COALESCE(amountPaidOnContentsClaim,0)")
    if "netBuildingPaymentAmount" in column_names:
        payout_terms.append("COALESCE(netBuildingPaymentAmount,0)")
    if "netContentsPaymentAmount" in column_names:
        payout_terms.append("COALESCE(netContentsPaymentAmount,0)")
    if "netIccPaymentAmount" in column_names:
        payout_terms.append("COALESCE(netIccPaymentAmount,0)")
    payout_sql = " + ".join(payout_terms) if payout_terms else "0"

    total_payout = None
    if payout_terms:
        try:
            total_payout = fetch_scalar(f"SELECT SUM({payout_sql}) FROM {table_name}", engine)
        except Exception:
            total_payout = None
    with metrics_cols[1]:
        st.metric("Total Payout", f"${total_payout:,.0f}" if total_payout is not None else "—")

    if "state" in column_names:
        try:
            distinct_states = fetch_scalar(f"SELECT COUNT(DISTINCT state) FROM {table_name}", engine)
        except Exception:
            distinct_states = None
        metrics_cols[2].metric("States", distinct_states if distinct_states is not None else "—")
    else:
        metrics_cols[2].metric("States", "N/A")

    if "dateOfLoss" in column_names:
        try:
            date_range = fetch_dataframe(
                f"SELECT MIN(dateOfLoss) AS min_date, MAX(dateOfLoss) AS max_date FROM {table_name}",
                engine,
            )
            min_date = pd.to_datetime(date_range.at[0, "min_date"]).date()
            max_date = pd.to_datetime(date_range.at[0, "max_date"]).date()
            date_display = f"{min_date} → {max_date}"
        except Exception:
            date_display = "—"
        metrics_cols[3].metric("Date Range", date_display)
    elif "yearOfLoss" in column_names:
        try:
            year_range = fetch_dataframe(
                f"SELECT MIN(yearOfLoss) AS min_year, MAX(yearOfLoss) AS max_year FROM {table_name}",
                engine,
            )
            date_display = f"{int(year_range.at[0, 'min_year'])} → {int(year_range.at[0, 'max_year'])}"
        except Exception:
            date_display = "—"
        metrics_cols[3].metric("Year Range", date_display)
    else:
        metrics_cols[3].metric("Date Range", "N/A")

    st.markdown("---")
    st.subheader("📈 Key Breakdowns")
    tab_states, tab_zones, tab_timeline = st.tabs(["Top Regions", "Flood Zones", "Timeline"])

    if "state" in column_names:
        with tab_states:
            try:
                df_states = fetch_dataframe(
                    f"""
                    SELECT state,
                           COUNT(*) AS claim_count,
                           SUM({payout_sql}) AS total_payout
                    FROM {table_name}
                    WHERE state IS NOT NULL AND state <> ''
                    GROUP BY state
                    ORDER BY claim_count DESC
                    LIMIT 10
                    """,
                    engine,
                )
                if not df_states.empty:
                    fig = px.bar(
                        df_states,
                        x="state",
                        y="claim_count",
                        hover_data=["total_payout"],
                        title="Top 10 States by Claim Count",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No state data available.")
            except Exception as e:
                st.warning(f"Unable to build state chart: {e}")
    else:
        tab_states.info("State information not available in this table.")

    zone_column = None
    for candidate in ["ratedFloodZone", "floodZoneCurrent", "floodEvent"]:
        if candidate in column_names:
            zone_column = candidate
            break
    if zone_column:
        with tab_zones:
            try:
                df_zones = fetch_dataframe(
                    f"""
                    SELECT {zone_column} AS zone,
                           COUNT(*) AS claim_count,
                           SUM({payout_sql}) AS total_payout
                    FROM {table_name}
                    WHERE {zone_column} IS NOT NULL AND {zone_column} <> ''
                    GROUP BY {zone_column}
                    ORDER BY claim_count DESC
                    LIMIT 10
                    """,
                    engine,
                )
                if not df_zones.empty:
                    fig = px.bar(
                        df_zones,
                        x="zone",
                        y="claim_count",
                        hover_data=["total_payout"],
                        title=f"Top 10 {zone_column} by Claim Count",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No flood-zone data available.")
            except Exception as e:
                st.warning(f"Unable to build flood-zone chart: {e}")
    else:
        tab_zones.info("Flood-zone columns not available in this table.")

    if "yearOfLoss" in column_names:
        with tab_timeline:
            try:
                df_year = fetch_dataframe(
                    f"""
                    SELECT yearOfLoss AS year,
                           COUNT(*) AS claim_count,
                           SUM({payout_sql}) AS total_payout
                    FROM {table_name}
                    WHERE yearOfLoss IS NOT NULL
                    GROUP BY yearOfLoss
                    ORDER BY yearOfLoss
                    """,
                    engine,
                )
                if not df_year.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_year["year"],
                        y=df_year["claim_count"],
                        mode="lines+markers",
                        name="Claims",
                    ))
                    fig.update_layout(title="Claim Volume Over Time", xaxis_title="Year", yaxis_title="Claims")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No year data available.")
            except Exception as e:
                st.warning(f"Unable to build timeline chart: {e}")
    elif "dateOfLoss" in column_names:
        with tab_timeline:
            try:
                df_year = fetch_dataframe(
                    f"""
                    SELECT strftime('%Y', dateOfLoss) AS year,
                           COUNT(*) AS claim_count,
                           SUM({payout_sql}) AS total_payout
                    FROM {table_name}
                    WHERE dateOfLoss IS NOT NULL
                    GROUP BY year
                    ORDER BY year
                    """,
                    engine,
                )
                if not df_year.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_year["year"],
                        y=df_year["claim_count"],
                        mode="lines+markers",
                        name="Claims",
                    ))
                    fig.update_layout(title="Claim Volume Over Time", xaxis_title="Year", yaxis_title="Claims")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timeline data available.")
            except Exception as e:
                st.warning(f"Unable to build timeline chart: {e}")
    else:
        tab_timeline.info("Year/date columns not available in this table.")

    st.markdown("---")
    st.subheader("🔍 Sample Records")
    try:
        df_sample = fetch_dataframe(f"SELECT * FROM {table_name} LIMIT 500", engine)
        st.dataframe(df_sample, height=400, width="stretch")
    except Exception as e:
        st.warning(f"Unable to load sample rows: {e}")


@st.cache_resource
def initialize_model(_api_key, model_name: str | None = None):
    """Initialize models for LLM and embedding"""
    try:
        # Initialize models for LLM and embedding with OpenRouter
        # llm = OpenAI(model="gpt-4o-mini", api_key=_api_key)
        selected_model = (
            model_name
            or os.environ.get("OPENROUTER_MODEL")
            or "qwen/qwen-turbo"
        )
        llm = OpenRouter(model=selected_model, api_key=_api_key)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return llm, embed_model
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None


def handle_file_upload(uploaded_files):
    """Function to handle multiple file uploads with temporary directory"""
    try:
        # Create a temporary directory if it doesn't exist yet
        if not hasattr(st.session_state, "temp_dir") or not os.path.exists(
            st.session_state.temp_dir
        ):
            st.session_state.temp_dir = tempfile.mkdtemp()

        # Track uploaded files
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        # Process each uploaded file
        file_paths = []
        for uploaded_file in uploaded_files:
            # Save file to temporary location
            temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)

            # Write the file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Add to list of uploaded files
            st.session_state.uploaded_files.append(
                {"name": uploaded_file.name, "path": temp_file_path}
            )
            file_paths.append(temp_file_path)

        st.session_state.file_uploaded = True
        st.session_state.current_pdf = (
            file_paths[0] if file_paths else None
        )  # Set first file as current for preview
        st.session_state.workflow_needs_update = True  # Mark workflow for update

        return file_paths
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")
        return None


def initialize_workflow(tools):
    """Initialize workflow with the given tools"""
    try:
        workflow = RouterOutputAgentWorkflow(tools=tools, verbose=False, timeout=120)
        st.session_state.workflow = workflow
        return workflow
    except Exception as e:
        st.error(f"Error initializing workflow: {str(e)}")
        return None


async def process_query(query, workflow):
    """Function to process a query using the workflow"""
    try:
        # Clear chat history before processing new query to avoid persistence
        workflow.chat_history = []
        
        with st.spinner("Processing your query..."):
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(workflow.run(message=query), timeout=60.0)
            
            # Extract tool usage information from chat history
            tool_usage_info = []
            for msg in workflow.chat_history:
                if msg.role == "tool" and hasattr(msg, 'additional_kwargs'):
                    tool_name = msg.additional_kwargs.get('tool_used', 'Unknown Tool')
                    trust_score = msg.additional_kwargs.get('trust_score')
                    
                    # Skip cancelled or error messages in the display
                    if "cancelled" not in msg.content.lower() and "error executing tool" not in msg.content.lower():
                        # Handle the response content - it might be a dictionary string
                        response_content = msg.content
                        
                        # If it's a dictionary string (like the context not found case), extract fields
                        if response_content.startswith("{'response':") or response_content.startswith('{"response":'):
                            try:
                                import ast
                                # Try to safely evaluate the string as a dictionary
                                response_dict = ast.literal_eval(response_content)
                                if isinstance(response_dict, dict) and 'response' in response_dict:
                                    response_content = response_dict['response']
                                    # Update trust_score if it's in the dict
                                    if 'trust_score' in response_dict and trust_score is None:
                                        trust_score = response_dict['trust_score']
                                    # Capture SQL if provided by sql_tool wrapper
                                    sql_text = response_dict.get('sql')
                                    if sql_text:
                                        # Attach SQL to additional kwargs for downstream formatting
                                        if not hasattr(msg, 'additional_kwargs'):
                                            msg.additional_kwargs = {}
                                        msg.additional_kwargs['sql'] = sql_text

                            except (ValueError, SyntaxError, TypeError) as e:
                                # If parsing fails, keep the original content
                                if st.session_state.get('verbose', False):
                                    print(f"Failed to parse response dict: {e}")
                                pass
                        
                        tool_info = {
                            'tool_name': tool_name,
                            'response': response_content,
                            'trust_score': trust_score,
                            'sql': getattr(msg, 'additional_kwargs', {}).get('sql') if hasattr(msg, 'additional_kwargs') else None,
                        }
                        tool_usage_info.append(tool_info)
            
            # Simple formatting logic: if we have tool responses, format them
            if tool_usage_info:
                formatted_response = ""
                for i, tool_info in enumerate(tool_usage_info, 1):
                    # Determine if this is a document tool (anything not SQL)
                    is_doc_tool = tool_info['tool_name'] != 'sql_tool'
                    
                    if is_doc_tool:
                        formatted_response += "**🔧 Tool Used:** `document_tool`\n\n"
                    else:
                        formatted_response += f"**🔧 Tool Used:** `{tool_info['tool_name']}`\n\n"
                    
                    formatted_response += f"**📝 Response:**\n\n{tool_info['response']}\n\n"

                    # If SQL tool returned the exact SQL, show it
                    if tool_info['tool_name'] == 'sql_tool' and tool_info.get('sql'):
                        formatted_response += "**🧾 SQL Used:**\n\n"
                        formatted_response += f"```sql\n{tool_info['sql']}\n```\n\n"
                    
                    # Only show trust score for document tools
                    if is_doc_tool and tool_info['trust_score'] is not None:
                        trust_percentage = round(tool_info['trust_score'] * 100, 1)
                        trust_emoji = "🟢" if trust_percentage >= 70 else "🟡" if trust_percentage >= 50 else "🔴"
                        formatted_response += f"**{trust_emoji} Trust Score:** {trust_percentage}%\n\n"
                    
                    if i < len(tool_usage_info):
                        formatted_response += "---\n\n"
                
                return formatted_response
            else:
                return result
                
    except asyncio.TimeoutError:
        print("Query processing timed out")
        return "The query took too long to process. Please try a simpler question or try again."
    except asyncio.CancelledError:
        print("Query processing was cancelled")
        return "The query processing was cancelled. Please try again."
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try a different question."


#####################################
# Main Streamlit app
#####################################
def main():
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration 🔑")

        # Codex API Key input
        codex_logo_html = """
            <div style='display: flex; align-items: center; gap: 0px; margin-top: 2px;'>
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjgAAACACAMAAAAxreP1AAAC91BMVEVHcEz///8AAJP///7+/v9/f//////9/fzIyO/+/v/8/Pz+/v/7+/3////////n5/D////29vvp6e3////7+/v////7+/3////////+/v/6+vz////////+/v/5+fz////9/f7////////////29vr////39//39/v39/f////39/3////8/P3////4+Pv39/r+/v/////8/P75+fn////8/P35+fv////9/f7////6+v36+v3////////6+vv6+v36+vv9/f78/P38/P39/f79/f78/P78/P79/f/9/f/7+/z7+/v8/P77+/z////8/P/7+/v8/P37+/z+/v7+/v/8/P39/f79/f38/P79/f78/P/8/Pz8/P38/P3+/v79/f78/P3+/v/8/P38/P39/f78/P38/P38/P38/P78/P38/P38/P38/P39/f78/Pz9/f77+/39/f7+/v/9/f77+/3+/v78/P78/P3+/v/+/v7////7+/37+/v8/P38/P39/f79/f79/f78/P39/f39/f79/f78/P39/f77+/38/P77+/3+/v79/f39/f79/f38/P36+vz8/P79/f79/f39/f/9/f38/P39/f78/P39/f79/f39/f38/P38/P3////9/f79/f39/f39/f39/f79/f39/f78/P39/f79/f39/f38/P39/f7////9/f79/f38/P38/P3////8/P39/f79/f79/f78/P3////9/f78/P39/f79/f78/Pz////9/f79/f39/f78/P3+/v/7+/z////////+/v/8/P3////9/f/9/f78/P3////8/P38/Pz9/f/9/f79/f3////9/f78/P3////9/f/9/f78/P3////////9/f78/P37+/z////////9/f7////////9/f/8/P3////////9/f/9/f79/f38/P3////////9/f/9/f79/f38/P38/P38/Pz////////+/v/+/v79/f/9/f79/f38/P/8/P38/P159MFNAAAA/HRSTlMAAQEDAgIEBQQGBwgJCgsLDA0MDg8QERITFRQWFxkYGxocHh8eICEgISMiJCUnJygqLSwuMTAzNTQ3Njc5Ozo9PD8+QUJERklKTE1MTk9QUFBTUlRWV1hZ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" width="140"> 
            </div>
        """
        st.markdown(codex_logo_html, unsafe_allow_html=True)
        st.markdown("[Get your Codex API key](https://codex.cleanlab.ai/account)", unsafe_allow_html=True)
        codex_key_input = st.text_input(
            "Enter Codex API Key",
            type="password",
            value=st.session_state.codex_api_key,
            help="Required for enhanced response validation"
        )

        if codex_key_input != st.session_state.codex_api_key:
            st.session_state.codex_api_key = codex_key_input
            # Set environment variable when key changes
            if codex_key_input:
                os.environ["CODEX_API_KEY"] = codex_key_input
            st.session_state.workflow_needs_update = True

        # Display Codex status
        if st.session_state.codex_api_key:
            st.success("Codex API Key: ✓")

            # Show Codex project debug info
            if st.session_state.file_uploaded:
                codex_info = get_codex_project_info()
                # st.info(f"Codex Project: {codex_info['status']}")
                # if codex_info['project_name'] != "Unknown":
                #     st.info(f"Project: {codex_info['project_name']}")
                #     st.info(f"Session: {codex_info['session_id']}")
                st.info(f"Codex Project: {codex_info.get('status', 'Unknown')}")
                project_name = codex_info.get('project_name', 'Unknown')
                if project_name != "Unknown":
                    st.info(f"Project: {project_name}")
                session_id = codex_info.get('session_id')
                if session_id:
                    st.info(f"Session: {session_id}")
        else:
            st.warning("Codex API Key: ✗")

        # OpenRouter API Key input
        openrouter_logo_html = """
            <div style='display: flex; align-items: center; gap: 0px; margin-top: 0px;'>
                <img src="https://files.buildwithfern.com/openrouter.docs.buildwithfern.com/docs/2025-07-24T05:04:17.529Z/content/assets/logo-white.svg" width="180"> 
            </div>
        """
        st.markdown(openrouter_logo_html, unsafe_allow_html=True)
        st.markdown("[Get your OpenRouter key](https://openrouter.ai/keys)", unsafe_allow_html=True)
        openrouter_key_input = st.text_input(
            "Enter OpenRouter API Key",
            type="password",
            value=st.session_state.openrouter_api_key,
            help="Required for LLM functionality"
        )

        if openrouter_key_input != st.session_state.openrouter_api_key:
            st.session_state.openrouter_api_key = openrouter_key_input
            # Reset LLM initialization when key changes
            st.session_state.llm_initialized = False
            st.session_state.workflow_needs_update = True

        # Display OpenRouter status
        if st.session_state.openrouter_api_key:
            st.success("OpenRouter API Key: ✓")
        else:
            st.warning("OpenRouter API Key: ✗")

        # LLM initialization status
        if not st.session_state.llm_initialized and st.session_state.openrouter_api_key:
            with st.spinner("Initializing models..."):
                llm, embed_model = initialize_model(
                    st.session_state.openrouter_api_key,
                    model_name=st.session_state.openrouter_model,
                )
                if llm:
                    Settings.llm = llm
                    Settings.embed_model = embed_model
                    st.session_state.llm_initialized = True
                    st.success("Models initialized!")
                else:
                    st.error("Failed to initialize models.")
        elif st.session_state.llm_initialized:
            st.success("Models: Ready ✓")
        else:
            st.warning("Models: ✗ (OpenRouter API Key required)")

        # File upload section in the sidebar
        st.header("Documents 📄")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "pptx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Upload Documents"):
                file_paths = handle_file_upload(uploaded_files)
                if file_paths:
                    st.success(f"{len(file_paths)} document(s) uploaded")

                    # Display list of uploaded documents
                    st.write("Uploaded documents:")
                    for i, file in enumerate(st.session_state.uploaded_files):
                        st.write(f"- {file['name']}")

        # Document status indicator
        if st.session_state.file_uploaded:
            st.success("Documents: ✓")
        else:
            st.warning("Documents: ✗")

    # Retired quick-insight buttons (caused confusion when agent could not reach new tables reliably)
    st.markdown("---")

    # Chat title and reset button in the same row
    chat_header_col1, chat_header_col2 = st.columns([6, 1])
    with chat_header_col1:
        st.title("RAG + SQL Router 🔗")
        powered_by_html = """
            <div style='display: flex; align-items: center; gap: 10px; margin-top: -10px;'>
                <span style='font-size: 20px; color: #666;'>Powered by</span>
                <img src="data:image/jpeg;base64,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" width="40" height="50"> 
                <span style='font-size: 20px; color: #666;'>and</span>
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7d/Milvus-logo-color-small.png" width="100">
            </div>
        """
        st.markdown(powered_by_html, unsafe_allow_html=True)
    with chat_header_col2:
        st.button("Reset Chat ↺", on_click=reset_chat)
    
    # Add a small section for database access
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("💡 **Tip:** Want to explore the database? Click the button to view data and run custom queries!")
    with col2:
        if st.button("🗄️ View Database"):
            st.session_state.show_database = True
    
    # Show database visualization if requested
    if st.session_state.get('show_database', False):
        with st.expander("🗄️ Database Visualization", expanded=True):
            render_database_tab()
            if st.button("Close Database View"):
                st.session_state.show_database = False
                st.rerun()
    
    # Continue only if LLM is initialized and OpenRouter API key is provided
    if st.session_state.llm_initialized and st.session_state.openrouter_api_key:
        # Initialize tools with first tool as SQL tool
        # Sidebar DB selector
        st.sidebar.subheader("Database ⚙️")
        use_milvus = st.sidebar.checkbox("Use Milvus Vector Store", value=True,
                                         help="Toggle Milvus on/off. When off, FAISS is used as a fallback.")
        db_type = st.sidebar.selectbox("DB Type", ["SQLite (file)", "SQLAlchemy URL"], index=0)
        engine_url = None
        selected_tables = None
        db_path_input = None
        tables_input = ""
        if db_type == "SQLite (file)":
            default_db = os.environ.get("SQLITE_DB_PATH") or "fema_nfip_star.sqlite"
            db_path_input = st.sidebar.text_input("SQLite path", value=default_db)
            tables_env = os.environ.get("SQLITE_TABLES", "")
            tables_input = st.sidebar.text_input("Tables (comma-separated, optional)", value=tables_env)
            if tables_input.strip():
                selected_tables = tuple([t.strip() for t in tables_input.split(",") if t.strip()])
        else:
            engine_url = st.sidebar.text_input("SQLAlchemy URL", value=os.environ.get("ENGINE_URL", ""), help="e.g., postgresql+psycopg2://user:pass@host:5432/db")
            tables_input = st.sidebar.text_input("Tables (comma-separated, optional)", value="")
            if tables_input.strip():
                selected_tables = tuple([t.strip() for t in tables_input.split(",") if t.strip()])

        try:
            if db_type == "SQLite (file)":
                tools = [setup_sql_tool(db_path=db_path_input, tables=selected_tables)]
                set_active_db_config(db_type, sqlite_path=db_path_input, tables=selected_tables)
            else:
                tools = [setup_sql_tool(engine_url=engine_url if engine_url else None, tables=selected_tables)]
                set_active_db_config(db_type, engine_url=engine_url if engine_url else None, tables=selected_tables)
        except Exception as e:
            st.sidebar.error(f"Failed to connect to database: {e}")
            tools = []

        if st.session_state.file_uploaded:
            temp_dir = getattr(st.session_state, "temp_dir", None)
            if not temp_dir or not os.path.isdir(temp_dir):
                st.warning("Uploaded documents are unavailable. Please upload them again to rebuild the document index.")
            elif not any(os.path.isfile(os.path.join(temp_dir, name)) for name in os.listdir(temp_dir)):
                st.warning("No files found in the upload directory. Please upload at least one document.")
                st.session_state.file_uploaded = False
            else:
                progress_placeholder = st.empty()
                progress_placeholder.info(
                    "Processing documents… this runs Docling OCR, embeddings, and Milvus indexing. Please wait."
                )
                try:
                    document_tool = setup_document_tool(
                        file_dir=temp_dir,
                        session_id=str(st.session_state.id),
                        use_milvus=use_milvus,
                        progress_callback=lambda msg: progress_placeholder.info(msg),
                    )
                except Exception as doc_err:
                    progress_placeholder.error(f"Failed to process documents: {doc_err}")
                    # force re-upload on next run
                    st.session_state.file_uploaded = False
                else:
                    progress_placeholder.success(
                        "Documents processed successfully! Reusing cached embeddings if available."
                    )
                    st.session_state.file_cache[f"{st.session_state.id}-documents"] = document_tool
                    tools.append(document_tool)

        # Initialize or update workflow with tools
        if not st.session_state.workflow or st.session_state.workflow_needs_update:
            workflow = initialize_workflow(tools)
            st.session_state.workflow_needs_update = False
        else:
            workflow = st.session_state.workflow

        # Chat interface
        if workflow:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            pending_query = None
            if "pending_query" in st.session_state:
                pending_query = st.session_state.pop("pending_query")

            query = pending_query or st.chat_input("Ask your question...")

            # Process query if submitted
            if query:
                st.session_state.messages.append({"role": "user", "content": query})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(query)

                # Process and show assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")

                    # Process the query
                    response = asyncio.run(process_query(query, workflow))

                    # Initialize displayed response
                    displayed_response = ""

                    # Check if this is already a formatted response
                    if "**🔧 Tool Used:**" not in response:
                        # No tool call was used; show raw assistant response without fake trust score
                        response = f"**📝 Response:**\n\n{response}\n\n"

                    # Display final response without streaming cursor (preserve formatting)
                    message_placeholder.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
    else:
        # Application information if requirements not met
        if not st.session_state.openrouter_api_key:
            st.error("OpenRouter API Key is required to use this application.")
            st.markdown(
                """
                ### Getting Started
                1. Enter your OpenRouter API Key in the sidebar
                2. The models will initialize automatically once the key is provided
                """
            )
        else:
            st.error("Models could not be initialized. Please check your API key.")
            st.markdown(
                """
                ### Troubleshooting
                1. Verify your OpenRouter API key is correct
                2. Try refreshing the application
                """
            )


if __name__ == "__main__":
    # Clean up any temporary directories on exit
    if hasattr(st.session_state, "temp_dir") and os.path.exists(
        st.session_state.temp_dir
    ):
        import shutil
        try:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        except (OSError, FileNotFoundError) as e:
            print(f"Failed to clean up temp directory: {e}")

    # Run the main Streamlit app
    main()
