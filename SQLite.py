import sqlite3
import pandas as pd

# Connect to the existing FEMA database
src = sqlite3.connect("/Users/vamsikrishna/Desktop/Misc_text2sql/rag-sql-router/fema_nfip.sqlite")
dst = sqlite3.connect("/Users/vamsikrishna/Desktop/Misc_text2sql/rag-sql-router/fema_nfip_star.sqlite")

# Load the original table
df = pd.read_sql_query("SELECT * FROM nfip", src)

# Dimension: states
dim_state = (df[['state']]
             .dropna()
             .drop_duplicates()
             .reset_index(drop=True))
dim_state['state_id'] = dim_state.index + 1
dim_state.to_sql('dim_state', dst, index=False)

# Dimension: flood zone
dim_zone = (df[['ratedFloodZone']]
            .dropna()
            .drop_duplicates()
            .rename(columns={'ratedFloodZone': 'zone_code'})
            .reset_index(drop=True))
dim_zone['zone_id'] = dim_zone.index + 1
dim_zone.to_sql('dim_flood_zone', dst, index=False)

# Dimension: event (use event or FICO/EDN codes)
dim_event = (df[['floodEvent', 'ficoNumber']]
             .drop_duplicates()
             .rename(columns={'floodEvent': 'event_name', 'ficoNumber': 'fico_code'})
             .reset_index(drop=True))
dim_event['event_id'] = dim_event.index + 1
dim_event.to_sql('dim_event', dst, index=False)

# Dimension: time (year/month)
time_df = (df[['dateOfLoss']]
           .dropna()
           .assign(year=lambda x: x['dateOfLoss'].str.slice(0,4),
                   month=lambda x: x['dateOfLoss'].str.slice(5,7)))
time_df = time_df[['year', 'month']].drop_duplicates().reset_index(drop=True)
time_df['time_id'] = time_df.index + 1
time_df.to_sql('dim_time', dst, index=False)

# Merge back keys into the fact table
fact = (df.merge(dim_state, on='state', how='left')
          .merge(dim_zone.rename(columns={'zone_code': 'ratedFloodZone'}), on='ratedFloodZone', how='left')
          .merge(dim_event.rename(columns={'event_name':'floodEvent','fico_code':'ficoNumber'}), on=['floodEvent','ficoNumber'], how='left')
          .merge(time_df, on=['yearOfLoss','month'], how='left'))

# Keep relevant fact columns
fact = fact[['id', 'state_id', 'zone_id', 'event_id', 'time_id',
             'amountPaidOnBuildingClaim', 'amountPaidOnContentsClaim',
             'netBuildingPaymentAmount', 'netContentsPaymentAmount', 'netIccPaymentAmount',
             'yearOfLoss', 'dateOfLoss', 'ratedFloodZone', 'floodEvent', 'ficoNumber']]
fact.to_sql('nfip_fact_claim', dst, index=False)

src.close()
dst.close()