import pandas as pd
import numpy as np

# 1. Load the entire dataset
df = pd.read_csv('borg_traces_data.csv')

# 2. Vectorized Parsing (Fastest way to handle 40k rows)
def extract_val(series, key):
    return pd.to_numeric(series.str.extract(f"'{key}': ([\d.e+-]+)")[0], errors='coerce').fillna(0)

print("Parsing 40,000+ rows...")
df['cpu_req'] = extract_val(df['resource_request'], 'cpus')
df['mem_req'] = extract_val(df['resource_request'], 'memory')
df['cpu_avg'] = extract_val(df['average_usage'], 'cpus')
df['mem_avg'] = extract_val(df['average_usage'], 'memory')

# 3. Clean up existing numeric columns
cols_to_fix = ['scheduling_class', 'priority', 'failed', 'time']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 4. Create "Context" Features (Crucial for 1-to-1 modeling)
# Since we aren't grouping, the LSTM needs to know what the cluster 
# looked like just before this specific job arrived.
print("Calculating rolling cluster state...")
df['rolling_cpu_load'] = df['cpu_avg'].rolling(window=50, min_periods=1).mean()
df['rolling_failed_rate'] = df['failed'].rolling(window=50, min_periods=1).sum()

# 5. Select only the columns needed for the LSTM
# This keeps the file clean but preserves all 40k rows
final_columns = [
    'time', 'cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 
    'priority', 'scheduling_class', 'rolling_cpu_load', 
    'rolling_failed_rate', 'failed'
]
processed_df = df[final_columns]

# 6. Save to a new CSV (This will have ~40k lines)
output_file = 'lstm_training_data_40k.csv'
processed_df.to_csv(output_file, index=False)

print(f"Success! Saved {len(processed_df)} lines to {output_file}")