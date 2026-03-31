import pandas as pd
import numpy as np

# This script translates the borg traces csv into a csv readable by LSTM.

df = pd.read_csv('borg_traces_data.csv')

def extract_val(series, key):
    return pd.to_numeric(series.str.extract(f"'{key}': ([\d.e+-]+)")[0], errors='coerce').fillna(0)

print("Parsing 40,000+ rows...")
df['cpu_req'] = extract_val(df['resource_request'], 'cpus')
df['mem_req'] = extract_val(df['resource_request'], 'memory')
df['cpu_avg'] = extract_val(df['average_usage'], 'cpus')
df['mem_avg'] = extract_val(df['average_usage'], 'memory')

cols_to_fix = ['scheduling_class', 'priority', 'failed', 'time']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print("Calculating rolling cluster state...")
df['rolling_cpu_load'] = df['cpu_avg'].rolling(window=50, min_periods=1).mean()
df['rolling_failed_rate'] = df['failed'].rolling(window=50, min_periods=1).sum()

final_columns = [
    'time', 'cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 
    'priority', 'scheduling_class', 'rolling_cpu_load', 
    'rolling_failed_rate', 'failed'
]
processed_df = df[final_columns]

output_file = 'lstm_training_data_40k.csv'
processed_df.to_csv(output_file, index=False)

print(f"Success! Saved {len(processed_df)} lines to {output_file}")