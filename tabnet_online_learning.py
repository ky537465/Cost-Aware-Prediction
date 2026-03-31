import pandas as pd
import numpy as np
import torch
import ast
import sys
import time
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

torch.cuda.empty_cache() 
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("borg_traces_data.csv")

def safe_eval(x):
    if pd.isna(x) or str(x).strip() in ['[]', '', 'None']:
        return {}
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x if isinstance(x, dict) else {}
    except:
        return {}

complex_cols = ['resource_constraint', 'average_usage', 'maximum', 'sample_rate']
for col in complex_cols:
    if col in df.columns:
        df[col] = df[col].apply(safe_eval)
        res_df = pd.json_normalize(df[col]).add_prefix(f'{col}_')
        df = pd.concat([df.drop(columns=[col]), res_df], axis=1)

df = df.fillna(0)

target = 'failed'
unused_cols = ['time', 'instance_id', 'collection_id']
features = [col for col in df.columns if col not in [target] + unused_cols]

num_fail = df[target].sum()
num_success = len(df) - num_fail
fail_weight = int(num_success / max(num_fail, 1))

cat_idxs = []
cat_dims = []
for i, col in enumerate(features):
    if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_idxs.append(i)
        cat_dims.append(int(df[col].max() + 1))

X = df[features].values.astype(np.float32)
y = df[target].values.astype(int)

clf = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=4,
    n_d=16, n_a=16,
    n_steps=4,
    gamma=1.3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
    device_name=device,
    verbose=0
)

predictions = []
actuals = []
cumulative_accuracies = []
window_size = 1000

X_memory = []
y_memory = []
memory_limit = 2000 

print(f"Starting online learning on {len(X)} samples...")

for i in range(len(X)):
    X_line = X[i].reshape(1, -1)
    y_line = y[i]

    if hasattr(clf, 'network'):
        pred = clf.predict(X_line)
        predictions.append(pred[0])
        actuals.append(y_line)
        
        if i > 0 and i % window_size == 0:
            current_acc = balanced_accuracy_score(actuals[-window_size:], predictions[-window_size:])
            cumulative_accuracies.append(current_acc)
            print(f"Index {i}: Recent Balanced Acc = {current_acc:.4f}")

    X_memory.append(X[i])
    y_memory.append(y[i])

    if len(X_memory) >= 128:
        idx = np.random.choice(len(X_memory), 64, replace=False)
        
        clf.fit(
            X_train=np.array(X_memory)[idx], 
            y_train=np.array(y_memory)[idx],
            max_epochs=1,
            patience=0,
            batch_size=64,
            virtual_batch_size=32,
            drop_last=False,
            compute_importance=False,
            weights=1
        )
        
        if len(X_memory) > memory_limit:
            X_memory.pop(0)
            y_memory.pop(0)

plt.figure(figsize=(12, 6))
plt.plot(range(window_size, (len(cumulative_accuracies)+1)*window_size, window_size), cumulative_accuracies, marker='o', color='#2ca02c')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
plt.title('Borg Job Failure Prediction: Balanced Accuracy Over Time')
plt.xlabel('Jobs Processed')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()