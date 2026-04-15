import pandas as pd
import numpy as np
import torch
import ast
import sys
import time
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

torch.cuda.empty_cache() 

if torch.cuda.is_available():
    device = "cuda"
else:
    sys.exit()

df = pd.read_csv("borg_traces_data.csv")

def clean_bracket_strings(x):
    if str(x).strip() == '[]' or x is None:
        return 0
    return x

df = df.map(clean_bracket_strings)

complex_cols = ['resource_constraint', 'average_usage', 'maximum', 'sample_rate']
for col in complex_cols:
    if col in df.columns:
        def safe_eval(x):
            try:
                if isinstance(x, str) and '{' in x:
                    return ast.literal_eval(x)
                return x if isinstance(x, dict) else {}
            except:
                return {}
        df[col] = df[col].apply(safe_eval)
        res_df = pd.json_normalize(df[col]).add_prefix(f'{col}_')
        df = pd.concat([df.drop(columns=[col]), res_df], axis=1)

df = df.fillna(0)

target = 'failed'
unused_cols = ['time', 'instance_id', 'collection_id']
features = [col for col in df.columns if col not in [target] + unused_cols]

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
    cat_emb_dim=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    mask_type='sparsemax',
    device_name=device,
    verbose=0
)

predictions = []
actuals = []
cumulative_accuracies = []
start_time = time.time()

buffer_size = 64 
X_buffer = []
y_buffer = []

for i in range(len(X)):
    X_line = X[i].reshape(1, -1)
    y_line = y[i]

    if hasattr(clf, 'network'):
        pred = clf.predict(X_line)
        predictions.append(pred[0])
        actuals.append(y_line)
        
        if i > 0 and i % 1000 == 0:
            current_acc = accuracy_score(actuals, predictions)
            cumulative_accuracies.append(current_acc)
    
    X_buffer.append(X[i])
    y_buffer.append(y[i])

    if len(X_buffer) >= buffer_size:
        clf.fit(
            X_train=np.array(X_buffer), 
            y_train=np.array(y_buffer),
            max_epochs=1,
            patience=0,
            batch_size=buffer_size,
            virtual_batch_size=buffer_size,
            drop_last=False,
            compute_importance=False
        )
        X_buffer = []
        y_buffer = []

if cumulative_accuracies:
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(cumulative_accuracies)*1000, 1000), cumulative_accuracies)
    plt.title('TabNet Incremental Job Success Prediction (RTX 5080)')
    plt.xlabel('Jobs Processed')
    plt.ylabel('Predictive Accuracy')
    plt.grid(True, alpha=0.3)
    plt.show()