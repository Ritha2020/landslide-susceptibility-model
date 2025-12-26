#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[3]:


# Load your dataset
df = pd.read_csv(r"G:\SECOND PAPER\TRAINING AND TESTING DATA\LANDSLIDE POINTS\csv\lanaslide points.csv")
print(df)


# In[4]:


print(df.columns.tolist())


# In[5]:


X = df.drop(columns='landslides')
y = df['landslides']


# In[6]:


# Replace 'landslide' with your actual target column name
target_column = 'landslides'  # ← change this if needed


# In[7]:


# Creating independent and dependent variables
X = df.drop(columns='landslides')
y = df['landslides']


# In[8]:


print(df.head())


# In[9]:


df['landslides']


# In[10]:


df = df.drop(['STI','FID','Shape','Area_m2','NDWI', 'NDVI','ength_m','X_coord','Y_coord','RDLS','REL'],axis=1)


# In[11]:


df


# In[12]:


# Drop rows with missing values (optional)
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


from sklearn.model_selection import train_test_split

X = df.drop('landslides', axis=1)  # Features
y = df['landslides']              # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# If you don't have separate features and target, you can split the whole DataFrame
# X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)

# Check the sizes of the resulting datasets
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_curve, auc
)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, InputLayer, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 1. Simulated dataset
X, y = make_classification(n_samples=3000, n_features=20, n_informative=15,
                           n_classes=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_seq = X.reshape((X.shape[0], X.shape[1], 1))  # For RNN/LSTM

# 2. Deep Learning Model Builders
def build_bpnn():
    model = Sequential([
        InputLayer(input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['AUC'])
    return model

def build_dnn():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['AUC'])
    return model

def build_rnn():
    model = Sequential([
        InputLayer(input_shape=(X.shape[1], 1)),
        SimpleRNN(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['AUC'])
    return model

# ✅ Improved LSTM
def build_lstm():
    model = Sequential([
        InputLayer(input_shape=(X.shape[1], 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0005),
                  metrics=['AUC'])
    return model

# 3. Stable Meta-Learner (Stacking) with light learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('lgbm', LGBMClassifier(max_depth=5, n_estimators=50, random_state=42)),
]

final_estimator = LogisticRegression(max_iter=200)

meta_learner = StackingClassifier(
    estimators=base_learners,
    final_estimator=final_estimator,
    passthrough=True,
    cv=5,
    n_jobs=1  # Stability
)

# 4. Define models dictionary
models = {
    "Meta-learner": meta_learner,
    "Boosting": GradientBoostingClassifier(n_estimators=100),
    "Stacking": StackingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(3)),
            ('nb', GaussianNB()),
            ('dt', DecisionTreeClassifier(max_depth=4))
        ],
        final_estimator=LogisticRegression(max_iter=200),
        n_jobs=1
    ),
    "Voting": VotingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(3)),
            ('nb', GaussianNB()),
            ('dt', DecisionTreeClassifier(max_depth=4))
        ],
        voting='soft',
        n_jobs=1
    )
}

# 5. Metric Computation
def compute_metrics(y_true, y_pred_labels, y_pred_proba):
    acc = accuracy_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels)
    rec = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    kappa = cohen_kappa_score(y_true, y_pred_labels)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    return acc, prec, rec, f1, kappa, auc_score, fpr, tpr

# 6. Train & Evaluate All Models
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}
roc_data = {}

# Traditional models
for name, model in models.items():
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for train_idx, test_idx in kf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
    acc, prec, rec, f1, kappa, auc_score, fpr, tpr = compute_metrics(y_true_all, y_pred_all, y_proba_all)
    results[name] = (acc, prec, rec, f1, kappa, auc_score)
    roc_data[name] = (fpr, tpr, auc_score)

# Deep learning models (BPNN, DNN)
for name, builder in [("BPNN", build_bpnn), ("DNN", build_dnn)]:
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for train_idx, test_idx in kf.split(X, y):
        model = builder()
        model.fit(X[train_idx], y[train_idx], epochs=20, verbose=0)
        y_proba = model.predict(X[test_idx]).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
    acc, prec, rec, f1, kappa, auc_score, fpr, tpr = compute_metrics(y_true_all, y_pred_all, y_proba_all)
    results[name] = (acc, prec, rec, f1, kappa, auc_score)
    roc_data[name] = (fpr, tpr, auc_score)

# RNN and Improved LSTM
for name, builder in [("RNN", build_rnn), ("LSTM", build_lstm)]:
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for train_idx, test_idx in kf.split(X_seq, y):
        model = builder()
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_seq[train_idx], y[train_idx], epochs=50, batch_size=32, verbose=0,
                  validation_data=(X_seq[test_idx], y[test_idx]),
                  callbacks=[early_stop])
        y_proba = model.predict(X_seq[test_idx]).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
    acc, prec, rec, f1, kappa, auc_score, fpr, tpr = compute_metrics(y_true_all, y_pred_all, y_proba_all)
    results[name] = (acc, prec, rec, f1, kappa, auc_score)
    roc_data[name] = (fpr, tpr, auc_score)

# 7. Print Results Table
print(f"{'Model':<15} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Kappa':>6} {'AUC':>6}")
print("="*60)
for name, (acc, prec, rec, f1, kappa, auc_score) in results.items():
    print(f"{name:<15} {acc:6.3f} {prec:6.3f} {rec:6.3f} {f1:6.3f} {kappa:6.3f} {auc_score:6.3f}")

# 8. ROC Plot
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, auc_score) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", linewidth=2)
# Diagonal baseline
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

plt.xlabel('1−Specificity', fontsize=30)
plt.ylabel('Sensitivity', fontsize=30)
plt.title('(a) Training dataset', fontsize=30, fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', fontsize=18)
plt.grid(False)
# Enhanced axis border
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color('black')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




