#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shapefile  # pyshp


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[4]:


import shapefile  # pyshp
import pandas as pd

# Path to your shapefile (only the .shp is needed — the others must be in the same folder)
shapefile_path = r"G:\SECOND PAPER\study area\FOR LSM\LSMR1.shp"  # Note the r before the string (raw string)

# Read shapefile
sf = shapefile.Reader(shapefile_path)

# Extract field names (skip DeletionFlag)
fields = [field[0] for field in sf.fields[1:]]

# Extract records and shapes
records = sf.records()
shapes = sf.shapes()

# Combine into list of dictionaries
data = []
for record, shape in zip(records, shapes):
    row = dict(zip(fields, record))
    row['geometry'] = shape.__geo_interface__  # Store geometry as GeoJSON-like dict
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Print first few rows
print(df.head())


# In[5]:


print(df.columns.tolist())


# In[6]:


df=df[df !=-9999]
df=df.dropna()


# In[7]:


df_model=df.drop(labels=["pointid","grid_code","geometry"],axis=1)
df_model


# In[8]:


inv=pd.read_csv(r"G:\SECOND PAPER\TRAINING AND TESTING DATA\LANDSLIDE POINTS\csv\lanaslide points.csv")
inv


# In[9]:


inv.isin([-9999]).sum()


# In[10]:


inv=inv[inv !=-9999]
inv=inv.dropna()


# In[11]:


Y=inv["landslides"].values
X=inv.drop(labels=["landslides"],axis=1,)
from sklearn.preprocessing import normalize


# In[12]:


features_list=list(X.columns)
# Only needed if the data was not normalized
#X = normalize(X, axis=1)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
x_train_val, x_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.3,shuffle=True, random_state=42)


# In[13]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[14]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2']
}


# In[15]:


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=2, verbose=2)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier( max_depth=10, n_estimators=300, min_samples_leaf=2,  max_features='log2', min_samples_split=10)


# In[17]:


# Identify non-numeric columns
non_numeric_cols = x_train_val.select_dtypes(include=['object']).columns
print("Dropping non-numeric columns:", list(non_numeric_cols))

# Drop those columns before conversion
x_train_val_numeric = x_train_val.drop(columns=non_numeric_cols)

# Convert only numeric columns to float32
x_train_val_numeric = x_train_val_numeric.astype('float32')


# In[18]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ------------------------------------------------------------------
# 1)  KEEP ONLY NUMERIC COLUMNS
# ------------------------------------------------------------------
non_numeric_cols = x_train_val.select_dtypes(exclude=[np.number]).columns
print("Dropping non‑numeric columns:", list(non_numeric_cols))

x_train_val_num = x_train_val.drop(columns=non_numeric_cols).astype('float32')

# ------------------------------------------------------------------
# 2)  RANDOM‑FOREST + VERY SAFE GRID SEARCH
#     * single‑threaded everywhere  (n_jobs = 1)
#     * small grid (2 × 2 × 2 = 8 candidates)
#     * 3‑fold CV  (3 × 8 = 24 fits total)
# ------------------------------------------------------------------
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1        #  NO parallelism inside each forest
)

param_grid = {
    'n_estimators'    : [120, 200],
    'max_depth'       : [10, 20],
    'max_features'    : ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator   = rf,
    param_grid  = param_grid,
    cv          = 3,         # fewer folds → less RAM
    n_jobs      = 1,         # NO parallel fits
    verbose     = 2,
    error_score = 'raise'    # easier debugging
)

# ------------------------------------------------------------------
# 3)  FIT
# ------------------------------------------------------------------
grid_search.fit(x_train_val_num, y_train_val)

print("\n✅  Best parameters:", grid_search.best_params_)
print("✅  Best CV score :", grid_search.best_score_)


# In[19]:


best_rf = RandomForestClassifier(
    max_depth=10,
    max_features='sqrt',
    n_estimators=120,
    random_state=42,
    n_jobs=1
)
best_rf.fit(x_train_val_num, y_train_val)


# In[20]:


# Identify and drop the same non-numeric columns as before
x_test_num = x_test.drop(columns=non_numeric_cols).astype('float32')


# In[21]:


print(len(best_rf.feature_importances_))  # Should match number of features used for training
print(len(features_list))                  # Should match columns used in x_train_val_num
print(x_train_val_num.columns.tolist())   # Confirm the actual features used


# In[22]:


features_list = x_train_val_num.columns.tolist()


# In[23]:


feature_imp = pd.Series(best_rf.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)


# In[24]:


from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Predict labels and probabilities
y_pred_rf = best_rf.predict(x_test_num)
y_pred_prob_rf = best_rf.predict_proba(x_test_num)[:, 1]

# Print metrics
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob_rf))

# Compute ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rf)


# In[25]:


import pandas as pd

print(pd.Series(y_test).value_counts())


# In[26]:


from sklearn.ensemble import RandomForestClassifier

best_rf = RandomForestClassifier(
    max_depth=10,
    max_features='sqrt',
    n_estimators=120,
    class_weight='balanced',  # 🔑 important!
    random_state=42,
    n_jobs=1
)
best_rf.fit(x_train_val_num, y_train_val)


# In[27]:


y_pred_rf = best_rf.predict(x_test_num)
y_pred_prob_rf = best_rf.predict_proba(x_test_num)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred_rf, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob_rf))


# In[28]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='auto', random_state=42)
x_train_bal, y_train_bal = sm.fit_resample(x_train_val_num, y_train_val)


# In[29]:


best_rf.fit(x_train_bal, y_train_bal)


# In[30]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    best_rf, x_train_val_num, y_train_val,
    scoring='roc_auc', cv=cv
)

print("Cross-validated AUC-ROC scores:", scores)
print("Mean AUC-ROC:", scores.mean())


# In[31]:


from imblearn.pipeline import Pipeline  # ✅ Use imblearn's pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Pipeline with SMOTE + Scaling + Random Forest
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),  # ✅ Oversample minority class
    ('scaler', StandardScaler()),       # ✅ Normalize features
    ('rf', RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        max_features='sqrt',
        random_state=42
    ))
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#


# In[32]:


# Train on entire training data
pipeline.fit(x_train_val_num, y_train_val)

# Predict on test
y_pred_rf = pipeline.predict(x_test_num)
y_pred_prob_rf = pipeline.predict_proba(x_test_num)[:, 1]

# Metrics
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("✅ AUC-ROC on test set:", roc_auc_score(y_test, y_pred_prob_rf))


# In[33]:


from lightgbm import LGBMClassifier

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('lgbm', LGBMClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ))
])


# In[34]:


print(x_train_val_num.columns)


# In[35]:


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rf)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

print("🎯 Optimal threshold:", optimal_threshold)

# Apply optimal threshold
y_pred_optimized = (y_pred_prob_rf >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimized, zero_division=0))


# In[36]:


get_ipython().system('pip install imbalanced-learn')


# In[37]:


pip install imbalanced-learn lightgbm matplotlib scikit-learn


# In[38]:


from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# 1. Compute FPR, TPR, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rf)

# 2. Compute optimal threshold (Youden’s J statistic)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]
print("🎯 Optimal threshold:", optimal_threshold)

# 3. Apply optimal threshold
y_pred_optimized = (y_pred_prob_rf >= optimal_threshold).astype(int)

# 4. Print classification report using optimized threshold
print(classification_report(y_test, y_pred_optimized, zero_division=0))

# 5. Compute AUC-ROC score
auc_score = roc_auc_score(y_test, y_pred_prob_rf)
print("✅ AUC–ROC score:", auc_score)

# 6. Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc_score:.3f}")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Curve (LightGBM + SMOTE)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[39]:


get_ipython().system('pip install scikeras')


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam

# 1. Generate example data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. Prepare sequence data (reshape for RNN/LSTM): (samples, timesteps, features)
X_seq = X.reshape(X.shape[0], X.shape[1], 1)

# 3. Train-test split for both X and X_seq, keeping indices aligned
X_train, X_test, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    X, X_seq, y, test_size=0.2, random_state=42
)

# --------- Model definitions ---------

def create_dnn():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    return model

def create_rnn():
    model = Sequential([
        SimpleRNN(32, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    return model

def create_lstm():
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    return model

def create_bpnn():
    model = Sequential([
        Dense(50, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    return model

# --------- Wrap models ---------

dnn = KerasClassifier(model=create_dnn, epochs=10, batch_size=32, verbose=0)
rnn = KerasClassifier(model=create_rnn, epochs=10, batch_size=32, verbose=0)
lstm = KerasClassifier(model=create_lstm, epochs=10, batch_size=32, verbose=0)
bpnn = KerasClassifier(model=create_bpnn, epochs=10, batch_size=32, verbose=0)

# --------- Classical ML models ---------

rf = RandomForestClassifier(n_estimators=100, random_state=42)
ada = AdaBoostClassifier(random_state=42)
logreg = LogisticRegression(max_iter=1000)

# Voting ensemble
voting = VotingClassifier(
    estimators=[('rf', rf), ('ada', ada), ('logreg', logreg)],
    voting='soft'
)

# Stacking ensemble
stacking = StackingClassifier(
    estimators=[('rf', rf), ('ada', ada), ('logreg', logreg)],
    final_estimator=LogisticRegression(),
    cv=5
)

# Meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# --------- Train models ---------

print("Training DNN...")
dnn.fit(X_train, y_train)
print("Training RNN...")
rnn.fit(X_train_seq, y_train)
print("Training LSTM...")
lstm.fit(X_train_seq, y_train)
print("Training BPNN...")
bpnn.fit(X_train, y_train)
print("Training Voting ensemble...")
voting.fit(X_train, y_train)
print("Training Boosting (AdaBoost)...")
ada.fit(X_train, y_train)
print("Training Stacking ensemble...")
stacking.fit(X_train, y_train)
print("Training Meta-Learner...")
meta_learner.fit(X_train, y_train)

# --------- Predict & plot ROC ---------

models = {
    "DNN": (dnn, X_test),
    "RNN": (rnn, X_test_seq),
    "LSTM": (lstm, X_test_seq),
    "BPNN": (bpnn, X_test),
    "Voting": (voting, X_test),
    "Boosting (AdaBoost)": (ada, X_test),
    "Stacking": (stacking, X_test),
    "Meta-Learner (LogReg)": (meta_learner, X_test),
}

plt.figure(figsize=(10, 8))

for name, (model, X_data) in models.items():
    y_pred_proba = model.predict_proba(X_data)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Multiple Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# In[41]:


get_ipython().system('pip install scikeras')


# In[42]:


print(df.columns.tolist())


# In[43]:


df


# In[44]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, StackingClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

# === 1. Load & Clean Data ===
# List all features EXCEPT non-numeric or irrelevant columns
features = [
    'reclassdis', 'reclassslo', 'reclass_tw', 'reclass_tp',
    'reclass_sp', 'reclass_pg', 'reclass_lu', 'reclass__1',
    'reclass_dr', 'reclass_di', 'reclass__2', 'reclass_cu', 'reclass_as'
]

# Clean up: ensure features exist and are numeric
X = df[features].apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# === 2. Define Labels ===
# Replace this line with your actual label column if available, e.g., df['landslide_class']
y = np.random.randint(0, 2, size=len(X))

# === 3. Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RNN/LSTM expect 3D input: (samples, timesteps, features)
X_rnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# === 4. Build Models ===
input_dim = X.shape[1]

def build_dnn():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_rnn():
    model = Sequential([
        Input(shape=(input_dim, 1)),
        SimpleRNN(64, return_sequences=True, dropout=0.3),
        BatchNormalization(),
        SimpleRNN(32, dropout=0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_lstm():
    model = Sequential([
        Input(shape=(input_dim, 1)),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_bpnn():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    return model

# === 5. Wrap with scikeras ===
dnn = KerasClassifier(model=build_dnn, epochs=10, batch_size=32, verbose=0)
rnn = KerasClassifier(model=build_rnn, epochs=10, batch_size=32, verbose=0)
lstm = KerasClassifier(model=build_lstm, epochs=10, batch_size=32, verbose=0)
bpnn = KerasClassifier(model=build_bpnn, epochs=10, batch_size=32, verbose=0)

# === 6. Fit Models ===
dnn.fit(X_scaled, y)
bpnn.fit(X_scaled, y)
rnn.fit(X_rnn, y)
lstm.fit(X_rnn, y)

# === 7. Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb_clf.fit(X_scaled, y)

# === 8. Voting
voting_clf = VotingClassifier(estimators=[
    ('dnn', dnn), ('bpnn', bpnn)
], voting='soft')
voting_clf.fit(X_scaled, y)

# === 9. Stacking
stacking_clf = StackingClassifier(
    estimators=[('dnn', dnn), ('bpnn', bpnn), ('gb', gb_clf)],
    final_estimator=LogisticRegression(),
    passthrough=True
)
stacking_clf.fit(X_scaled, y)

# === 10. Meta-learner
meta_X = pd.DataFrame({
    'dnn': dnn.predict_proba(X_scaled)[:, 1],
    'rnn': rnn.predict_proba(X_rnn)[:, 1],
    'lstm': lstm.predict_proba(X_rnn)[:, 1],
    'bpnn': bpnn.predict_proba(X_scaled)[:, 1],
    'voting': voting_clf.predict_proba(X_scaled)[:, 1],
    'boosting': gb_clf.predict_proba(X_scaled)[:, 1],
    'stacking': stacking_clf.predict_proba(X_scaled)[:, 1]
})

meta_learner = LogisticRegression()
meta_learner.fit(meta_X, y)

# === 11. Append Predictions to df ===
df['LSI_dnn'] = meta_X['dnn']
df['LSI_rnn'] = meta_X['rnn']
df['LSI_lstm'] = meta_X['lstm']
df['LSI_bpnn'] = meta_X['bpnn']
df['LSI_voting'] = meta_X['voting']
df['LSI_boosting'] = meta_X['boosting']
df['LSI_stacking'] = meta_X['stacking']
df['LSI_meta'] = meta_learner.predict_proba(meta_X)[:, 1]

# === 12. Save Results ===
df.to_csv("LSI_full_results.csv", index=False)
print("✅ LSI predictions saved for all models!")


# In[45]:


# Show just the LSI predictions
df[["LSI_dnn", "LSI_rnn", "LSI_lstm","LSI_bpnn","LSI_voting","LSI_boosting","LSI_stacking","LSI_meta"]].head()


# In[46]:


df.head()


# In[48]:


df.sort_values(by="LSI_dnn", ascending=False).head(705818)


# In[49]:


df[["LSI_dnn", "LSI_rnn", "LSI_lstm","LSI_bpnn","LSI_voting","LSI_boosting","LSI_stacking","LSI_meta"]].describe()


# In[50]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.hist(df["LSI_dnn"], bins=30, alpha=0.6, label='DNN')
plt.hist(df["LSI_rnn"], bins=30, alpha=0.6, label='RNN')
plt.hist(df["LSI_lstm"], bins=30, alpha=0.6, label='LSTM')
plt.hist(df["LSI_bpnn"], bins=30, alpha=0.6, label='BPNN')
plt.hist(df["LSI_voting"], bins=30, alpha=0.6, label='VOTING')
plt.hist(df["LSI_boosting"], bins=30, alpha=0.6, label='BOOSTING')
plt.hist(df["LSI_stacking"], bins=30, alpha=0.6, label='STACKING')
plt.hist(df["LSI_meta"], bins=30, alpha=0.6, label='META')
plt.xlabel("LSI Value")
plt.ylabel("Frequency")
plt.title("LSI Prediction Distribution")
plt.legend()
plt.grid(True)
plt.show()


# In[51]:


print(df.columns)


# In[53]:


print(df.columns)


# In[54]:


import numpy as np

num_points = len(df)
grid_size = int(np.ceil(np.sqrt(num_points)))  # Ensure it's enough to cover all rows

# Generate grid coordinates
x_vals = np.tile(np.arange(grid_size), grid_size)
y_vals = np.repeat(np.arange(grid_size), grid_size)

# Trim to match DataFrame length
df['X'] = x_vals[:num_points]
df['Y'] = y_vals[:num_points]


# In[55]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sc = plt.scatter(df['X'], df['Y'], c=df['LSI_dnn'], cmap='YlOrRd', s=30)
plt.colorbar(sc, label='LSI (DNN)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Landslide Susceptibility Index (DNN-based)')
plt.grid(True)
plt.show()


# In[56]:


def classify_lsi(value):
    if value <= 0.33:
        return 'Low'
    elif value <= 0.66:
        return 'Medium'
    else:
        return 'High'

df['LSI_zone'] = df['LSI_dnn'].apply(classify_lsi)


# In[57]:


df


# In[58]:


df = df.drop(['X','Y'],axis=1)


# In[59]:


df


# In[60]:


import pandas as pd
import shapefile  # pyshp
from pyproj import CRS
import zipfile, glob
import os

# Step 1: Assume df is already in memory (if not, load your data)
# df = pd.read_csv("your_lsi_data.csv")

# Step 2: Rename coordinate columns for consistency
df = df.rename(columns={"X_coord": "x", "Y_coord": "y"})

# ✅ Optional: If your file is massive, filter or sample
# df = df.sample(n=5000000, random_state=42)  # example: limit to 5 million rows

# Step 3: Set output shapefile name
shapefile_name = "LSI_DNN"

# Step 4: Create shapefile writer with POINT geometry
w = shapefile.Writer(shapefile_name, shapefile.POINT)
w.autoBalance = 1

# Step 5: Define fields (minimize size to reduce file size)
w.field("LSI_DNN", "F", decimal=4)     # reduce decimal precision
w.field("ZONE", "C", size=8)           # reduce zone string length

# Step 6: Write each point and attributes
for _, row in df.iterrows():
    try:
        w.point(row["x"], row["y"])
        w.record(round(float(row["LSI_dnn"]), 4), str(row["LSI_zone"])[:8])
    except:
        continue  # Skip invalid points

w.close()

# Step 7: Create PRJ file (CRS — EPSG:32645 = UTM Zone 45N)
with open(f"{shapefile_name}.prj", "w") as f:
    f.write(CRS.from_epsg(32645).to_wkt())

# Step 8: Zip shapefile components using compression
with zipfile.ZipFile(f"{shapefile_name}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for file in glob.glob(f"{shapefile_name}.*"):
        zipf.write(file, os.path.basename(file))

print(f"✅ Zipped shapefile saved as {shapefile_name}.zip")


# In[61]:


import shutil, os

total, used, free = shutil.disk_usage(os.getcwd())
print(f"Total: {total / 1e9:.2f} GB")
print(f"Used:  {used / 1e9:.2f} GB")
print(f"Free:  {free / 1e9:.2f} GB")


# In[62]:


import shutil
shutil.rmtree(os.getenv("TEMP"), ignore_errors=True)


# In[63]:


import shutil, os
free = shutil.disk_usage(os.getcwd()).free / 1e9
print(f"✅ Free space: {free:.2f} GB")


# In[66]:


df_subset = df.head(705818)  # or use df.sample(n=329998) for a random subset


# In[67]:


df = df.rename(columns={'X': 'x', 'Y': 'y'})


# In[68]:


import pandas as pd
import shapefile

# ✅ Load your DataFrame here if not already loaded
# df = pd.read_csv(...)  # or pd.read_excel(...) or geopandas.read_file(...) etc.

# ✅ Rename coordinate columns to 'x' and 'y'
df = df.rename(columns={'X': 'x', 'Y': 'y'})


# In[69]:


print(df.columns.tolist())


# In[70]:


# Safely get the first row regardless of index label
first_row = df.iloc[0]

# Print the type of the 'geometry' field
print(type(first_row['geometry']))


# In[71]:


from shapely.geometry import shape
import geopandas as gpd

# Convert dict geometry to Shapely objects
df['geometry'] = df['geometry'].apply(shape)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Extract x and y
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y


# In[72]:


import shapefile
import pandas as pd

models = ["LSI_dnn", "LSI_rnn", "LSI_lstm", "LSI_bpnn",
          "LSI_voting", "LSI_boosting", "LSI_stacking", "LSI_meta"]

errors = {}

for model in models:
    print(f"Processing {model} ...")
    try:
        w = shapefile.Writer(f"{model}_points", shapeType=shapefile.POINT)
        w.autoBalance = 1
        w.field("LSI", "F", decimal=4)
        w.field("ZONE", "C", size=10)

        for _, row in gdf.iterrows():
            if pd.isna(row["x"]) or pd.isna(row["y"]) or pd.isna(row[model]):
                continue
            w.point(row["x"], row["y"])
            zone = row.get("LSI_zone", "Unknown")
            w.record(round(float(row[model]), 4), str(zone))

        w.close()
        print(f"✅ {model} shapefile written successfully")

    except Exception as e:
        print(f"❌ Error for {model}: {e}")
        errors[model] = str(e)

if errors:
    print("\nSummary of errors:")
    for model, err in errors.items():
        print(f"{model}: {err}")
else:
    print("\nAll models processed without errors.")


# In[73]:


import pandas as pd
import shapefile
import zipfile
from pyproj import CRS
import os, tempfile, shutil

# ✅ List of LSI models to export (add/remove as needed)
lsi_models = [
    "LSI_dnn", "LSI_rnn", "LSI_lstm", "LSI_bpnn",
    "LSI_voting", "LSI_boosting", "LSI_stacking", "LSI_meta"
]

zone_col = "LSI_zone"
crs_epsg = 32645  # EPSG for your UTM Zone or use 4326 if in WGS84

# ✅ Rename X and Y for consistency
df = df.rename(columns={"X_coord": "x", "Y_coord": "y"})

# ✅ Export each LSI model as a zipped shapefile
for model_col in lsi_models:
    print(f"⏳ Exporting {model_col}...")

    # Temp directory for shapefile parts
    tmp_dir = tempfile.mkdtemp()
    shp_base = os.path.join(tmp_dir, "LSI_export")

    # Create shapefile writer
    w = shapefile.Writer(shp_base, shapefile.POINT)
    w.autoBalance = 1
    w.field(model_col.upper(), "F", decimal=6)
    w.field("ZONE", "C", size=10)

    for _, row in df.iterrows():
        w.point(row["x"], row["y"])
        w.record(float(row[model_col]), str(row[zone_col]))

    w.close()

    # Write .prj file for CRS
    with open(shp_base + ".prj", "w") as f:
        f.write(CRS.from_epsg(crs_epsg).to_wkt())

    # Zip the shapefile
    zip_name = f"{model_col}.zip"
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for ext in [".shp", ".shx", ".dbf", ".prj"]:
            zf.write(shp_base + ext, arcname=f"{model_col}{ext}")

    # Clean up
    shutil.rmtree(tmp_dir)
    print(f"✅ {zip_name} saved.")

print("🎉 All LSI model shapefiles exported successfully!")


# In[74]:


import numpy as np
from PIL import Image
import os

# Use your DataFrame (assumes 'x', 'y', and LSI_* columns exist)
df = df.copy()

# Output folder
os.makedirs("basic_rasters", exist_ok=True)

# Define resolution
pixel_size = 30

# Calculate bounds
xmin, ymin = df['x'].min(), df['y'].min()
xmax, ymax = df['x'].max(), df['y'].max()

# Raster size
width = int((xmax - xmin) / pixel_size) + 1
height = int((ymax - ymin) / pixel_size) + 1

# For each model
models = ["LSI_dnn", "LSI_rnn", "LSI_lstm", "LSI_bpnn",
          "LSI_voting", "LSI_boosting", "LSI_stacking", "LSI_meta"]

for model in models:
    print(f"Creating raster for {model} ...")
    # Initialize raster with 0 (or np.nan if you want to skip empty)
    raster = np.zeros((height, width), dtype=np.float32)

    for _, row in df.iterrows():
        x = int((row['x'] - xmin) / pixel_size)
        y = int((ymax - row['y']) / pixel_size)
        if 0 <= x < width and 0 <= y < height:
            val = row[model]
            if not np.isnan(val):
                raster[y, x] = val

    # Normalize values to 0–255 for image (if needed)
    norm = 255 * (raster - np.nanmin(raster)) / (np.nanmax(raster) - np.nanmin(raster))
    norm = np.nan_to_num(norm)

    img = Image.fromarray(norm.astype(np.uint8))
    img.save(f"basic_rasters/{model}.tif")  # or .png

    print(f"✅ Saved: basic_rasters/{model}.tif")


# In[ ]:




