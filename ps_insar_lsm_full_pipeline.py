#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
1️⃣ Load Landslide Sample Points and Rasters
python
Copy code
points = gpd.read_file("data/samples/landslide_points.shp")
labels = points["label"].values

features = [
    'grid_code','curvclass','Elevclass','Ndwiclass','Aspectclas',
    'Disfaultcl','Disroadcla','Drainclass','Lulcclass','NDVIclass',
    'Pgaclass','Popclass','Rainclass','Rdlsclass','Relclass1',
    'Slopeclass','Spiclass','Sticlass','tpiclass','Triclass',
    'Twiclass','Disrivercl'
]

raster_files = {feat: f"data/rasters/{feat}.tif" for feat in features}
rasters = {name: rasterio.open(path) for name, path in raster_files.items()}

points = points.to_crs(rasters[features[0]].crs)
coords = [(geom.x, geom.y) for geom in points.geometry]

X = []
for feat in features:
    raster = rasters[feat]
    vals = np.array([v[0] for v in raster.sample(coords)])
    X.append(vals)

X = np.vstack(X).T
df = pd.DataFrame(X, columns=features)
df['label'] = labels
df = df.dropna()

X = df.drop(columns=['label']).values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
2️⃣ Deep Learning Models
python
Copy code
def build_dnn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bpnn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(input_dim):
    model = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(input_dim,1)),
        Dropout(0.3),
        SimpleRNN(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_dim):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_dim,1)),
        Dropout(0.4),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
3️⃣ Hybrid Ensemble Machine Learning
python
Copy code
voting_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('svc', SVC(probability=True))
    ], voting='soft'
)

stacking_model = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier()),
        ('gnb', GaussianNB()),
        ('svc', SVC(probability=True))
    ],
    final_estimator=RandomForestClassifier(n_estimators=200),
    cv=5
)

xgb_model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4)
ada_model = RandomForestClassifier(n_estimators=200, max_depth=10)

meta_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=5)),
        ('lgb', LGBMClassifier(n_estimators=50)),
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
4️⃣ Pearson Correlation Matrix
python
Copy code
correlation_matrix = df[features].corr()
plt.figure(figsize=(35, 28))

ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True,
    linewidths=3,
    annot_kws={"size":30, "fontfamily": "Times New Roman"},
    cbar_kws={
        "shrink":1,   # scale the colorbar (80% of original size)
        "pad": 0.02      # reduce the distance between heatmap and colorbar
    }
)

# Colorbar font settings
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=35)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontname("Times New Roman")

# Axis labels
plt.xticks(rotation=45, ha='right', fontsize=30, fontfamily="Times New Roman")
plt.yticks(fontsize=30, fontfamily="Times New Roman")

plt.tight_layout()
plt.show()
5️⃣ Multicollinearity Assessment (VIF and Tolerance)
python
Copy code
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
vif_df = pd.DataFrame()
vif_df['Feature'] = X_scaled_df.columns
vif_df['VIF'] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]
vif_df['Tolerance'] = 1 / vif_df['VIF']
print(vif_df)
# --- VIF and Tolerance Data ---
data = {
    "Feature": [
        "Slope","Geology","SPI","STI","Distance to river","TWI","TRI","TPI",
        "Rainfall","Population density","PGA","LULC","Elevation",
        "Drainage density","Distance to road","Distance to fault",
        "Curvature","Aspect"
    ],
    "VIF": [
        2.219318,1.196339,1.110284,1.014331,1.742798,1.601761,1.678034,2.322200,1.331283,1.040450,2.937724,1.159976,3.547935,
        1.946141,2.342689,2.271639,1.237192,1.087582
    ],
    "Tolerance": [
        0.450589,0.835883,0.900671,0.985872,0.573790,0.624313,0.595935,0.430626,
        0.751155,0.961122,0.340400,0.862086,0.281854,
        0.513837,0.426860,0.440211,0.808282,0.919471
    ]
}

df = pd.DataFrame(data)
df = df.sort_values("VIF", ascending=True).reset_index(drop=True)

# ----------------------------------------
#            PLOTTING SECTION
# ----------------------------------------
fig, ax = plt.subplots(figsize=(45, 42))  # Large figure for presentations

# Horizontal bar positions
y = np.arange(len(df)) * 1.3  # spacing between bars

# Plot bars
bars1 = ax.barh(y - 0.30, df["VIF"], height=0.57,color="#FF6F61", label="VIF")
bars2 = ax.barh(y + 0.30, df["Tolerance"], height=0.57,color="#6BAED6", label="Tolerance")

# Set x-limits with small space at end
max_val = max(df["VIF"].max(), df["Tolerance"].max())
ax.set_xlim(0, max_val * 1.03)  # small space at right

# Set y-limits with small space at top and bottom
ax.set_ylim(y[0] - 1, y[-1] + 1)

# Remove default margins
ax.margins(x=0, y=0)

# ----------------------------------------
#          ADD VALUES TO BARS
# ----------------------------------------
for bar in bars1:
    ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.2f}", fontsize=40, va="center", fontfamily="Times New Roman")

for bar in bars2:
    ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.2f}", fontsize=40, va="center", fontfamily="Times New Roman")

# ----------------------------------------
#        Y-AXIS & X-AXIS FORMATTING
# ----------------------------------------
ax.set_yticks(y)
ax.set_yticklabels(df["Feature"], fontsize=40, fontfamily="Times New Roman")
ax.tick_params(axis='y', pad=20)
ax.tick_params(axis='x', labelsize=40)

ax.set_xlabel("VIF / Tolerance Values", fontsize=38, fontfamily="Times New Roman")
ax.set_title("Multicollinearity Assessment (VIF and Tolerance)", fontsize=55,
             fontfamily="Times New Roman", pad=30)

# Legend inside bottom-right
ax.legend(loc='lower right', fontsize=55, frameon=True, ncol=1)

# ----------------------------------------
#        IMPROVE FRAME / SPINES
# ----------------------------------------
for spine in ax.spines.values():
    spine.set_linewidth(5)       # thicker frame
    spine.set_color('black')     # black color

plt.tight_layout()
plt.show()
6️⃣ SHAP Feature Importance
python
Copy code
explainer = shap.TreeExplainer(ada_model)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values[1], X_scaled, feature_names=features)
# Generate SHAP summary plot without showing
shap.summary_plot(shap_values, X, plot_size=(15, 10), show=False)

# Get current axes
ax = plt.gca()

# 🔠 Increase y-axis (feature names) font size
for label in ax.get_yticklabels():
    label.set_fontsize(20)

# 🔠 Increase x-axis tick font size
for label in ax.get_xticklabels():
    label.set_fontsize(20)

# 🏷️ Increase x-axis label font size
ax.set_xlabel("SHAP value (impact on model output)", fontsize=30)

# 📊 Increase color bar font size
cbar = plt.gcf().axes[-1]  # the last axis is the color bar
cbar.tick_params(labelsize=20)
cbar.set_ylabel("Feature value", fontsize=20)

# 🖼️ Add a plot title
plt.title("Landslide Factor Importance", fontsize=30)

# 🧹 Clean layout
plt.tight_layout()
plt.show()
7️⃣ 10-Fold Stratified CV with ROC–AUC
python
Copy code
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "Kappa": [], "ROC-AUC": []}
roc_curves = []

for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob>0.5).astype(int)

    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1"].append(f1_score(y_test, y_pred))
    metrics["Kappa"].append(cohen_kappa_score(y_test, y_pred))
    metrics["ROC-AUC"].append(roc_auc_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves.append((fpr, tpr))

# Print mean CV metrics
print("10-Fold Stratified CV Results:")
for k,v in metrics.items():
    print(f"{k}: {np.mean(v):.3f}")
8️⃣ ROC Plot
python
Copy code
plt.figure(figsize=(10,8))
for fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, color='lightgray', lw=1, alpha=0.5)
plt.plot([0,1],[0,1], linestyle='--', color='black')
plt.title(f'Random Forest ROC Curves (mean AUC = {np.mean(metrics["ROC-AUC"]):.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
9️⃣ Generate Final Raster-Based Susceptibility Map with PS-InSAR Correction
python
Copy code
template = rasters[features[0]]
profile = template.profile
profile.update(dtype=rasterio.float32, count=1)

stack = np.stack([r.read(1) for r in rasters.values()])
flat_stack = stack.reshape(stack.shape[0], -1).T
flat_scaled = scaler.transform(flat_stack)

sus_prob = model.predict_proba(flat_scaled)[:,1]

# PS-InSAR deformation correction
ps_idx = features.index('Disrivercl')  # replace with Vslope raster if applicable
vslope_flat = flat_stack[:, ps_idx]
sus_prob[vslope_flat>0.005] += 0.1
sus_prob = np.clip(sus_prob, 0, 1)

sus_map = sus_prob.reshape(template.shape)
with rasterio.open("lsm_corrected.tif", "w", **profile) as dst:
    dst.write(sus_map,1)

print("Final PS-InSAR corrected Landslide Susceptibility Map saved as 'lsm_corrected.tif'")

