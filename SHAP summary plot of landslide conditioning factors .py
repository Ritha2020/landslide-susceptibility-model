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


import pandas as pd
df = pd.read_csv(r"G:\SECOND PAPER\TRAINING AND TESTING DATA\LANDSLIDE POINTS\csv\lanaslide points.csv")
print(df)


# In[3]:


print(df.columns.tolist())


# In[4]:


X = df.drop(columns='landslides')
y = df['landslides']


# In[5]:


# Replace 'landslide' with your actual target column name
target_column = 'landslides'  # ← change this if needed


# In[6]:


# Creating independent and dependent variables
X = df.drop(columns='landslides')
y = df['landslides']


# In[8]:


print(df.head())


# In[9]:


df['landslides']


# In[10]:


df = df.drop(['FID','Shape','Area_m2','ength_m','NDVI','NDWI','RDLS','REL','X_coord','Y_coord'],axis=1)


# In[11]:


df


# In[ ]:


# Drop rows with missing values (optional)
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


from sklearn.model_selection import train_test_split

X = df.drop('landslides', axis=1)  # Features
y = df['landslides']              # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[17]:


# Split the data: 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# If you don't have separate features and target, you can split the whole DataFrame
# X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)

# Check the sizes of the resulting datasets
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))


# In[18]:


get_ipython().system('pip install shap==0.39.0')
get_ipython().system('pip install numba==0.53.1')
get_ipython().system('pip install ipykernel')


# In[19]:


import shap
import numba
print("✅ SHAP and Numba are working correctly!")


# In[21]:


import shap
import matplotlib.pyplot as plt

# Assuming model and X are defined
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For binary or multiclass classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Choose class 1 SHAP values

# Now shape will match
print(shap_values.shape, X.shape)

# Plot
plt.figure(figsize=(15, 10))
shap.summary_plot(shap_values, X, plot_size=(15, 10), show=False)
plt.title("Landslide Factor Importance", fontsize=30)
plt.tight_layout()
plt.show()


# In[25]:


# -----------------------------------------------------
# 0. IMPORT LIBRARIES
# -----------------------------------------------------
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# -----------------------------------------------------
# 2. STANDARDIZATION
# -----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for SHAP column names
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# -----------------------------------------------------
# 3. TRAIN-TEST SPLIT
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------
# 4. TRAIN MODEL (XGBoost)
# -----------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------------------
# 5. SHAP EXPLAINER
# -----------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# For binary classification → choose class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# -----------------------------------------------------
# 6. SHAP SUMMARY PLOT (DOT PLOT)
# -----------------------------------------------------
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values, X_train, plot_type="dot", show=False)
plt.title("SHAP Summary Plot", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 7. SHAP BAR PLOT (IMPORTANCE)
# -----------------------------------------------------
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 8. SHAP DEPENDENCE PLOT
# -----------------------------------------------------
top_feature = X_train.columns[0]

plt.figure(figsize=(12, 7))
shap.dependence_plot(top_feature, shap_values, X_train, show=False)
plt.title(f"SHAP Dependence Plot — {top_feature}", fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:




