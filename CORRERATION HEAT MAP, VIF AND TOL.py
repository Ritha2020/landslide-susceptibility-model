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


df = df.drop(['FID','Shape','Area_m2','ength_m','X_coord','Y_coord','landslides'],axis=1)


# In[5]:


df


# In[6]:


# Compute Pearson correlation
correlation_matrix = df.corr(method='pearson')


# In[7]:


# Print the matrix
print("📊 Pearson Correlation Matrix:\n")
print(correlation_matrix)


# In[9]:


# Plot the correlation heatmap
plt.figure(figsize=(40,30))
# Assign heatmap to ax
ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True,
    linewidths=3,
    annot_kws={"size":28, "fontfamily": "Times New Roman"},
    cbar_kws={
        "shrink":1,   # scale the colorbar (80% of original size)
        "pad": 0.02      # reduce the distance between heatmap and colorbar
    }
)

# Get the colorbar and set font size and font family
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=35, labelrotation=0)  # Size of ticks
for tick in cbar.ax.get_yticklabels():
    tick.set_fontname("Times New Roman")  # Set Times New Roman

# Align x-axis and y-axis labels
plt.xticks(rotation=45, ha='right', fontsize=30, fontfamily="Times New Roman")
plt.yticks(fontsize=30, fontfamily="Times New Roman")

plt.tight_layout()
plt.show()


# In[16]:


get_ipython().system('pip install statsmodels')


# In[17]:


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


# In[18]:


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[19]:


# Create DataFrame for VIF
X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)


# In[20]:


# Calculate VIF and Tolerance
vif_df = pd.DataFrame()
vif_df['Feature'] = X_scaled_df.columns
vif_df['VIF'] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]
vif_df['Tolerance'] = 1 / vif_df['VIF']


# In[21]:


# Display results
print("📊 Multicollinearity Assessment (VIF and Tolerance):")
print(vif_df)


# In[23]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[26]:


import pandas as pd

data = {
    "Feature": [
        "Slope","Geology","SPI","Distance to river","TWI","TRI","TPI","RDLS",
        "Rainfall","Population density","PGA","LULC","Elevation",
        "Drainage density","Distance to road","Distance to fault",
        "Curvature","Aspect"
    ],
    "VIF": [
        2.219318,1.196339,1.110284,1.742798,1.601761,1.678034,2.322200,
        2.084887,1.331283,1.040450,2.937724,1.159976,3.547935,
        1.946141,2.342689,2.271639,1.237192,1.087582
    ],
    "Tolerance": [
        0.450589,0.835883,0.900671,0.573790,0.624313,0.595935,0.430626,
        0.479642,0.751155,0.961122,0.340400,0.862086,0.281854,
        0.513837,0.426860,0.440211,0.808282,0.919471
    ]
}

df = pd.DataFrame(data)

# Sort by VIF (optional)
df = df.sort_values("VIF", ascending=True)

print(df)


# In[27]:


# Create DataFrame and sort by VIF (optional)
df = pd.DataFrame(data)
df = df.sort_values("VIF", ascending=True)


# In[28]:


# Set y-axis positions
y = np.arange(len(df))*4
bar_height = 1.3
gap = 0.7


# In[98]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- VIF and Tolerance Data ---
data = {
    "Feature": [
        "Slope","Geology","SPI","Distance to river","TWI","TRI","TPI","RDLS",
        "Rainfall","Population density","PGA","LULC","Elevation",
        "Drainage density","Distance to road","Distance to fault",
        "Curvature","Aspect"
    ],
    "VIF": [
        2.219318,1.196339,1.110284,1.742798,1.601761,1.678034,2.322200,
        2.084887,1.331283,1.040450,2.937724,1.159976,3.547935,
        1.946141,2.342689,2.271639,1.237192,1.087582
    ],
    "Tolerance": [
        0.450589,0.835883,0.900671,0.573790,0.624313,0.595935,0.430626,
        0.479642,0.751155,0.961122,0.340400,0.862086,0.281854,
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


# In[ ]:





# In[ ]:




