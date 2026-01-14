#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from shapely.geometry import Point


# In[ ]:


points = gpd.read_file("data/samples/landslide_points.shp")
labels = points["label"].values

features = [
    'grid_code','curvclass','Elevclass','Ndwiclass','Aspectclas',
    'Disfaultcl','Disroadcla','Drainclass','Lulcclass','NDVIclass',
    'Pgaclass','Popclass','Rainclass','Rdlsclass','Relclass1',
    'Slopeclass','Spiclass','Sticlass','tpiclass','Triclass',
    'Twiclass','Disrivercl'
]


# In[ ]:


def sample_raster_at_points(raster_path, gdf):
    with rasterio.open(raster_path) as src:
        coords = [(x,y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        sampled_values = [val[0] for val in src.sample(coords)]
    return np.array(sampled_values)

# Sample all layers
for var, path in raster_paths.items():
    landslide_gdf[var] = sample_raster_at_points(path, landslide_gdf)


# In[ ]:


def generate_random_points(bounds, n, crs):
    minx, miny, maxx, maxy = bounds
 
# Combine datasets
data = pd.concat([landslide_gdf, non_landslide_gdf])   xs = np.random.uniform(minx, maxx, n)
    ys = np.random.uniform(miny, maxy, n)
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=crs)

non_landslide_gdf = generate_random_points(landslide_gdf.total_bounds, 1000, landslide_gdf.crs)

for var, path in raster_paths.items():
    non_landslide_gdf[var] = sample_raster_at_points(path, non_landslide_gdf)

# Label data
landslide_gdf['label'] = 1
non_landslide_gdf['label'] = 0


# In[ ]:


features = list(raster_paths.keys())
X = data[features]
y = data['label']

model = LogisticRegression()
model.fit(X, y)


# In[ ]:


def predict_raster(raster_paths, model):
    with rasterio.open(raster_paths[features[0]]) as ref:
        profile = ref.profile
        width, height = ref.width, ref.height

    stacked = np.zeros((len(features), height, width))

    for i, var in enumerate(features):
        with rasterio.open(raster_paths[var]) as src:
            stacked[i] = src.read(1)

    X_pred = stacked.reshape(len(features), -1).T
    preds = model.predict_proba(X_pred)[:, 1]
    pred_raster = preds.reshape((height, width))

    return pred_raster, profile

susceptibility, profile = predict_raster(raster_paths, model)

# Save the susceptibility map
with rasterio.open('output/landslide_susceptibility.tif', 'w', **profile) as dst:
    dst.write(susceptibility, 1)


# In[ ]:


plt.imshow(susceptibility, cmap='Reds')
plt.title('Landslide Susceptibility Map')
plt.colorbar(label='Susceptibility Score')
plt.show()


# In[ ]:




