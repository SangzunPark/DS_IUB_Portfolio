#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from PIL import Image

image_path = "cover_2.png" 
img = Image.open(image_path)

plt.figure(figsize=(15, 15)) 

plt.imshow(img)
plt.axis('off') 
plt.show()


# # Part 1 / EDA

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import sys
import geopandas as gp
import scipy.stats as stats
import plotly.express as px
sys.version


# In[3]:


data = pd.read_csv("immo_data.csv")


# In[4]:


# Check col
data.info()


# In[5]:


# Check nan value
nan_count = data.isnull().sum()
print(nan_count)


# In[6]:


# 1st Features drop based on the characteristic of column
data.drop(columns=['telekomHybridUploadSpeed','telekomUploadSpeed','description','livingSpaceRange','street',
                                 'scoutId','facilities','geo_krs','telekomTvOffer','petsAllowed','pricetrend',
                                 'noRoomsRange','picturecount','houseNumber','streetPlain','firingTypes','interiorQual',
                                 'interiorQual'],inplace=True)


# In[7]:


# 2nd features drop based on nan value
data.drop(columns=['electricityKwhPrice','electricityBasePrice','energyEfficiencyClass','lastRefurbish',
                   'heatingCosts','noParkSpaces','thermalChar',],inplace=True)


# In[8]:


# Check nan value in columns 
nan_count_v2 = data.isnull().sum()
print(nan_count_v2)


# In[9]:


# 3rd features drop low-important columns
data.drop(columns=['yearConstructedRange','numberOfFloors'],inplace=True)


# In[10]:


# Apply replacement value 'Other' to columns that are absolutely necessary but have many nan values
columns_to_fillna = ['condition', 'heatingType', 'typeOfFlat']
for column in columns_to_fillna:
    data[column].fillna('Other', inplace=True)


# In[11]:


# Delete all rows where the main value total rent is nan (do not replace with other statistical values)
data.dropna(subset=['totalRent'],inplace=True)


# In[12]:


# Fill NaN values in floor, yearConstructed, and serviceCharge columns with median values, respectively.
columns_to_fillna = ['floor', 'yearConstructed', 'serviceCharge']

for column in columns_to_fillna:
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)


# In[13]:


# Outlier handling
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

def remove_outliers(column):
    upper_range = column.mean() + 3 * column.std()
    lower_range = column.mean() - 3 * column.std()
    return (column <= upper_range) & (column >= lower_range)

# Update the data frame, leaving only rows without outliers
data = data[data[numeric_columns].apply(remove_outliers, axis=0).all(axis=1)]


# In[14]:


nan_count_v3 = data.isnull().sum()
print(nan_count_v3)


# In[15]:


numeric_columns = [ 'totalRent', 'serviceCharge', 'newlyConst', 'yearConstructed','hasKitchen', 'cellar', 
                   'garden', 'baseRent', 'livingSpace', 'baseRentRange', 'noRooms', 'floor']


numeric_df = data[numeric_columns]
correlation_matrix = numeric_df.corr()

cmap = sns.color_palette(["#FFD700", "#FF6666", "#808080"]) 

sns.heatmap(correlation_matrix, fmt='.2f', annot=True, cmap=cmap, linewidths=.6)
plt.title('Correlation Heatmap', fontsize=10)

plt.show()


# In[17]:


# Drop the column and 'baseRentRange' that are clearly unrelated to the total rental
data.drop(['cellar', 'garden', 'floor','baseRentRange'],axis=1,inplace=True)


# In[18]:


plt.figure(figsize=(10, 6))
sns.kdeplot(data['totalRent'], shade=True, color='skyblue')
plt.xlabel('Total Rent')
plt.ylabel('Density')
plt.title('Total Rent Density Plot')
plt.show()


# In[19]:


# The tail on the right is too long, so I think the data needs to be adjusted.data['totalRent'].describe()
data['totalRent'].describe()


# In[20]:


Q1 = data['totalRent'].quantile(0.25)
Q3 = data['totalRent'].quantile(0.75)

# IQR
IQR = Q3 - Q1

# boundary calculation
lower_bound = Q1 - 1.5 * IQR
upper_bound = 10000

# Delete data that are not in the zone
data = data[(data['totalRent'] >= lower_bound) & (data['totalRent'] <= upper_bound)]


# In[21]:


plt.figure(figsize=(10, 6))
sns.kdeplot(data['totalRent'], shade=True, color='skyblue')
plt.xlabel('Total Rent')
plt.ylabel('Density')
plt.title('Total Rent Density Plot')
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
sns.kdeplot(data['totalRent'], shade=True, color='skyblue')
plt.xlabel('Total Rent')
plt.ylabel('Density')
plt.title('Total Rent Density Plot')
plt.xlim(0, 4000)
plt.show()


# In[23]:


data['totalRent'].describe()


# # Part 2 / Geopandas& Geodata

# In[24]:


geo_data = gp.read_file("plz-5stellig.shp")


# In[25]:


geo_data.info()


# In[26]:


geo_data.head(10)


# In[27]:


geo_data.plot(figsize=(5,5))


# In[28]:


odata = pd.read_csv("immo_data.csv")
subset_data = odata[['regio2', 'geo_plz']]
subset_data.to_csv("subset_immo_data.csv", index=False)


# # Part 3 / Visualization

# In[91]:


gray_color = '#989898' # Hexadecimal code!!
dark_gray_color = '#707070'
more_dark_gray_color = '#404040'
background_image = Image.open('germancity_1.jpg')

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(background_image, extent=[-0.5, 14.5, 0, 20000], aspect='auto', alpha=0.15)
top_cities = data['regio2'].value_counts().head(15)
barplot = sns.barplot(x=top_cities.index, y=top_cities.values, color=dark_gray_color, ax=ax)

for p in barplot.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points',
                fontsize=9, color=more_dark_gray_color)

plt.xticks(rotation=60, color=dark_gray_color)
plt.yticks(color=dark_gray_color)
plt.ylim(0, 14000)
plt.xlabel('City', size=15, color=more_dark_gray_color)
plt.ylabel('Number of Listings', size=15, color=more_dark_gray_color)
plt.title('Top 15 Cities with Highest Listings', size=22, color= more_dark_gray_color)
plt.show()


# In[90]:


mean_rent_by_regio2 = data.groupby('regio2')['totalRent'].mean().reset_index()
top_15_cities = mean_rent_by_regio2.sort_values(by='totalRent', ascending=False).head(15)

background_image = Image.open('germancity_2.jpg')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(background_image, extent=[-0.5, 14.5, 0, 3000], aspect='auto', alpha=0.15)
barplot = sns.barplot(x='regio2', y='totalRent', data=top_15_cities, color=dark_gray_color, ax=ax)

for p in barplot.patches:
    ax.annotate(f'{p.get_height():.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points',
                fontsize=9, color=more_dark_gray_color)

plt.xticks(rotation=60, color=dark_gray_color)
plt.yticks(color=dark_gray_color)
plt.xlabel('City', size=15, color=more_dark_gray_color)
plt.ylabel('Average Total Rent Price', size=15, color=more_dark_gray_color)
plt.title('Top 15 Cities with Highest Average Total Rent Price / €', size=18, color= more_dark_gray_color)
plt.ylim(1000, 2000)

plt.show()


# In[29]:


# Shapefile load
shapefile_path = 'plz-5stellig.shp'

mean_rent_by_regio2 = data.groupby('regio2')['totalRent'].mean().reset_index()
top_15_cities = mean_rent_by_regio2.sort_values(by='totalRent', ascending=False).head(24)

germany_shapes = gp.read_file(shapefile_path)

# PLZ Matching
top_cities_plz = data[data['regio2'].isin(top_15_cities['regio2'])].drop_duplicates(subset=['regio2'])

germany_shapes['plz'] = germany_shapes['plz'].astype(str)
top_cities_plz['geo_plz'] = top_cities_plz['geo_plz'].astype(str)

# Shapefile Matching
mapped_data = germany_shapes.merge(top_cities_plz, left_on='plz', right_on='geo_plz')

# Extract lat and lon from the center of goe
mapped_data['lon'] = mapped_data.geometry.apply(lambda x: x.centroid.x)
mapped_data['lat'] = mapped_data.geometry.apply(lambda x: x.centroid.y)

mapped_data = mapped_data.set_index('regio2').reindex(top_15_cities['regio2']).reset_index()


# Plotly map creation
fig = px.scatter_mapbox(
    mapped_data,
    lat='lat',
    lon='lon',
    hover_name='regio2',
    zoom=5,
    center={"lat": 51.1657, "lon": 10.4515},
    mapbox_style="open-street-map",
    size_max=20,  
    color='regio2',  
    size='totalRent'  
)


fig.update_layout(
    title_text="Top 24 Cities in Germany by Average Total Rent",
    title_font=dict(size=20),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    paper_bgcolor="rgba(245,245,245,1)")

#Legend
fig.update_layout(
    legend=dict(
        title=("<- Circle size = Costs"),  
        x=1,                  
        y=1,                
        bordercolor="Black",   
        borderwidth=0.5       
    ))

fig.show()


# In[99]:


top_15_cities = (
    data.groupby('regio2')['totalRent']
    .mean()
    .reset_index()
    .sort_values(by='totalRent', ascending=False)
    .head(15)
)

top_15_regio2 = top_15_cities['regio2']
top_regio2_data = data[data['regio2'].isin(top_15_regio2)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='regio2', y='totalRent', data=top_regio2_data, order=top_15_regio2, palette="Set3")
plt.xticks(rotation=90)
plt.xlabel('City')
plt.ylabel('Total Rent Price')
plt.title('Total Rent Price Distribution by City Top 15')
plt.tight_layout()
plt.show()


# In[105]:


mean_livingspace_by_regio2 = data.groupby('regio2')['livingSpace'].mean().reset_index()

top_15_cities_livingspace = (
    mean_livingspace_by_regio2
    .sort_values(by='livingSpace', ascending=False)
    .head(15)
)

background_image = Image.open('germancity_7_apt.jpg')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(background_image, extent=[90, 105, -0.5, 15], aspect='auto', alpha=0.15, origin='lower')
barplot = sns.barplot(x='livingSpace', y='regio2', data=top_15_cities_livingspace, color=dark_gray_color, ax=ax)

for p in barplot.patches:
    ax.annotate(f'{p.get_width():.1f}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha = 'left', va = 'center', 
                xytext = (5, 0), 
                textcoords = 'offset points',
                fontsize=9, color=more_dark_gray_color) 


plt.xlabel('Average Living Space', color=more_dark_gray_color)
plt.ylabel('City', color=more_dark_gray_color)
plt.xticks(color=dark_gray_color)
plt.yticks(color=dark_gray_color)
plt.title('Top 15 Cities with Highest Average Living Space / sqm', size=18, color=more_dark_gray_color)
plt.xlim(90, 105)

plt.show()


# In[107]:


mean_totalrent_by_heatingtype = data.groupby('heatingType')['totalRent'].mean().reset_index()

sorted_heatingtype_totalrent = mean_totalrent_by_heatingtype.sort_values(by='totalRent', ascending=False)

background_image = Image.open('germancity_10_heating.jpg')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(background_image, extent=[0, 2000, -15, 20], aspect='auto', alpha=0.15, origin='lower')
barplot = sns.barplot(x='totalRent', y='heatingType', data=sorted_heatingtype_totalrent, color=dark_gray_color, ax=ax)

for p in barplot.patches:
    ax.annotate(f'{p.get_width():.0f}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha = 'left', va = 'center', 
                xytext = (5, 0), 
                textcoords = 'offset points',
                fontsize=9, color=more_dark_gray_color)

plt.xlabel('Average Total Rent', color=more_dark_gray_color)
plt.ylabel('Heating Type', color=more_dark_gray_color)
plt.xticks(color=dark_gray_color)
plt.yticks(color=dark_gray_color)
plt.title('Average Total Rent by Heating Type / €', size=18, color=more_dark_gray_color)
plt.xlim(400, 1500)

plt.show()


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd

mean_totalrent_by_heatingtype = data.groupby('condition')['totalRent'].mean().reset_index()

sorted_heatingtype_totalrent = mean_totalrent_by_heatingtype.sort_values(by='totalRent', ascending=False)

background_image = Image.open('germancity_11_apt.jpg')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(background_image, extent=[0, 2000, -5, 12], aspect='auto', alpha=0.15, origin='lower')

barplot = sns.barplot(x='totalRent', y='condition', data=sorted_heatingtype_totalrent, color=dark_gray_color, ax=ax)

for p in barplot.patches:
    ax.annotate(f'{p.get_width():.0f}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha = 'left', va = 'center', 
                xytext = (5, 0), 
                textcoords = 'offset points',
                fontsize=9, color=more_dark_gray_color)

plt.xlabel('Average Total Rent', color=more_dark_gray_color)
plt.ylabel('Condition', color=more_dark_gray_color)
plt.xticks(color= dark_gray_color)
plt.yticks(color= dark_gray_color)
plt.title('Average Total Rent by Condition / €', size=18, color=more_dark_gray_color)
plt.xlim(400, 1500)

plt.show()


# In[30]:


# Please use %matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib.animation import FuncAnimation

total_rent_mean = np.mean(data['totalRent'])
base_rent_mean = np.mean(data['baseRent'])
difference = total_rent_mean - base_rent_mean

fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

# 1000ms / 50ms = 20 frames
pause_frames = int(5000 / 50)

light_black = (0.5, 0.5, 0.5) 
light_red = (1, 0.5, 0.5)     
light_gold = (1, 0.85, 0.5)    

positions = np.arange(3) 
bar_width = 0.95  

def animate_with_percentage_and_pause(i):
    if i <= 25:  # First bar
        ax.clear()
        ax.bar(positions[0], total_rent_mean * i / 25, bar_width, alpha=opacity, color=light_black, label='Total Rent')
    elif i <= 50:  # Second bar
        ax.clear()
        ax.bar(positions[0], total_rent_mean, bar_width, alpha=opacity, color=light_black, label='Total Rent')
        ax.bar(positions[1], base_rent_mean * (i - 25) / 25, bar_width, alpha=opacity, color=light_red, label='Base Rent')
    elif i <= 75:  # Third Bar
        ax.clear()
        ax.bar(positions[0], total_rent_mean, bar_width, alpha=opacity, color=light_black, label='Total Rent')
        ax.bar(positions[1], base_rent_mean, bar_width, alpha=opacity, color=light_red, label='Base Rent')
        ax.bar(positions[2], difference * (i - 50) / 25, bar_width, alpha=opacity, color=light_gold, label='Difference')

    if i == 75:  # Figure
        percentage = (difference / total_rent_mean) * 100  # percentage caculation 
        ax.text(positions[2], difference, f'{percentage:.2f}%', ha='center', va='bottom')

    ax.set_title('Additional Charge Percentage / €')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Total Rent', 'Base Rent', 'Difference'])

# Run animation (pause for 5 seconds after displaying numbers)
ani = FuncAnimation(fig, animate_with_percentage_and_pause, frames=np.arange(0, 75 + pause_frames + 1), interval=50, repeat=True, blit=True)

plt.show()


# In[35]:


kitchen_count = data['hasKitchen'].value_counts(normalize=True)  

true_count = kitchen_count.get(True, 0)
false_count = kitchen_count.get(False, 0)

df = pd.DataFrame({'True': [true_count], 'False': [false_count]})

background_image = Image.open('germancity_12_kitchen.jpeg')

fig, ax = plt.subplots(figsize=(7, 10))
ax.imshow(background_image, extent=[0.01, 1, -0.7, 1.2], aspect='auto', alpha=0.3)

light_lilac = (200/255, 162/255, 200/255)
light_turquoise = (160/255, 220/255, 220/255)

# horizontal Stacked Bar Chart 
barplot = df.plot(kind='barh', stacked=True, color=[light_lilac, light_turquoise], ax=ax, width=0.1)

# Insert text in the bars
for bar in barplot.patches:
    width = bar.get_width()
    label_x_pos = bar.get_x() + width / 2
    text = 'Yes' if bar.get_x() < 0.1 else 'No'
    percentage = f'{width:.0%}'
    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{text} ({percentage})', 
            ha='center', va='center', fontsize=12) 

plt.title('Does the house have a kitchen?')
plt.xlabel('Proportion')
plt.ylabel('Has Kitchen ?!')
plt.yticks(rotation=0)  # rotate y axis lable 
plt.legend(title='Kitchen')

plt.xlim(0, 1.0)

plt.show()



# ## End of Document 

# In[36]:


image_path = "germany_kitchen_1.png" 
img = Image.open(image_path)

new_width = 150  
new_height = 150 
img = img.resize((new_width, new_height), Image.LANCZOS)

plt.imshow(img)
plt.axis('off') 
plt.show()

