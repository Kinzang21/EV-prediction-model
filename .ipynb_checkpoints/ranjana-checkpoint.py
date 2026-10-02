#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#Loading dataset
data=pd.read_csv("Airbnb_NewYorkCity_Data.csv",low_memory=False)
data


# In[2]:


# printing the basic informationa and sample data
shape=data.shape
print('Initial Data Shape:',shape)


# In[3]:


columns=data.columns
print('Columns in the dataset:',columns)


# In[4]:


head=data.head(5)
print('First few rows',head)


# In[5]:


info=data.info()
print (info)


# In[6]:


# printing basic statistics
print(data.describe())


# In[7]:


# printing the data types of each column name.
print(data.dtypes)


# In[5]:


# Checking  for missing values
print('𝗕𝗲𝗳𝗼𝗿𝗲 𝗳𝗶𝗹𝗹𝗶𝗻𝗴 𝘁𝗵𝗲 𝗠𝗶𝘀𝘀𝗶𝗻𝗴 𝘃𝗮𝗹𝘂𝗲𝘀:\n',data.isnull().sum())


# In[6]:


# Dropping the columns with high missing percentages (if these columns are not useful) 
# and specifying axis as 1 to choose column to be droped.
columns_to_drop = ['license', 'house_rules'] 
data = data.drop(columns=columns_to_drop,axis=1)


# In[7]:


# Checking the current columns of the dataset Airbnb
print("Columns in the dataset:", data.columns)

# Dropping the columns with high missing percentages
drop_colunms = ['license', 'house_rules']
data = data.drop(columns=drop_colunms, errors='ignore')

# Confirming the columns were dropped
print("Updated columns of the dataset:", data.columns)
data 


# In[8]:


# Droping rows with missing critical fields like name and country, and other related coulmns which had no values.
data = data.dropna(subset=['NAME','country','instant_bookable','Construction year','lat','long'])
data


# In[9]:


# Handling Missing Values in Airbnb Dataset

###Filling  missing values in categorical columns.###

# Filling missing values of 'host name' column with 'No Name'
data['host name'].fillna('No Name', inplace=True)

# Filling missing values in 'neighbourhood group' column with 'Unknown'
data['neighbourhood group'].fillna('Unknown', inplace=True)

# Filling missing values in 'neighbourhood' column with 'Unknown'
data['neighbourhood'].fillna('Unknown', inplace=True)

# Filling missing values in 'cancellation_policy' column with 'No Policy'
data['cancellation_policy'].fillna('No Policy', inplace=True)

# Filling missing values in 'host_identity_verified' column with 'Unknown'
data['host_identity_verified'].fillna('Unknown', inplace=True)

# Forward-filling of missing values in 'country' column since all values are expected to be 'United States'
data['country'].fillna(method='ffill', inplace=True)

# Forward-filling of missing values in 'country_code' column since all values are expected to be 'US'
data['country code'].fillna(method='ffill', inplace=True)


###Filling missing values in numerical columns with the mean and 0 #######

# Filling missing values in 'calculated_host_listings_count' column with the mean of the column
data['calculated host listings count'].fillna(data['calculated host listings count'].mean(), inplace=True)

# Filling missing values in 'minimum_nights' column with the mean of the column
data['minimum nights'].fillna(data['minimum nights'].mean(), inplace=True)

# Filling missing values in 'review_rate_number' column with the mean of the column
data['review rate number'].fillna(data['review rate number'].mean(), inplace=True)

# Filling missing values in 'availability_365' column with the mean of the column
data['availability 365'].fillna(data['availability 365'].mean(), inplace=True)

# Filling missing values in 'reviews_per_month' column with 0, indicating no reviews
data['reviews per month'].fillna(0, inplace=True)

# Filling missing values in 'number_of_reviews' column with 0, indicating no reviews
data['number of reviews'].fillna(0, inplace=True)

# Confirming that all the missing values have been handled
print("𝗔𝗳𝘁𝗲𝗿 𝗳𝗶𝗹𝗹𝗶𝗻𝗴 𝘁𝗵𝗲 𝗠𝗶𝘀𝘀𝗶𝗻𝗴 𝘃𝗮𝗹𝘂𝗲𝘀:\n", data.isnull().sum())


# In[10]:


### Cleaning and Converting Data Types###
# Removing dollar signs and converting price and service fee to numeric values
data.loc[:, 'price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
data.loc[:, 'service fee'] = data['service fee'].replace('[\$,]', '', regex=True).astype(float)
# Filling null values of service fee and price column with the mean.
data.fillna({'service fee': data['service fee'].mean(), 'price': data['price'].mean()}, inplace=True)
data


# In[13]:


# Converting last review to datetime format
data['last review'] = pd.to_datetime(data['last review'], errors='coerce')
# Replacing the NaT of last review with the most frequent date (mode)
frequent_date = data['last review'].mode()[0]  # Find the most frequent date
data['last review'].fillna(frequent_date, inplace=True)
data


# In[11]:


# Changing data types of number of review from float64 to int
data['number of reviews'] = data['number of reviews'].astype(int)

# Changing data types of cAalculated host listing count from float64 to int
data['calculated host listings count'] = data['calculated host listings count'].astype(int)

# Changing data types of availability 365 from float64 to int
data['availability 365'] = data['availability 365'].astype(int)

# Changing data types of minimum nights from float64 to int
data['minimum nights'] = data['minimum nights'].astype(int)

# Changing data types of Calculated host listing count from float64 to int
data['Construction year'] = data['Construction year'].astype(int)


# Changing data types of Price from float64 to int
data['price'] = data['price'].astype(int)

# Changing data types of Service_Fee from float64 to int
data['service fee'] = data['service fee'].astype(int)

# Changing data types of review rate number from float64 to int
data['review rate number'] = data['review rate number'].astype(int)

# Changing data types of instant bookable from object to bool
data['instant_bookable'] = data['instant_bookable'].astype(bool)
# Displaying the final Data type
print(data)


# In[15]:


# displaying the 
print('Revised DataTypes:\n',data.dtypes)


# In[14]:


### Encoding Categorical Data
# Binary encoding for columns with 'True'/'False' values
data['instant_bookable'] = data['instant_bookable'].map({'TRUE': 1, 'False': 0})
data['host_identity_verified'] = data['host_identity_verified'].map({'verified': 1, 'unconfirmed': 0})
data 


# In[16]:


# Checking for duplicates
print(data.duplicated().sum())


# In[17]:


# Removing duplicates if any
data= data.drop_duplicates()
data


# In[18]:


### Identifying and Removing the Outliers form the dataset Airbnb###

# Defining a function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Applying outlier removal to relevant columns like 'price' and 'service fee'

data = remove_outliers(data, 'price')
data = remove_outliers(data, 'service fee')
data


# In[19]:


### Final Cleaned Data Check

print("Final Data Shape:", data.shape)
print("First few rows after cleaning:\n", data.head())
print("No Missing values after cleaning:\n", data.isnull().sum())


# In[20]:


# Saving the Cleaned Dataset
data.to_csv("cleaned_airbnb_data.csv")
print("Cleaned dataset saved as 'cleaned_airbnb_data.csv")


# In[21]:


new_data = pd.read_csv('cleaned_airbnb_data.csv')
print(new_data.head())


# In[26]:


new_data = new_data.dropna()
new_data


# In[22]:


new_data.to_csv("clean_airbnb_data.csv")
print("Cleaned dataset saved as 'clean_airbnb_data.csv")


# In[25]:


# Ensure 'price' column is numeric
new_data['price'] = pd.to_numeric(new_data['price'], errors='coerce')  # Convert to numeric, invalid values become NaN

# Drop rows where 'price' is NaN (if needed)
new_data = new_data.dropna(subset=['price'])

# Find the most expensive listing
most_expensive = new_data.loc[new_data['price'].idxmax()]
print(f"Most Expensive Listing: {most_expensive['NAME']} in {most_expensive['neighbourhood']} costs ${most_expensive['price']}")


# In[27]:


# Identify the most expensive listing
most_expensive = new_data.loc[new_data['price'].idxmax()]
print(f"Most Expensive Listing: {most_expensive['NAME']} in {most_expensive['neighbourhood']} costs ${most_expensive['price']}")


# In[37]:


import matplotlib.pyplot as plt
# Calculating average price by neighborhood group
avg_price_group = new_data.groupby('neighbourhood group')['price'].mean().sort_values(ascending=False)
print(avg_price_group)
# Bar chart for average price by neighborhood group
avg_price_group.plot(kind='bar', color='purple')
plt.title("Average Price by Neighborhood Group")
plt.xlabel("Neighborhood Group")
plt.ylabel("Average Price")
plt.show()


# In[51]:


# Finding the most common room type
room_type_counts = new_data['room type'].value_counts()
print(room_type_counts)
# Pie chart for room types
room_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['gold', 'skyblue', 'lightgreen'])
plt.title("Room Type Distribution")
plt.ylabel('')
plt.show()


# In[42]:


# Finding the most common room type
room_type_counts = new_data['room type'].value_counts()
print(room_type_counts)


# In[44]:


# Host with the most listings
host_counts = new_data['host id'].value_counts()
top_host = host_counts.idxmax()
print(f"Host {top_host} has the most listings: {host_counts.max()}")


# In[48]:


# Analyzing price distribution
print(new_data['price'].describe())


# In[50]:


# Calculating correlation between price and number of reviews
correlation = new_data['price'].corr(new_data['number of reviews'])
print(f"Correlation between price and number of reviews: {correlation}")


# In[60]:


# Affordable listings below $100
affordable_listings = new_data[new_data['price'] < 100]
top_neighborhoods = affordable_listings['neighbourhood'].value_counts().head(5)
print(top_neighborhoods)
# Bar chart for top 5 affordable neighborhoods
top_neighborhoods.plot(kind='bar', color='orange')
plt.title("Top 5 Neighborhoods for Affordable Stays")
plt.xlabel("Neighborhood")
plt.ylabel("Number of Listings")
plt.show()


# In[64]:


# Analyze availability trends (number of available days per year)
availability = new_data['availability 365'].value_counts()
print(availability.head())


# In[66]:


# Grouping by room type and neighborhood group, then calculate average price
avg_price = new_data.groupby(['room type', 'neighbourhood group'])['price'].mean()
print(avg_price)
# Bar chart for average price by room type and neighborhood group
avg_price.unstack().plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightcoral', 'skyblue', 'gold'])
plt.title("Average Price by Room Type and Neighborhood Group")
plt.xlabel("Neighborhood Group")
plt.ylabel("Average Price")
plt.show()


# In[70]:


import matplotlib.pyplot as plt

# Plot histogram of listing prices
plt.figure(figsize=(10, 6))
plt.hist(new_data['price'], bins=50, color='skyblue', edgecolor='black')
plt.title("Price Distribution of Airbnb Listings", fontsize=16)
plt.xlabel("Price (USD)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ploting a density plot of price
plt.figure(figsize=(10, 6))
sns.kdeplot(data=new_data, x='price', fill=True, color="blue", bw_adjust=0.5)
plt.title("Density Plot of Airbnb Prices", fontsize=16)
plt.xlabel("Price (USD)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 500)  # Limit x-axis to exclude extreme outliers
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[79]:


import matplotlib.pyplot as plt

# Grouping the rest of the neighborhood groups into 'Others' if there are too many categories
neighborhood_group_counts = new_data['neighbourhood group'].value_counts()

# If there are too many groups, combine the smaller groups into 'Others'
threshold = 0.05  # Consider groups that have less than 5% of the total as 'Others'
small_groups = neighborhood_group_counts[neighborhood_group_counts / neighborhood_group_counts.sum() < threshold]
neighborhood_group_counts = neighborhood_group_counts[neighborhood_group_counts / neighborhood_group_counts.sum() >= threshold]
neighborhood_group_counts['Others'] = small_groups.sum()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
neighborhood_group_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0', '#ffb3e6'])
plt.title("Distribution of Listings by Neighborhood Group", fontsize=16)
plt.ylabel("")  # Remove the y-axis label for a cleaner appearance
plt.show()



# In[80]:


# Group availability by room type
availability_by_room = new_data.groupby('room type')['availability 365'].mean()

# Stacked bar chart
plt.figure(figsize=(10, 6))
availability_by_room.plot(kind='bar', color=['#6a5acd', '#ffa07a', '#20b2aa', '#ffb6c1'])
plt.title("Average Availability (Days) by Room Type", fontsize=16)
plt.xlabel("Room Type", fontsize=12)
plt.ylabel("Average Availability (Days)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[81]:


# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = new_data[['price', 'number of reviews', 'availability 365']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt='.2f')
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


# In[84]:


# Availability vs. Price with Bubble Size as Number of Reviews
plt.figure(figsize=(10, 6))
plt.scatter(new_data['availability 365'], new_data['price'], s=new_data['number of reviews']*2, color='gold', alpha=0.5)
plt.title("Availability vs. Price with Reviews as Bubble Size", fontsize=16)
plt.xlabel("Availability (Days per Year)", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.grid(True)
plt.show()


# In[85]:


# Price Distribution by Room Type (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='room type', y='price', data=new_data, palette='Set2')
plt.title("Price Distribution by Room Type", fontsize=16)
plt.xlabel("Room Type", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[86]:


# Listings Distribution by Neighborhood Group and Room Type (Stacked Bar Chart)
room_type_by_neighborhood = pd.crosstab(new_data['neighbourhood group'], new_data['room type'])

plt.figure(figsize=(12, 6))
room_type_by_neighborhood.plot(kind='bar', stacked=True, color=['lightcoral', 'lightseagreen', 'skyblue'])
plt.title("Listings Distribution by Neighborhood Group and Room Type", fontsize=16)
plt.xlabel("Neighborhood Group", fontsize=12)
plt.ylabel("Number of Listings", fontsize=12)
plt.legend(title="Room Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[87]:


pip show bokeh


# In[91]:


pip install --upgrade bokeh


# In[92]:


pip install bokeh pandas numpy


# In[108]:


from bokeh.io import output_notebook, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.palettes import Purples256, Magma256
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Load Airbnb dataset
data = pd.read_csv("clean_airbnb_data.csv")

# Set output for Jupyter Notebook
output_notebook()

# Data Preparation
data['price'] = data['price'].astype(float)
data['availability_365'] = data['availability 365'].astype(float)

# Custom Pink & Purple Color Palette
primary_color = "#D291BC"  # Light Pink
secondary_color = "#9D5C91"  # Medium Purple
accent_color = "#6B2E5E"  # Dark Purple
gradient_colors = Purples256

# 1. Price Distribution
price_hist, edges = np.histogram(data['price'], bins=30)
price_source = ColumnDataSource(data=dict(
    left=edges[:-1], right=edges[1:], top=price_hist
))
price_fig = figure(
    title="Distribution of Airbnb Prices",
    x_axis_label="Price (USD)",
    y_axis_label="Frequency",
    width=450,
    height=350,
    tools="hover,pan,box_zoom,reset,save"
)
price_fig.quad(
    source=price_source, top="top", bottom=0, left="left", right="right",
    fill_color=secondary_color, line_color="white", alpha=0.8
)
price_fig.add_tools(HoverTool(
    tooltips=[("Price Range", "@left - @right"), ("Frequency", "@top")]
))

# 2. Density Plot for Price
density = gaussian_kde(data['price'])
x_vals = np.linspace(data['price'].min(), data['price'].max(), 1000)
y_vals = density(x_vals)
density_source = ColumnDataSource(data=dict(x=x_vals, y=y_vals))
density_fig = figure(
    title="Density Plot of Prices",
    x_axis_label="Price (USD)",
    y_axis_label="Density",
    width=450,
    height=350,
    tools="hover,pan,box_zoom,reset,save"
)
density_fig.line('x', 'y', source=density_source, line_width=3, color=primary_color)
density_fig.add_tools(HoverTool(
    tooltips=[("Price", "$x{0.00}"), ("Density", "$y{0.000}")]
))

# 3. Distribution of Listings by Neighborhood Group (Pie Chart)
neighborhood_counts = data['neighbourhood group'].value_counts()
neighborhood_data = pd.DataFrame({
    'group': neighborhood_counts.index,
    'count': neighborhood_counts.values,
    'angle': neighborhood_counts / neighborhood_counts.sum() * 2 * np.pi,
    'color': gradient_colors[:len(neighborhood_counts)]
})
pie_source = ColumnDataSource(neighborhood_data)
pie_fig = figure(
    title="Distribution by Neighborhood Group",
    toolbar_location=None,
    tools="hover",
    tooltips="@group: @count listings",
    x_range=(-0.5, 1),
    width=450,
    height=350
)
pie_fig.wedge(
    x=0, y=0, radius=0.4,
    start_angle=cumsum('angle', include_zero=True),
    end_angle=cumsum('angle'),
    line_color="white",
    fill_color='color',
    legend_field='group',
    source=pie_source
)
pie_fig.axis.axis_label = None
pie_fig.axis.visible = False
pie_fig.grid.grid_line_color = None

# 4. Average Availability by Room Type
room_availability = data.groupby('room type')['availability_365'].mean()
availability_source = ColumnDataSource(data=dict(
    room_type=room_availability.index, availability=room_availability.values
))
availability_fig = figure(
    x_range=list(room_availability.index),
    title="Average Availability by Room Type",
    x_axis_label="Room Type",
    y_axis_label="Average Availability (days)",
    width=450,
    height=350,
    tools="hover,pan,box_zoom,reset,save"
)
availability_fig.vbar(
    x="room_type", top="availability", width=0.8,
    fill_color=primary_color, line_color="white", alpha=0.8,
    source=availability_source
)
availability_fig.add_tools(HoverTool(
    tooltips=[("Room Type", "@room_type"), ("Availability (days)", "@availability{0.00}")]
))

# 5. Price Distribution by Room Type (Mean Bar)
room_type_prices = data.groupby('room type')['price'].mean()
box_source = ColumnDataSource(data=dict(
    room_type=room_type_prices.index, price=room_type_prices.values
))
box_fig = figure(
    x_range=list(room_type_prices.index),
    title="Price Distribution by Room Type (Mean)",
    x_axis_label="Room Type",
    y_axis_label="Average Price (USD)",
    width=450,
    height=350,
    tools="hover,pan,box_zoom,reset,save"
)
box_fig.vbar(
    x="room_type", top="price", width=0.8,
    fill_color=secondary_color, line_color="white", alpha=0.8,
    source=box_source
)
box_fig.add_tools(HoverTool(
    tooltips=[("Room Type", "@room_type"), ("Average Price", "$@price{0.00}")]
))

# 6. Stacked Bar Chart: Listings by Neighborhood and Room Type
stacked_data = data.groupby(['neighbourhood group', 'room type']).size().unstack(fill_value=0)
stacked_colors = gradient_colors[:stacked_data.shape[1]]
stacked_fig = figure(
    x_range=stacked_data.index.tolist(),
    title="Listings by Neighborhood and Room Type",
    x_axis_label="Neighborhood Group",
    y_axis_label="Number of Listings",
    width=450,
    height=350,
    tools="hover,pan,box_zoom,reset,save"
)
for i, room_type in enumerate(stacked_data.columns):
    stacked_fig.vbar(
        x=stacked_data.index, top=stacked_data[room_type].cumsum(), width=0.5,
        fill_color=stacked_colors[i], line_color="white", legend_label=room_type, alpha=0.7
    )
stacked_fig.legend.orientation = "horizontal"
stacked_fig.legend.location = "top_center"

# Arrange visualizations in a dashboard layout
dashboard = column(
    row(price_fig, density_fig),
    row(pie_fig, availability_fig),
    row(box_fig, stacked_fig)
)

# Show the dashboard
show(dashboard, notebook_handle=True)

