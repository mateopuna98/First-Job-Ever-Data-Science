#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import operator
import geopandas as gpd
import descartes
import ast
from shapely.geometry import Point , Polygon
from collections import Counter
from sklearn.cluster import OPTICS,  cluster_optics_dbscan
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


#If you need a graphical debugger you can run this line
#import pixiedust


# In[26]:


file_path = '/Users/Mariana/Desktop/project/Python-Project/EDA_datasets/'
common3_n = 'common3.csv'
brand_day_df_n = 'brand_day_df.csv'
brand_month_df_n = 'brand_month_df.csv'
brand_top_df_n = 'brand_top_df.csv'


# In[27]:


#Read everything from the csv file and format the columns related to brands with the right type
common3 = pd.read_csv((file_path+common3_n), dtype={'census_block_group':str},
                      converters={"related_same_day_brand": ast.literal_eval,
                                  "related_same_month_brand": ast.literal_eval,
                                  "top_brands" : ast.literal_eval })


# In[4]:


#Load several files form csv
brand_day_df= pd.read_csv((file_path+brand_day_df_n), dtype={'census_block_group':str})


# In[5]:


brand_month_df= pd.read_csv((file_path+brand_month_df_n), dtype={'census_block_group':str})


# In[6]:


brand_top_df= pd.read_csv((file_path+brand_top_df_n), dtype={'census_block_group':str})


# In[28]:


hour_list = ['12am','1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']
week_list = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# In[29]:


common3.head()


# <p>Functions to find top brands in a given dataframe</p>

# In[13]:


def count_brands_frequency(brands_list,brands_dict):
    for brand in brands_list:#receives a list with all brands (brands list) and iterates over every brand
        if brand in brands_dict: # if the brand has appeared before it updates the brand count by one
            brands_dict[brand] += 1
        else: # if the brand hasn't appeared before it adds it to the dict with the value of 1 and the key the name of the brand
            brands_dict.update({brand : 1})
    

def find_top_x_brands(dataframe,number):
    brands_dict = dict()
    dataframe.apply(lambda brands_list : count_brands_frequency(brands_list, brands_dict))#apply count_brands_frequency to every row in the dataframe
    counter = Counter(brands_dict)#Format needed to find top x brands
    top_x = counter.most_common(number)#Find most common brands
    top_x = [elem[0] for elem in top_x] #returns only first element of brand (the name of the brand the second is the frequency)
    
    return top_x
    

def find_top_x_brands_with_vals(dataframe,number):
    brands_dict = dict()
    dataframe.apply(lambda brands_list : count_brands_frequency(brands_list, brands_dict))
    counter = Counter(brands_dict)
    top_x = counter.most_common(number)
    
    return top_x # The only difference with the method above is that this one returns both the brand name and the frequency


# <b>Plot cbgs with given latitude and longitude</b>

# <p>Get latitude and longitude from main dataframe</p>

# In[25]:


#Gets only latitude and longitude columns
coordinates_df = common3[['latitude','longitude']].copy()


# In[11]:


coordinates_df.head()


# In[12]:


# gets the  us map over which we will plot the cbgs
us_map = gpd.read_file("/Users/Mariana/Desktop/project/Python-Project/Maps/USA States/USA_States.shp")


# In[13]:


#Coordinate reference system that the graphic will use
crs = {'init': 'epsg:4326'}


# In[14]:


#Creation of the geometry object needed to plot the cbgs
geometry = [Point(xy) for xy in zip(coordinates_df['longitude'],coordinates_df['latitude'])]
#zip() takes 2  separated values, eg. 10 and 4
#and merges then into a  single tuple, eg (10,4)


# In[15]:


geometry[:3]


# In[16]:


us_map.head()


# In[17]:


#create a dataset with latitude and longitude to common3 main dataset
#axis = 1  means it will merge the values as new column 
cbg_w_coordinates = pd.concat([common3['census_block_group'],coordinates_df],axis = 1, ignore_index= False, sort=False)


# In[18]:


#Dataset needed to plot the object with the geopandas library
#Very similar to Pandas DataFrame
geo_data = gpd.GeoDataFrame(cbg_w_coordinates, crs= crs,geometry = geometry)


# In[19]:


geo_data.head()


# In[20]:


geo_data.shape


# <b>Plot most_popular cbgs (with the percentage given)</b>

# In[21]:


common3.shape


# In[22]:


column_name = 'raw_visit_count'


# In[23]:


def get_most_popular_cbgs(number,column_name):
    """Returns a dataframe containing the k-largest cbgs"""
    if number <= 0:
        print("Please enter a valid number of cbgs")
        return
    #first get the number that the percentage represents
    num_of_cbgs = number 
    #get the cbg list from main dataset
    cbgs_list = common3[['census_block_group', column_name]]
    
    return cbgs_list.sort_values(by = column_name, ascending = False).iloc[0:(num_of_cbgs -1)]['census_block_group']
#sort_values() returns the values ordered. ascending = False means the highest values go first


# In[24]:


#So now I have to the the most popular cbgs and then get the subset from the geo_data variable to plot it
def plot_most_popular_cbgs_in_map(number,column_name,geo_data,dot_size):  
    """This method provides an easy way to plot the cbgs, the first arg is the number of cbgs, the the column (visitors per day, per week, etc), dot_size is the radio of the plotted dots"""
    #The next line gets the rows in the main dataset that correspond to the most popular dataset
    cbgs_list = geo_data[geo_data['census_block_group'].isin(get_most_popular_cbgs(number,column_name))]
    figure, ax = plt.subplots(figsize  = (50,30))#Figsize values are the dimension of the plot
    #plots the US map
    us_map.plot(ax = ax, alpha = 0.5, color = "grey")
    #plots the cbgs in map
    cbgs_list.plot(ax = ax, markersize = dot_size, color = '#7C1809', marker = "o", label = "census block group")
    #Saves the figure as pdf
    plt.savefig("Top_" + str(number) + "cbgs.pdf")
    plt.legend(prop = {'size' : 15})

    
    


# <p>Plot top 5 cbgs in map (raw visitor count)</p>

# In[25]:


#First argument is the number of cbgs (top 1000 in this case) to be plotted
#column_name is the name of the column which we will sort by the dataset. In this case 'raw visitor count'
#geo_data is the dataframe containing cbg ket, latitude and longitude in the format needed to be plotted
#10 is the size of the dotis in the map
plot_most_popular_cbgs_in_map(1000,column_name,geo_data,10)


#  <p>Plot top 10 cbgs in map (raw visitor count)</p>

# In[26]:


plot_most_popular_cbgs_in_map(10,column_name,geo_data,100)


#  <p>List of top 10 cbgs in dataset (raw visitor count)</p>

# In[27]:


get_most_popular_cbgs(10,column_name)


# <b>Plot popularity of days of the week</b>

# <p>First, get the sum of all days of the week in common3</p>

# In[28]:


#Gets the total amount of visits in all cbgs for every day of the week (common3[week_list].sum) 
#and formats it to work with plot methods (.tolist())
#skips Nan values to avoid doing unnecesary calculations
week_sum = common3[week_list].sum(axis = 0, skipna = True).tolist()


# In[29]:


#Plots the histogram where a bar is the total amount of visits of a day (In x axis 1 is Monday, 2 is Tueday, etc)
plot = plt.bar(range(1,8), height = week_sum) #range(1,8) gives the amount of bars to be plotted (7)
                                              #height is the value of each bar in y  
plt.show()


# <b>Plot popularity of hours in a day</b>

# In[30]:


#The same logic that was used in the last lines
hour_sum = common3[hour_list].sum(axis = 0 ,skipna = True).tolist()


# In[31]:


plot = plt.bar(range(0,24), height = hour_sum) 
plt.show()


# <b>Plot top x cbgs (with the number x given)</b>

# In[32]:


def plot_most_popular_cbgs_in_hist(number,column_name):
    
    cbg_list = common3[['census_block_group',column_name]]#Gets dataframe with dataframe and the column to compare the amount of visits from
    cbg_list = cbg_list.sort_values(by = column_name, ascending = False)#Sort the values in dataframe 
    height =  cbg_list[column_name].iloc[0:number].tolist()# Get only the top x values we need
    
    plt.bar(range(1,number + 1), height = height) # Plot the hist
    


# <p>Top 10</p>

# In[33]:


#The first parameter is the number of top cbgs we want to find
#The second is the column we will use to get the visits of each cbg
plot_most_popular_cbgs_in_hist(10,column_name)


# <p>Top 20</p>

# In[34]:


plot_most_popular_cbgs_in_hist(20,column_name)


# <p>Top 100</p>

# In[35]:


plot_most_popular_cbgs_in_hist(100,column_name)


# <p>Top 500</p>

# In[36]:


plot_most_popular_cbgs_in_hist(500,column_name)


# <p>Top 1000</p>

# In[37]:


plot_most_popular_cbgs_in_hist(1000,column_name)


# <b>Now, divide days in categories (week days and weekends)</b>
# <p>First find cbgs where the most popular day is Monday</p>

# In[38]:


#week_list has a list of all days of the week
# we will append census block group because it's needed to make operations
week_list_with_cbg = ['census_block_group'] + week_list


# In[39]:


weekday_list = list()
weekend_list = list()


# In[40]:


week_days = week_list[0:4]#Gets days from Monday to Thursday
weekend_days = week_list[4:7]#Gets days from Friday to Sunday


# In[41]:


week_days


# In[42]:


common3[week_list].head()


# In[43]:


def max_in_weekday(row):
    return row.idxmax(axis = 1) in week_days 
#So this methods has to parts:
#First it finds the element of the row where the highest value is (row.idxmax())
#Axis  = 1 means it will return the name of the column where the element was found and not the element itself
#Next we compare if that value is in the list of week_days (from Monday to Thursday)
#Then return True or False given the result of that operation


# In[44]:


for index,value in common3[week_list_with_cbg].iterrows():#Iterrows() allows to iterate over every row in a dataframe
                                                          #values is the row itself (cbg columb, monday to friday visits columns)
                                                          #index is the index in the datafram (0,1,2,etc)
    if max_in_weekday(value[1:8].astype(int)):#if the highest value of the row is in a weekday
                                              #astype(int) is only formatting necessary to make operations
        weekday_list.append(value[0])#We append the value to the list of cbgs where the highest visitor count is in a weekday
        
    


# In[45]:


len(weekday_list)


# In[46]:


#To obtain the list of cbgs where the highest visitor count is in a weekend day 
#We only have to find which values are not in the weekday cbg dataset (.difference())
weekend_list = list(set(common3['census_block_group']).difference(set(weekday_list)))


# In[47]:


len(weekend_list)


# <p>Top 10 brands in weekday</p>

# In[48]:


#Gets rows from main dataset where highest visitor count is in weekdays
weekday_df = common3[common3['census_block_group'].isin(set(weekday_list))]


# In[49]:


#Gets rows from main dataset where highest visitor count is in weekend days
weekend_df = common3[common3['census_block_group'].isin(set(weekend_list))]


# In[50]:


#Detail if every column in the dataset recently created
weekday_df.info()


# <p>Now we can compare differences of top brands between categories</p>

# In[51]:


#Find top x brands of every column related to brands in weekdays dataset
brands_10_weekday_day = find_top_x_brands_with_vals(weekday_df['related_same_day_brand'],10)
brands_10_weekday_month = find_top_x_brands_with_vals(weekday_df['related_same_month_brand'],10)
brands_10_weekday_all = find_top_x_brands_with_vals(weekday_df['top_brands'],10)


# In[52]:


brands_10_weekday_day
#Uncomment these lines if you wnat to see the result of the other brand columns
#brands_10_weekday_month
#brands_10_weekday_all


# In[53]:


#Find top x brands of every column related to brands in weekend days dataset

brands_10_weekend_day = find_top_x_brands_with_vals(weekend_df['related_same_day_brand'],10)
brands_10_weekend_month = find_top_x_brands_with_vals(weekend_df['related_same_month_brand'],10)
brands_10_weekend_all = find_top_x_brands_with_vals(weekend_df['top_brands'],10)


# In[54]:


brands_10_weekend_day
#brands_10_weekend_month
#brands_10_weekend_all


# <b>Now, divide hours in categories (morning, evening, etc)</b>
# 

# In[55]:


#The process with the hours is very similar to the process with week days and weekend days
hour_list_with_cbg = ['census_block_group'] + hour_list


# In[56]:


df_hours = common3[hour_list_with_cbg]


# In[57]:


morning_list = list()
afternoon_list = list()
evening_list = list()
early_morning_list = list()


# In[58]:


early_morning_hours =hour_list_with_cbg[slice(2,8)]
morning_hours = hour_list_with_cbg[slice(8,14)]
afternoon_hours = hour_list_with_cbg[slice(14,20)]
evening_hours = hour_list_with_cbg[slice(20,26)] + ['12am']


# In[59]:


morning_hours


# In[60]:


afternoon_hours


# In[61]:


evening_hours


# In[62]:


early_morning_hours


# In[63]:


common3[hour_list].iloc[-5:]


# In[64]:


def max_in_timeframe(row,timeframe):
    return row.idxmax(axis = 1) in timeframe


# In[65]:


def cbg_list_for_max(timeframe,dataframe):
    elem_list = list()
    for index,value in dataframe.iterrows():
        if max_in_timeframe(value[1:25].astype(int),timeframe):
            elem_list.append(value[0])
    return elem_list


# In[66]:


morning_list = cbg_list_for_max(morning_hours,common3[hour_list_with_cbg])


# In[67]:


len(morning_list)


# In[68]:


afternoon_list = cbg_list_for_max(afternoon_hours,common3[hour_list_with_cbg])


# In[69]:


len(afternoon_list)


# In[70]:


evening_list = cbg_list_for_max(evening_hours,common3[hour_list_with_cbg])


# In[71]:


len(evening_list)


# In[72]:


early_morning_list = cbg_list_for_max(early_morning_hours,common3[hour_list_with_cbg])


# In[73]:


len(early_morning_list)


# In[74]:


#Find rows from main dataset where the cbgs has the highest visit count in the morning
morning_df = common3[common3['census_block_group'].isin(set(morning_list))]


# In[75]:


morning_df.head()


# In[76]:


#Find rows from main dataset where the cbgs has the highest visit count in the afternoon

afternoon_df = common3[common3['census_block_group'].isin(set(afternoon_list))]


# In[77]:


#Find rows from main dataset where the cbgs has the highest visit count in the evening

evening_df = common3[common3['census_block_group'].isin(set(evening_list))]


# In[78]:


#Find rows from main dataset where the cbgs has the highest visit count in the early morning

early_morning_df = common3[common3['census_block_group'].isin(set(early_morning_list))]


# In[79]:


#Find top x brands of every column related to brands in morning dataset

brands_10_morning_day = find_top_x_brands_with_vals(morning_df['related_same_day_brand'],10)
brands_10_morning_month = find_top_x_brands_with_vals(morning_df['related_same_month_brand'],10)
brands_10_morning_all = find_top_x_brands_with_vals(morning_df['top_brands'],10)


# In[80]:


brands_10_morning_all


# In[81]:


brands_10_afternoon_day = find_top_x_brands_with_vals(afternoon_df['related_same_day_brand'],10)
brands_10_afternoon_month = find_top_x_brands_with_vals(afternoon_df['related_same_month_brand'],10)
brands_10_afternoon_all = find_top_x_brands_with_vals(afternoon_df['top_brands'],10)


# In[82]:


brands_10_afternoon_all


# In[83]:


brands_10_evening_day = find_top_x_brands_with_vals(evening_df['related_same_day_brand'],10)
brands_10_evening_month = find_top_x_brands_with_vals(evening_df['related_same_month_brand'],10)
brands_10_evening_all = find_top_x_brands_with_vals(evening_df['top_brands'],10)


# In[84]:


brands_10_evening_all


# In[85]:


brands_10_early_morning_day = find_top_x_brands_with_vals(early_morning_df['related_same_day_brand'],10)
brands_10_early_morning_month = find_top_x_brands_with_vals(early_morning_df['related_same_month_brand'],10)
brands_10_early_morning_all = find_top_x_brands_with_vals(early_morning_df['top_brands'],10)


# In[86]:


brands_10_early_morning_all


# #  Data Clustering
# 

# <p>Implementation with OPTICS algorithm</p>
# 

# In[87]:


import time
from math import radians, cos, sin, atan2, sqrt


# In[88]:


cbg_w_coordinates.head()    


# In[90]:


EARTH_RADIO = 6371e3; 
#haversine distance is a mathematical function to find the shortest distance between two points in an sphere
#You can find a highly more detailed explanation in https://www.movable-type.co.uk/scripts/latlong.html
#Several optimizations have been made, eg. using math library instead of numpy to save execution time
#also using multiplication with 0.5 instead of dividing by 2, since division is about 6 times slower for computers
#Passing the values of latitude directly in radians to avoid that extra calculation inside the method, etc
def haversine_distance(point1,point2):
    
    phi1 = point1[0]
    phi2 = point2[0]
    delta_phi = point2[0] - point1[0]
    delta_lambda = point2[1] - point1[1]
    a = (sin(delta_phi*0.5))**2 + cos(phi1)*cos(phi2)*(sin(delta_lambda*0.5))**2
    c = 2*atan2(sqrt(a),sqrt(1-a))
    
    return c #(EARTH_RADIO * c)/1000 if you want in kms


# In[91]:



haversine_distance(point1,point2)
#The value of distance it returns is small. To have sense for us we should make its conversion con km
#but for computer it's fine and it saves computing power by not doing that conversion


# In[92]:


#Obtain the values needed to make the analysis and clustering
test = cbg_w_coordinates[['latitude','longitude']]
test1 = geo_data.copy()


# In[93]:


#Conversion to radians to save time as explained before
test = test.apply(lambda row: np.radians(row))


# In[112]:



#Min samples is the amount of 'core' points needed to be recognized as a cluster
#core points are basically points in the 'middle' (or just semi-central) of a high density zone
#Metric is the way the algorithm will measure distance. Since we are working in a 'sphere' we will use haversine 
#(although Earth is not REALLY a sphere, but the error is acceptable, about 0.3%
#Max_exps is the maximun distance the algorithm will search for neighbors for a cluster
#since the haversine function returns very small values (above the distance was 0.04, which converted is about 268 km),
#the value in this function is also really small. With 0.01 a many points that weren't noise were considered as such,
#with 0.08 it takes WAY to long to run, I suggest to lower it to 0.05 and see it the results are good enough
clust = OPTICS(min_samples = 3, metric = haversine_distance, min_cluster_size= 3, max_eps = 0.08)


# In[ ]:


start_time = time.time()#Line to measure the time it takes to run the algorithm
clust.fit(test)#the process of clustering itself
print("--- %s seconds ---" % (time.time() - start_time))#Also to measure time


# In[96]:


space = np.arange(test.shape[0])#array with numbers from 0 to the number of cbgs
labels = np.asarray(clust.labels_)# cluster to which the data point belongs (has the same length and index that the main dataset)
test1['Cluster'] = labels #Add the cluster labels to the dataset with all values


# In[97]:


#There are 8622 clusters (from 0 to 8621, -1 is considered noise)
np.unique(labels)


# In[98]:


len(np.unique(labels))


# In[99]:


test1.head()


# <p>Save the clustered dataset to a csv file</p>
# 

# In[101]:


common3 = pd.concat([common3, test1['Cluster']],axis = 1, ignore_index= False, sort=False)


# In[102]:


common3.head()


# In[103]:


file_path = '/Users/Mariana/Desktop/project/Python-Project/EDA_datasets/'
common3_clustered = 'common3_clustered2.csv'


# In[104]:


common3.to_csv(path_or_buf = (file_path + common3_clustered))


# In[105]:


import random


# In[106]:


def generate_colors(num):
    """This function returns a random list of colors. It is useful to plot hundreds of cbgs and make sure they 
        are distinguishable from each other"""
    return ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num)]


# In[107]:


def gen_colors_list(num):
    """Makes a color list calling to generate_colors() and then adding  white color to the beginning of the list"""
    colors_list = list()
    cbgs = list(test1['Cluster'])
    colors  = ['#FFFFFF'] + generate_colors(num - 1)# FFFFFF is white, and it will always be the color of noise
    for cbg in cbgs:
        if cbg == -1:
            colors_list.append(colors[0])
        else:
            colors_list.append(colors[cbg+1])
    return colors_list


# In[108]:


def plot_clustered_cbgs_in_map(geo_data,dot_size):  
    """This method provides an easy way to plot the cbgs, the first arg is the number of cbgs, the the column (visitors per day, per week, etc), dot_size is the radio of the plotted dots"""
    cbgs_list = geo_data
    figure, ax = plt.subplots(figsize  = (50,30))
    us_map.plot(ax = ax, alpha = 0.5, color = "grey")#Plots US map
    clusters =  list(np.unique(test1['Cluster']))#Gets all unique clusters to create the colors list
    colors = gen_colors_list(len(clusters))#Create the color list
    
    #Plots the clustered cbgs over the make,  each cluster with a given color
    cbgs_list.plot(ax = ax, markersize = dot_size, color = colors, marker = "o", label = "census block group")



# In[118]:


plot_clustered_cbgs_in_map(test1,1)


# ## Much faster implementation with HDBSCAN but the results are not that good
# <p>Implementation with HDBSCAN algorithm</p>
# 

# In[20]:


#You shouldn't run all this section since it takes a lot of time and information obtain is not useful
#It's just an example to see an implementation of a different algorithm
import hdbscan


# In[129]:


test = cbg_w_coordinates[['latitude','longitude']].iloc[0:1000]
test1 = geo_data.iloc[0:10000].copy()


# In[130]:


#The minimun cluster size should be around ~20000 people
clusterer = hdbscan.HDBSCAN(min_cluster_size = 10, min_samples = 40, metric = haversine_distance)


# In[109]:


test_hdbscan = test.apply(lambda x: np.radians(x))


# In[87]:


test_hdbscan.head()


# In[131]:


start_time = time.time()
clusterer.fit(test)
print("--- %s seconds ---" % (time.time() - start_time))


# In[123]:


labels = np.asarray(clusterer.labels_)# cluster to which the data point belongs
test1['Cluster'] = labels


# In[124]:


np.unique(labels)


# In[125]:


labels = np.asarray(clusterer.labels_)# cluster to which the data point belongs
test1['Cluster'] = labels


# In[126]:


plot_clustered_cbgs_in_map(test1,1)


# ### EDA with the clustered data

# In[6]:


file_path = '/Users/Mariana/Desktop/project/Python-Project/EDA_datasets/'
full_test_n = 'common3_clustered.csv'


# In[7]:


#Gets clustered dataset from file previously saved
full_test = pd.read_csv((file_path+full_test_n), dtype={'census_block_group':str},
                      converters={"related_same_day_brand": ast.literal_eval,
                                  "related_same_month_brand": ast.literal_eval,
                                  "top_brands" : ast.literal_eval })


# In[8]:


#Deletes unnecesary columns (unnamed, which were basically a repetition of the values in the index)
full_test = full_test.loc[:, ~full_test.columns.str.match('Unnamed')]
full_test.head()


# In[9]:


#Changes the format of visitor_home_cbgs column from str to dict to work with it
%%capture
full_test['visitor_home_cbgs'] = full_test['visitor_home_cbgs'].apply(lambda elem: ast.literal_eval(elem))


# In[10]:


full_test['visitor_home_cbgs'].head()


# <head>Top 10 brands per cluster</head>
# 

# In[11]:


#Simply gets the rows corresponding to a cluster and then reuses find_top_x_brands_with_vals() method
def top_brands_cluster(dataframe, cluster,number_brands,column_name):
    dataframe = dataframe[dataframe['Cluster'] == cluster][column_name]# gets the rows
    return  find_top_x_brands_with_vals(dataframe,number_brands)


# In[2]:


#full_test is the dataset we are working with
#1976 the cbg we want to analyze
#10 the amount of top brands we want to find
#top_brands the column we are analyzing
top_brands_cluster(full_test,1976,10,'top_brands')


# <p>Most popular clusters</p>
# 

# In[15]:


def total_visits_to_cluster(clusters_dict,row):
    cluster  = row['Cluster']#gets the cluster column value from the received row
    visits =  row[column_name]#gets the column_name value (eg. 'raw visit count') from the received row
    if cluster in list(clusters_dict.keys()):#if the cluster has appeared before
        clusters_dict[cluster] += visits #add the visits value to the visit total in the dict
    else:                                    #if it hasn't appeared before
        clusters_dict.update({cluster : visits}) # add it to the dict with the visits from the row as value


# In[16]:


def get_clusters_where_visits(criteria,clusters_num,column_name,dataframe):
    clusters_dict = dict()
    if clusters_num <= 0:
        print("Please enter a valid number of clusters")
        return
    
    if criteria == "come_to":
        dataframe = dataframe[['Cluster', column_name]]
        dataframe.apply(lambda row: total_visits_to_cluster(clusters_dict , row), axis = 1)#apply to every row
    elif criteria == "come_from":
    
        dataframe = dataframe[['census_block_group','Cluster', column_name]]
    #This part of the method used to be really slow given its nature, so the next line helps to save calculations
    #and make other caculations faster before even starting applying the funcion
    
        #First it assings cbgs as indexes of the dataframe (.set_index()) since index search is much faster than
        #searching values in a column of the dataframe
        #Then, it orders the indexes, optimizing even further the search 
        #(Unique and sorted indexes search order of complexity is very close to constant)
        #Comparing that to value search which is linear (~ O(1) vs O(n))
        dataframe = dataframe.set_index('census_block_group').sort_index()
        #Create the dict where we will save the results
        cbgs_cluster_dict = dict()
        #apply the function to every row
        dataframe.apply(lambda row: total_visits_from_cluster(clusters_dict , row ,dataframe,column_name, cbgs_cluster_dict), axis = 1)
    counter = Counter(clusters_dict)
    top_x = counter.most_common(clusters_num+1
    return top_x


# In[17]:


column_name = 'raw_visit_count'
top_10_to = get_clusters_where_visits('come_to',10,'raw_visit_count', full_test)
top_10_to   #every tuple represents the number of the cluster and the total visit count for it


# In[18]:


column_name = 'visitor_home_cbgs'


# In[19]:


full_test1 = full_test.copy()


# In[20]:



full_test1  = full_test1.set_index('census_block_group')


# In[21]:


#Example of the datasets with the cbgs as indexes
full_test1.sort_index()


# <p>Top 10 clusters where people visit from </p>
# 

# In[22]:


def total_visits_from_cluster(clusters_dict , row, dataframe , column_name , cbgs_cluster_dict):
    
    #clusters_dict has the format {cluster : visit_count} and is where we get results from
    #cbgs_cluster_dict has the format {cbg : cluster} and it helps us save doing calculations
    visits_cbg = row[column_name]# gets the dictionary containing the list of cbgs from which 
                                 #the row cbg gets visits from along with the amount of visits
    for cbg, visits in visits_cbg.items(): # for cbg in the dict
        if cbg in cbgs_cluster_dict:       #if the cbg has appeared before:
                                           #Checking this helps a lot to optimize since we don't have find the cluster of that
                                           #cbg again and we skip directly to updating its values in the corresponding dict
                    
            cluster = cbgs_cluster_dict[cbg] #Get the cluster from the dict where it's saved
            clusters_dict[cluster] += visits #update the value
        else:                              #If the cbg hasn't appeared before    
            cluster = int(dataframe.loc[cbg]['Cluster'])  #Find the cluster searching in the dataframe by index     
            if cluster in clusters_dict:     #If the cluster has appeared before
                 clusters_dict[cluster] += visits # Add the current visits to the visits it already has
            else:                            #If it hasn't appeared before
                 clusters_dict.update({cluster : visits}) # Add it to the corresponding dict with the current visit count as value
            
            cbgs_cluster_dict.update({cbg : cluster})  # add cbg and cluster to corresponding dict to save finding
                                                       # the cbg cluster if the cbg appears again in the dataset


# In[23]:


top_10_from = get_clusters_where_visits('come_from',10,column_name, full_test)


# In[24]:


top_10_from


# ### Plot a given cluster

# <p>Update geodata to include the cluster column</p>

# In[41]:


#Append cluster to the dataframe with the data needed to plot
geo_data_c = pd.concat([geo_data,full_test['Cluster']], axis = 1, ignore_index= False, sort=False)


# In[42]:


def plot_given_cluster_in_map(dataframe,cluster,dot_size):  
    """This method provides an easy way to plot the cbgs, the first arg is the number of cbgs, the the column (visitors per day, per week, etc), dot_size is the radio of the plotted dots"""
    cbgs_list = dataframe[dataframe['Cluster'] == cluster]
    figure, ax = plt.subplots(figsize  = (50,30))
    us_map.plot(ax = ax, alpha = 0.5, color = "grey")
    cbgs_list.plot(ax = ax, markersize = dot_size, color = 'blue', marker = "o")
    
    #Uncomment lines below to save the plot to a pdf file
    #plt.savefig("Cluster" + str(cluster) + ".pdf")
    #plt.legend(prop = {'size' : 15})




# In[43]:


plot_given_cluster_in_map(geo_data_c,1976,0.01)


# In[ ]:





# In[ ]:




