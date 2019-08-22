#!/usr/bin/env python
# coding: utf-8

# # Step 1
# ## Importing data and viewing it

# <b> Import CSV 'cbg_patterns.csv' </b>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator


# In[2]:


pwd


# In[3]:


filename1 = '/Users/Mariana/Desktop/project/Python-Project/Dataset/cbg_patterns.csv'


# In[4]:


#Reading the data from the original dataset
data1 = pd.read_csv(filename1, dtype={'census_block_group':str})


# <b> Observe head </b>

# In[5]:


data1.head()


# <b> Observe data shape </b>

# In[6]:


data1.shape


# <b> Observe type </b>

# In[7]:


#Shows the type of every column in the dataset
data1.info()


# <b> Delete unnecessary columns </b>

# In[5]:


del data1['date_range_start']
del data1['date_range_end']


# Check that columns have disappeared

# In[9]:


data1.head()


# <b> Check if there are <i> null </i> rows in key column  </b>

# In[6]:


data1[data1['census_block_group'].isna()]


# <b> Remove <i> null </i> row/s in key column </b>

# In[7]:


data1 = data1.dropna(subset=['census_block_group'])


# Check that one row was removed

# In[12]:


data1.shape


# <b> Check statistics </b>

# In[13]:


data1.describe()


# <b> Import CSV 'cbg_geographic_data.csv' </b>

# In[8]:


filename2 = '/Users/Mariana/Desktop/project/Python-Project/Dataset/cbg_geographic_data.csv'
data2 = pd.read_csv(filename2, dtype={'census_block_group':str})


# <b> Observe head </b>

# In[15]:


data2.head()


# <b> Observe data shape </b>

# In[16]:


data2.shape


# <b> Observe type </b>

# In[17]:


data2.info()


# <b> Delete unnecessary columns </b>

# In[9]:


del data2['amount_land']
del data2['amount_water']


# Check that columns have disappeared

# In[19]:


data2.head()


# In[20]:


data2.shape


# <b> Check if there are <i> null </i> rows in key column  </b>

# In[21]:


data2[data2['census_block_group'].isna()]


# <b> Check statistics </b>

# In[22]:


data2.describe()


# Observations: data2 contains 220333 rows, data1 contains 220734 rows. This is a difference of 401 rows. 
# <p> I need to analyze 3 cases: </p>
#     <p> 1) Some rows are present in data1 and in data2 </p>
#     <p> 2) Some rows are present in data1 and not in data2 </p>
#     <p> 3) Some rows are present in data2 and not in data1 </p>

# <b> Case 1) Some rows are present in data1 and in data2 </b>

# In[23]:


s1 = data1.merge(data2)


# In[24]:


s1.shape


# 220331 rows are in common between data1 and data2

# In[25]:


s1[s1['census_block_group'].isna()]


# Second way to check common rows between data1 and data2

# In[10]:


common = data1.merge(data2,on=['census_block_group'])


# In[27]:


common.shape


# In[28]:


common.head()


# <b> Case 2) Some rows are present in data1 and not in data2 </b>

# In[30]:


#Convert cbg column in each dataset to set type to make set theory operations
s1 = set(data1['census_block_group'])
s2 = set(data2['census_block_group'])


# In[ ]:


#Find elements from s1 that are not present in s2
s3 = s1.difference(s2)
len(s3)


# <b> Case 3) Some rows are present in data2 and not in data1 </b>

# In[32]:


#Find elements from s2 that are not present in s1
s4 = s2.difference(s1)
len(s4)


# # Step 2
# #### Identify and group columns for first analysis
# <p> <b> Group 1: </b> Key column </p>

# <b> Select only column ‘census_block_group’ </b>

# In[36]:


common_key = common[['census_block_group']].copy()


# <b> Check that first character is a number and create a table counting the number of rows per first character </b>

# In[37]:


common_key['first'] = common_key['census_block_group'].str[:1]


# In[38]:


common_key.groupby('first').count()


# <b> Interpretation: </b> all cbgs start with a number in this range {0,1,2,3,4,5,7}, the majority starting with '3'

# <b> Check that last character is a number and create a table counting the number of rows per last character </b>

# In[39]:


common_key['last'] = common_key['census_block_group'].str[-1]


# In[40]:


common_key.groupby('last')['last'].count()


# <b> Interpretation: </b> all cbgs end with a number in this range {0:9}, the majority ending with '1'

# <b> Check length </b>

# In[41]:


common_key['slen'] = common_key['census_block_group'].str.len()


# In[42]:


common_key.groupby('slen')['slen'].count()


# <b> Interpretation: </b> The key column contains 220331 rows, all of length '12' 

# <b> Check that 'census_block_group' is a unique identifier </b>

# In[43]:


common_key['census_block_group'].is_unique


# <b> Group 2: </b> Columns starting with { </p>

# <b> Select only columns ‘visitor_home_cbgs’, ‘visitor_work_cbgs’, and ‘popularity_by_day’ </b>

# In[44]:


common_dict = common[['visitor_home_cbgs', 'visitor_work_cbgs', 'popularity_by_day']].copy()


# <b> Add index </b>

# In[45]:


common_dict['index_col'] = common_dict.index


# Check that index column was added

# In[46]:


common_dict['index_col'].head()


# <b> Unpivot other columns than Index </b>

# In[47]:


common_dict_melt = pd.melt(common_dict, id_vars=['index_col'])


# In[48]:


common_dict_melt.head()


# <p> <b> Remove {} <b/> </p>

# In[49]:


common_dict_melt['value'] = common_dict_melt['value'].map(lambda x: x.lstrip('{').rstrip('}'))


# In[50]:


common_dict_melt.head()


# In[51]:


common_dict_melt.iloc[-5:]


# <b> Split column by delimiter ',' </b>

# In[52]:


common_dict_melt_split = pd.concat([common_dict_melt, common_dict_melt['value'].str.split(',', expand=True)], axis=1)


# In[53]:


common_dict_melt_split.head()


# In[54]:


common_dict_melt_split.iloc[-3:]


# In[55]:


del common_dict_melt_split['value']


# In[56]:


common_dict_melt_split.head()


# <b> Filter on 'visitor_home_cbgs' </b>

# In[57]:


common_dict_melt_split_vis_home = common_dict_melt_split.loc[common_dict_melt_split['variable'] == 'visitor_home_cbgs']


# In[58]:


common_dict_melt_split_vis_home.head()


# In[59]:


common_dict_melt_split_vis_home_unpivot = pd.melt(common_dict_melt_split_vis_home, id_vars=['index_col','variable'])


# In[60]:


common_dict_melt_split_vis_home_unpivot.head()


# In[61]:


#calculate position of ':' for each row
# I should have "14" or "-1" or "NaN"
a = common_dict_melt_split_vis_home_unpivot['value'].str.find(':')


# In[62]:


a.shape


# In[63]:


# try 1
np.unique(a)


# In[64]:


# try 2
a.value_counts()


# In[65]:


# The sum of 14's and -1's is not the total, that means the other elements mus tbe nans,
#so the number of Nans is equal to:
print("Nan:    " + str(a.shape[0] - 4131721 - 28032))


# <b> Filter on 'visitor_work_cbgs' </b>

# In[66]:


common_dict_melt_split_vis_work = common_dict_melt_split.loc[common_dict_melt_split['variable'] == 'visitor_work_cbgs']


# In[67]:


common_dict_melt_split_vis_work.shape


# In[68]:


common_dict_melt_split_vis_work_unpivot = pd.melt(common_dict_melt_split_vis_work, id_vars=['index_col','variable'])


# In[69]:


common_dict_melt_split_vis_work_unpivot.head()


# In[70]:


#calculate position of ':' for each row
# I should have "14" or "-1" or "NaN"
b = common_dict_melt_split_vis_work_unpivot['value'].str.find(':')


# In[71]:


b.head()


# In[72]:


b.value_counts()


# In[73]:


#so the number of Nans is equal to:
print("Nan:    " + str(b.shape[0] - 1795704 - 50756))


# <b> Filter on 'popularity_by_day' </b>

# In[74]:


common_dict_melt_split_pop_day = common_dict_melt_split.loc[common_dict_melt_split['variable'] == 'popularity_by_day']


# In[75]:


common_dict_melt_split_pop_day.shape


# In[76]:


common_dict_melt_split_pop_day_unpivot = pd.melt(common_dict_melt_split_pop_day, id_vars=['index_col','variable'])


# In[77]:


common_dict_melt_split_pop_day_unpivot.head()


# In[78]:


#calculate position of ':' for each row
# I should have "8,9,10,11" or "-1" or "NaN"
c = common_dict_melt_split_pop_day_unpivot['value'].str.find(':')


# In[79]:


c.value_counts()


# In[80]:


#so the number of Nans is equal to:
print("Nan:    " + str(c.shape[0] - 660720 - 440480 - 220240 - 220240))


# <b> Check that there is a all days of week in each row, order is the same</b>

# In[81]:


#dictionary containing regex to compare to strings in every column
#the regex checks is the string contains a 'Monday', followed by ':' and followed by a number,
#and that's how it works for every single element
regex_dict = {0: "\"Monday\"\:[0-9]+", 1: "\"Tuesday\"\:[0-9]+", 2: "\"Wednesday\"\:[0-9]+",
              3: "\"Thursday\"\:[0-9]+", 4: "\"Friday\"\:[0-9]+", 5 :"\"Saturday\"\:[0-9]+", 6: "\"Sunday\"\:[0-9]+"}

# boolean list containing assertion on correctness of format, that is, to check if the elements are written correctly
bool_list = []
for n in range(7): # iterates from 0 to 6
    
    day_column = common_dict_melt_split_pop_day[n]# gets the column corresponding to a day (0 is monday, 1 tuesday and so on)
    evaluated_array = day_column.dropna().str.contains(regex_dict[n])#deletes nan values to avoid false positives and
                                                                     #evaluates every row to check is the format is correct
                                                                     #Returns an array of boolean values the same length that the original dataset
                                                                     #True means it has the right format and False means 
                                                                     #it's either empty or badly formatted
    bool_list.append(np.all(evaluated_array))#Evaluates all values in boolean for every column and if all values of column
                                             #Are correctly formatted it returns a single True in a list, if even a single
                                             #value is false it returns False
                                             # Then those values are appended to a list, every value corresponding to a day


# In[82]:


bool_list


# So only the Monday column contains bad formatted values. Check what indexes are these values from:

# In[83]:


monday_column = common_dict_melt_split_pop_day[0]
evaluated_monday_column = monday_column.dropna().str.contains(regex_dict[0])


# In[84]:


#These are the values that cointain empty rows in 'Monday' column
np.where(evaluated_monday_column == False)


# and those indices contain the value of an empty string ('')

# In[85]:


monday_column.iloc[220240:220330]


# <p> <b> Group 3: </b> Columns of type float </p>

# <b> Select only columns ‘raw_visit_count’, ‘raw_visitor_count’, and ‘distance_from_home’ </b>

# In[86]:


common_num = common[['raw_visit_count', 'raw_visitor_count', 'distance_from_home']].copy()


# <b> Add index </b>

# In[87]:


common_num['index_col'] = common_num.index


# <b> Unpivot other columns than Index </b>

# In[88]:


common_num_melt = pd.melt(common_num, id_vars=['index_col'])


# In[89]:


common_num_melt.head()


# In[90]:


common_num_melt.info()


# In[91]:


common_num_melt.describe()


# In[92]:


plt.hist(np.log(common_num_melt['value']))
plt.show()


# <p> <b> Group 4: </b> Columns containing text of brands </p>

# <b> Select only columns ‘related_same_day_brand’, ‘related_same_month_brand’, and ‘top_brands’ </b>

# In[93]:


common_brand = common[['related_same_day_brand', 'related_same_month_brand', 'top_brands']].copy()


# In[94]:


common_brand.head()


# <p> <b> Group 5: </b> Column 'popularity_by_hour' </p>

# <b> Select only columns ‘popularity_by_hour’ </b>

# In[95]:


common_poph = common[['popularity_by_hour']]


# In[96]:


common_poph.head()


# # Step 3
# #### Split, format, rename columns and deal with NaN values

# 1) Column "popularity_by_hour"

# In[11]:


#observe structure of column "popularity_by_hour"
common['popularity_by_hour'].head()


# In[12]:


#split column popularity_by_hour into one column per hour
common.columns.str
common1 = pd.concat([common, common['popularity_by_hour'].str.split(',', 24, expand=True)], axis = 1)
common1.head()


# In[13]:


common1.shape


# In[14]:


# keep only the number, remove []
common1[0] = common1[0].str.extract('(\d+)')
common1[23] = common1[23].str.extract('(\d+)')
common1.head()


# In[15]:


#rename columns
common1.rename(columns={0: '12am', 1: '1am', 2: '2am', 3: '3am', 4: '4am', 5: '5am', 6: '6am', 7: '7am', 8: '8am', 9: '9am', 10: '10am', 11: '11am', 12: '12pm', 13: '1pm', 14: '2pm', 15: '3pm', 16: '4pm', 17: '5pm', 18: '6pm', 19: '7pm', 20: '8pm', 21: '9pm', 22: '10pm', 23: '11pm'}, inplace=True)
common1.head()


# In[16]:


# delete original "popularity by hour" column
del common1['popularity_by_hour']


# In[17]:


common1.shape


# In[18]:


#check for empty rows
df1 = common1.iloc[:,12:36]
df1.head()


# In[19]:


#Indexes where the empty columns are
df1[df1.isna().all(axis=1)].index


# In[20]:


df1[df1.isna().all(axis=1)].shape


# 91 rows are empty. Since these are unvaluable rows, I will delete the full 91 rows from my "common" dataset.

# In[21]:


common2 = common1.drop(df1[df1.isna().all(axis=1)].index)
common2.shape


# In[22]:


#Next two lines format the output so all the columns are visible when head() method is called
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
common2.head()


# 2) Column "popularity by day"

# In[23]:


#Get a dataframe with the popularity by day
#dict(eval(x)) is the function that formats everything that way
pop_day = common2['popularity_by_day'].apply(lambda x : dict(eval(x))).apply(pd.Series)
pop_day.head()


# In[24]:


pop_day.shape


# Check for empty rows

# In[84]:


pop_day[pop_day.isna().all(axis=1)].index


# In[41]:


pop_day[pop_day.isna().all(axis=1)].shape


# In[111]:


common2.head()


# No empty rows.

# #### Calculate the ratio of visitors per week and per day
# <p> First, per week </p>

# In[78]:


data4 = pop_day.loc[:,["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]
data4['counter'] = range(len(data4))
data4.head()


# In[79]:


data5=pd.melt(data4, id_vars="counter").groupby(["counter"],axis=0).sum()


# In[80]:


data5.head()


# In[81]:


data6=data4.iloc[:,0:7].div(data5["value"],axis=0)
data6.head()


# 
# <p> Another way to calculate pop_day (Without heavy resources consuming funtion melt() </p>

# In[25]:


week_list = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# In[26]:


data4 = pop_day.loc[:, week_list]# Gets all columns from Monday to Sunday
week_total = data4.sum(axis = 1, skipna = True) #Gets the total sum of every element in a row
                                                #That is, the total number of visits per week


# In[27]:


#get a week total to then divide every day by it
week_total.shape


# In[28]:


data4.head()


# In[29]:


#To find the ratio of visits in a day, we simply divide the visits in a day with the total of visits in that week (Which is in week total)
data4 = data4.loc[:,week_list].div(week_total, axis = 0)


# In[30]:


#change column names to append without problem to main dataset
data4.rename(columns={"Monday": "Monday(ratio)", "Tuesday": "Tuesday(ratio)", "Wednesday": "Wednesday(ratio)", "Thursday": "Thursday(ratio)", "Friday": "Friday(ratio)", "Saturday": "Saturday(ratio)", "Sunday": "Sunday(ratio)"}, inplace=True)


# 
# <p> The type of all the columns is float: </p>

# In[31]:


data4.info()


# <p> 2) Popularity per day </p>

# In[32]:


hour_list = ['12am','1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']


# In[33]:


#Get only columns corresponding to visits per hour
data7 = common2.loc[:,hour_list]


# <p> Columns are not float type </p>

# In[34]:


data7.head()


# In[35]:


#Get rid of '[]' characters at the beginning and at the end of the data
data7['12am'] = data7['12am'].str.extract('(\d+)')
data7['11pm'] = data7['11pm'].str.extract('(\d+)')


# <p> Convert all columns to float </p>

# In[36]:


#Convert to float in order to make operations with them
data7 = data7.astype(float)


# In[37]:


#Check everything is converted to float
data7.info()


# In[38]:


data7.head()


# In[39]:


#The sum of all the elements in a row  (every hour) giving the total in a day
day_total = data7.sum(axis = 1, skipna = True)


# In[40]:


day_total.head()


# In[41]:


#Find the ratio of every hour dividing the value of every hour with the day_total
data8 = data7.loc[:,hour_list].div(day_total, axis = 0)


# In[42]:


data8.head()


# In[43]:


#change column names to append without problem to main dataset
data8.rename(columns={  "12am": "12am(ratio)", "1am": "1am(ratio)", "2am": "2am(ratio)", "3am": "3am(ratio)", "4am": "4am(ratio)", "5am": "5am(ratio)", "6am": "6am(ratio)", "7am": "7am(ratio)", "8am": "8am(ratio)", "8am": "8am(ratio)", "9am": "9am(ratio)", "10am": "10am(ratio)", "11am": "11am(ratio)", "12pm": "12m(ratio)", "1pm": "1pm(ratio)", "2pm": "2pm(ratio)", "3pm": "3pm(ratio)", "4pm": "4pm(ratio)", "5pm": "5pm(ratio)", "6pm": "6pm(ratio)", "7pm": "7pm(ratio)", "8pm": "8pm(ratio)", "9pm": "9pm(ratio)", "10pm": "10pm(ratio)", "11pm": "11pm(ratio)"}, inplace=True)


# In[44]:


data8.head()


# Delete string version od pop_hour in common2:

# In[45]:


common2.shape


# In[46]:


common2.drop(hour_list,axis = 1, inplace = True)


# In[47]:


common2.head()


# Append float version of popularity by hour to common2

# In[48]:


common3 = pd.concat([common2, data7], axis = 1, ignore_index= False, sort=False)


# In[49]:


common3.shape


# Append ratio version of popularity by hour to common2

# In[50]:


common3 = pd.concat([common3, data8], axis = 1, ignore_index= False, sort=False)


# In[51]:


common3.head()


# Append float version of popularity by day to common2

# In[52]:


common3 = pd.concat([common3, pop_day], axis = 1, ignore_index= False, sort=False)
common3.head()


# In[53]:


common3.head()


# Append "popularity by day (ratio)"

# In[54]:


common3 = pd.concat([common3, data4], axis = 1, ignore_index= False, sort=False)


# In[55]:


common3.shape


# In[56]:


del common3['popularity_by_day']


# In[57]:


common3.shape


# 2) Column "visitor_home_cbgs"

# In[58]:


#Get a copy of the column
test = common3['visitor_home_cbgs'].copy()


# In[59]:


test.head()


# In[60]:


test_2 = test.iloc[0]


# In[61]:


#check structure of one row
print(test_2)


# <b>Convert everything to dict type</b>

# In[62]:


import ast


# In[63]:


#Since 'visitor_home_cbgs' is a string, to make operations with it we need to convert it to a dict object
#that's what ast.literal_eval() does
test = test.apply(lambda x : ast.literal_eval(x)) 


# In[64]:


test.shape


# In[65]:


test_2 = test.iloc[0]


# <b>Find if elements are the correct type for the data set</b>

# In[66]:


#The elements have been corrected to the correct format
print(type(test_2))


# <b>Find if elements are the correct type for the data set</b>

# In[67]:


type(test.iloc[0]['010059501003'])


# <b>Since they are, test column can be appended to main dataframe</b>
# <p>First, delete original column 'visitor cbgs' with string type</p>

# In[68]:


del common3['visitor_home_cbgs']


# In[69]:


common3 = pd.concat([common3,test],axis = 1, ignore_index= False, sort=False)


# In[70]:


common3.head()


# <p>Create column with the number of visitor home cbgs visitings a given cbg</p>

# In[71]:


common3['vis_home_cbgs_count'] = test.apply(lambda x: len(x))


# In[72]:


common3.head()


# <p>Create column with the visitor home cbg where the highest amount of visitors come:</p>

# In[73]:


common3['vis_home_most_visits_from'] = test.apply(lambda x: max(x.items(), key=operator.itemgetter(1))[0] if x else None)


# In[74]:


common3['vis_home_most_visits_from'].iloc[0:10]


# <p>Create column with the visitor home cbg where the lowest amount of visitors come (but visitors are higher than zero):</p>

# In[75]:


common3['vis_home_least_visits_from'] = test.apply(lambda x: min(x.items(), key=operator.itemgetter(1))[0] if x else None)


# In[76]:


common3.head()


# <p>Create column with the average amount of visitors from all cbgs in 'visitor_home_cbgs':</p>

# In[77]:


#to find the average, first find the total amount of visitors (sum(list(x.values()))
#the divide it by the amount of cbgs where that cbg gets visits from (len(x))
common3['avg_vis_home_visitors'] = test.apply(lambda x: sum(list(x.values()))/len(x) if x else 0)


# In[78]:


common3.head(5)


# 3) Column "visitor_work_cbgs"

# In[79]:


vis_work_cp = common3['visitor_work_cbgs'].copy()


# In[80]:


vis_work_cp = vis_work_cp.apply(lambda x : ast.literal_eval(x))


# In[81]:


del common3['visitor_work_cbgs']


# In[82]:


#append 'visitor_home_cbgs'  with the correct type to main dataset 
common3 = pd.concat([common3,vis_work_cp],axis = 1, ignore_index= False, sort=False)


# In[178]:


common3.head()


# In[83]:


#Append column with the count of the cbgs that visit the cbg from the row to main dataset
common3['vis_work_cbgs_count'] = vis_work_cp.apply(lambda x: len(x))


# In[180]:


common3.head()


# In[84]:


#Finds the  cbg where the most visits come from (that is max()) for every row
#key=operator.itemgetter(1))[0] is to serach only in the values element of the
#row (key: cbg, value : visit count), since it is an object with more info
common3['vis_work_most_visits_from'] = vis_work_cp.apply(lambda x: max(x.items(), key=operator.itemgetter(1))[0] if x else None)


# In[182]:


common3.head()


# In[85]:


#Same that the method above but to find the least visits (min())
common3['vis_work_least_visits_from'] = vis_work_cp.apply(lambda x: min(x.items(), key=operator.itemgetter(1))[0] if x else None)


# In[184]:


common3.head()


# In[86]:


common3['avg_vis_work_visitors'] = vis_work_cp.apply(lambda x: sum(list(x.values()))/len(x) if x else 0)


# In[186]:


common3.head()


# In[187]:


common3.shape


# In[188]:


common3.shape


# # Step 4
# #### Identify Top ten brands for each column
# 

# <p>First convert columns to the right format</p>

# In[88]:


def clean_brands(brands_list):
    """Removes spaces and punctuation from every single brand in the dataset, also makes all letters lowercase"""
    return [brand.replace(" ", "").replace("\'","").lower() for brand in brands_list]
            
def count_brands_frequency(brands_list,brands_dict):
    """Finds the amount of times a brand has appeared in the dataset as a dict with the format{brand : frequency}"""
    for brand in brands_list:#For every brand in the row
        if brand in brands_dict:# If the brand has appeared before
            brands_dict[brand] += 1 #adds 1 to the times the brand has appeared in the dataset
        else: #If it's the first time the brand has appeared
            brands_dict.update({brand : 1})#add the brand to the dictionary of brands a give it a count of 1
            
def find_top_x_brands(dataframe,number):
    """Finds most popular brands"""
    brands_dict = dict()
    dataframe.apply(lambda brands_list : count_brands_frequency(brands_list, brands_dict))# Find the frequency of each brand
    counter = Counter(brands_dict) #Type convertion needed to sort the elements considering a dict in python has no order
    top_x = counter.most_common(number) #Find top x brands
    top_x = [elem[0] for elem in top_x] #return only the brand and not the count
    
    return top_x

#Same as the method above, only the last line is different
def find_top_x_brands_with_vals(dataframe,number):
    """Finds most popular brands and the frequency"""
    brands_dict = dict()
    dataframe.apply(lambda brands_list : count_brands_frequency(brands_list, brands_dict))
    counter = Counter(brands_dict)
    top_x = counter.most_common(number)#Returns the count
    
    return top_x


# In[89]:


from collections import Counter


# In[90]:


#Same as before, the data is in str format so we convert it to a type where we can make operations with it (list)
common3['related_same_day_brand']  = common3['related_same_day_brand'].apply(lambda x : ast.literal_eval(x))


# In[97]:


#Delete punctuation and spaces from brands, and make them all lowercases, next 4 rows do the same for the 
#detailed columns
common3['related_same_day_brand']  = common3['related_same_day_brand'].apply(lambda x : clean_brands(x) if x else [])


# In[92]:


common3['related_same_month_brand'] = common3['related_same_month_brand'].apply(lambda x : ast.literal_eval(x))


# In[98]:


common3['related_same_month_brand'] = common3['related_same_month_brand'].apply(lambda x : clean_brands(x) if x else [])


# In[94]:


common3['top_brands']               = common3['top_brands'].apply(lambda x : ast.literal_eval(x))


# In[99]:


common3['top_brands']               = common3['top_brands'].apply(lambda x : clean_brands(x) if x else [])


# <p>Now make all brands lowercase and punctuation-less</p>

# In[100]:


top_10_list_day_names =find_top_x_brands(common3['related_same_day_brand'],10)


# In[101]:


top_10_list_day_names


# In[102]:


def create_df_brands(timeframe,top_10):
     """Creates a datafrme where the top 10 brands are columns and 1 or 0 represent if they are in the top brands of a given cbg"""
    brand_df = pd.DataFrame()#Create an empty dataset to work with it later
    brand_df['census_block_group'] = common3['census_block_group'].copy()
    brand_column = pd.Series()#Create an empty dataset to work with it later
    
    for brand in top_10:
    #First create the empty column for the brand in the df brand_day_df
        brand_column = np.nan #np.nan is to fill the values since it's necessary taht the column has values
    #Then, checks whether the brand is on the list of brands in every row and puts a 0 or 1 depending the result
        brand_column = common3[timeframe].apply(lambda list_b : 1 if brand in list_b else 0)
    #finally, append to brand_day_df
        brand_df = pd.concat([brand_df,brand_column],axis = 1, ignore_index= False, sort=False)
    brand_df.columns = ['census_block_group'] + top_10
        
    return brand_df


# In[103]:


brand_day_df = create_df_brands('related_same_day_brand',top_10_list_day_names)#datagrame for 'related_same_day_brand' column


# In[104]:


brand_day_df.head()


# <p>Top ten brands for month</p>

# In[105]:


top_10_list_month_names = find_top_x_brands(common3['related_same_month_brand'],10)


# In[106]:


top_10_list_month_names


# In[107]:


brand_month_df =  create_df_brands('related_same_month_brand',top_10_list_month_names)


# In[108]:


brand_month_df.head()


# <p>Top ten brands for all time measured</p>

# In[109]:


top_10_list_top_names = find_top_x_brands(common3['top_brands'],10)


# In[110]:


top_10_list_top_names


# In[111]:


brand_top_df =  create_df_brands('top_brands',top_10_list_top_names)


# In[112]:


brand_top_df.head()


# In[113]:


brand_top_df.shape


# In[114]:


file_path = '/Users/Mariana/Desktop/project/Python-Project/EDA_datasets/'
common3_n = 'common3.csv'
brand_day_df_n = 'brand_day_df.csv'
brand_month_df_n = 'brand_month_df.csv'
brand_top_df_n = 'brand_top_df.csv'


# In[115]:


#save files in the computer to work with them later
common3.to_csv(path_or_buf = (file_path + common3_n))
brand_day_df.to_csv(path_or_buf = (file_path + brand_day_df_n))
brand_month_df.to_csv(path_or_buf = (file_path + brand_month_df_n))
brand_top_df .to_csv(path_or_buf = (file_path + brand_top_df_n))


# In[ ]:




