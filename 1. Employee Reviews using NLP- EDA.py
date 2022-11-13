#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import re # to support modifiers, identifiers or white space char
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


# In[77]:


# Read the data, assign column index and modify 'Comment Date' using parse_dates- Pandas will attempt to infer the format of the datetime strings.
df=pd.read_csv(r'C://Users/Namrata/Desktop/Corporate Projects/company-review-analysis-master/dataset/company1_review.csv', index_col=[0], parse_dates=['Comment Datetime'])


# In[78]:


df.head()


# In[79]:


df.shape


# In[80]:


df.columns


# In[81]:


df.info()


# In[82]:


df.isnull().sum()


# In[83]:


df.drop_duplicates(inplace=True)


# In[84]:


df.shape


# In[85]:


ratings_col=df.select_dtypes(include='float64')
string_col=df.select_dtypes(exclude='float64')
df_col_sorted=pd.concat([string_col,ratings_col],axis=1)


# In[86]:


df_col_sorted.head()


# In[87]:


# Print the text of some samples 
for i in range(4,7):
    print("Author Years:",df['Author Years'][i])
    print('Author Title:', df['Author Title'][i])
    print('Author Summary:', df['Summary'][i])
    print('Author Pro:', df['Pro'][i])
    print('Author Con:', df['Con'][i])
    print('====================')


# In[88]:


def extract_cat_info(row):
#   Extract Current/former employee from Author Years  
    if "worked at" in row['Author Years']:
        row['Current Employee']=0
    elif "work at" in row['Author Years']:
        row['Current Employee']=1
    elif "have been working" in row['Author Years']:
        row['Current Employee']=1
    else:
        row['Current Employee']=np.NaN

# Extract tenure from Author Years
    string_to_number = row["Author Years"].replace("a year", "1 year")  # replace 'a year' with '1 year'
    tenure = re.findall(r'\d+', string_to_number)                       # find the digit in the string
    
    if tenure: 
        row['Tenure'] = int(tenure[0])                 # use the number in the list
        if 'more than' in row["Author Years"]:         
            row['Tenure'] += 0.5                       # add 0.5 year if there is 'more than'
        elif 'less than' in row["Author Years"]:       
            row['Tenure'] -=0.5                        # minus 0.5 year if there is 'less than'
    else:
        row['Tenure'] = np.NaN                         # if no tenure is specified, set to NaN
# Extract full time and part time from Author Years
    if "full-time" in row['Author Years'] or "full time" in row['Author Years']:
        row['Full-Time']=1
    elif "part-time" in row['Author Years'] or "part time" in row['Author Years']:
        row['Full-Time']=0
    else:
        row['Full-Time']=np.NaN

    row['Recommended'] = 0
    row['Positive Outlook'] = 0
    row['Approves of CEO'] = 0
    
    if not pd.isna(row['Recommendation']):  
        if 'Recommends' in row['Recommendation']:
            row['Recommended'] = 1
        elif "Doesn't Recommend" in row['Recommendation']:
            row['Recommended'] = -1
        elif 'Positive Outlook' in row['Recommendation']:
            row['Positive Outlook'] = 1
        elif 'Negative Outlook' in row['Recommendation']:   
            row['Positive Outlook'] = -1
        elif 'Neutral Outlook' in row['Recommendation']: 
            row['Positive Outlook'] = 0
        elif 'Approves of CEO' in row['Recommendation']:
            row['Approves of CEO'] = 1
        elif 'Disapproves of CEO' in row['Recommendation']:
            row['Approves of CEO'] = -1
        elif 'No opinion of CEO' in row['Recommendation']:   
            row['Approves of CEO'] = 0

    return row

df_extract_cat_info=df.apply(extract_cat_info,axis=1)
            


# In[89]:


df_extract_cat_info.head()


# In[90]:


# Extracting more categorical information such as State and Employee Title using Author Location and Job Title resp.
def extract_cat_info2(row):
    # 1. extract location
    if not pd.isna(row['Author Location']):
        if re.search(r'[A-Z]{2}$',row['Author Location']): 
            # extract the last 2 captical letters as state
            row['State'] = re.search(r'[A-Z]{2}$',row['Author Location'])[0]
        else:
            row['State'] = np.NaN      
    else:
            row['State'] = np.NaN                               
    
    # 2. extract job title
    if pd.notnull(row['Author Title']) and row['Author Title']: 
        if '-'in row['Author Title']:  # author title usually starts like this: "Current Employee - Analyst" 
            row['Job Title'] = row['Author Title'].split("-")[1]  # get the 2nd element after the split 
        else:
            row['Job Title'] = row['Author Title']
    else:
         row['Job Title'] = 'Unknown Title'
    # remove "senior" and "principal" to get fewer job categories 
    # remove the beginning & end spaces
    row['Job Title'] = row['Job Title'].replace('Senior',"").replace('Principal',"").strip() 
    
    return row       
df_loc_job_filled = df_extract_cat_info.apply(extract_cat_info2,axis=1) 


# In[91]:


df_loc_job_filled.head()


# In[92]:


df_cleaned=df_loc_job_filled.drop(columns=['Recommendation','Author Title', 'Author Location','Author Years'])


# In[93]:


# reorder the columns

df_cleaned = df_cleaned[['Comment Datetime', 'State', 'Job Title','Tenure','Current Employee','Full-Time',
                          'Summary','Pro','Con','Recommended', 'Positive Outlook','Approves of CEO',
                          'Overall Rating','Career Opportunities','Compensation and Benefits',
                          'Work/Life Balance','Senior Management','Culture & Values']]


# In[94]:


df_cleaned.set_index('Comment Datetime', inplace=True)


# In[95]:


df_cleaned=df_cleaned.sort_index()


# In[96]:


# sort dataframe by index

df_cleaned = df_cleaned.sort_index()
df_cleaned.head(1)


# In[39]:


# check missing values one more time

df_cleaned.isnull().sum()


# In[40]:


df_cleaned.describe()


# In[43]:


# Plot all ratings columns

col_list=['Overall Rating','Career Opportunities','Compensation and Benefits','Work/Life Balance','Senior Management','Culture & Values']

figure,axis=plt.subplots(1,6,figsize=(13,6))

for column,curr_ax in zip(col_list,axis.ravel()):
    curr_ax.boxplot(df_cleaned[column].dropna())
    curr_ax.set_title(f'{column}')
plt.tight_layout()
plt.show()


# In[ ]:


# On a scale of 1-5, this company's median overall rating is 4 which is pretty good. 
# Breaking down to 5 categories, employees rated the highest in compensation & benefits, work/life balance and culture & values. Senior management rating's median value is 3, which is the lowest, and career opportunities'median rating is 3.5. Culture & Values has a median rating as 4. 
# There are a few outliers in career opportunities and culture & values.


# In[44]:


# plot overall rating over the years

from datetime import datetime
fig,ax = plt.subplots(figsize=(10,5))

x = sorted(df_cleaned.index.year.unique())
y = df_cleaned.groupby(df_cleaned.index.year)['Overall Rating'].mean()

ax.plot(x, y, color='blue', marker = 'o',label='Overall Rating')
ax.set_title('Overall Rating vs. Year', fontsize=20)
ax.set_xlabel('Year')
ax.set_ylabel('Overall rating')
# plt.savefig('overall_rating_vs_year.png')

plt.show()  


# In[54]:


# Plot recommendation
reco_count=df_cleaned['Recommended'].value_counts(normalize=True)

fig,ax=plt.subplots()
ax.bar(['Recommend','Unknown','Not Recommend'],reco_count, color=['tab:orange','tab:olive','tab:blue'])
ax.set_title('Recommend or Not?')
ax.set_ylabel('Freq')

plt.show()


# In[62]:


# plot sub categories average ratings

column_list = ['Overall Rating','Career Opportunities','Compensation and Benefits',
               'Work/Life Balance','Senior Management','Culture & Values']

avg_ratings=df_cleaned[column_list].mean()
color_code=['tab:orange','tab:pink','tab:blue','tab:olive','tab:brown','tab:green']

fig,ax=plt.subplots(figsize=(12,5))
ax.bar(avg_ratings.index, avg_ratings,color=color_code)
ax.set_title('Average Ratings')
ax.set_ylabel('Freq')
ax.set_xticklabels(avg_ratings.index,rotation=45)
plt.show()


# In[63]:


# plot the 10 states with top ratings 

top_10 = df_cleaned.groupby('State')['Overall Rating'].mean().nlargest(10)
colors2 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

fig,ax = plt.subplots(figsize=(10,5))
ax.bar(top_10.index,top_10, color=colors2)
ax.set_title('Top 10 States with Highest Ratings', fontsize=20)
ax.set_xlabel('States')
ax.set_ylabel('Overall rating')
# plt.savefig('top10_states_high_rate.png')

plt.show()


# In[64]:


# plot the 5 states with lowest ratings 

lowest_5 = df_cleaned.groupby('State')['Overall Rating'].mean().nsmallest(5)
colors3 = ['tab:blue','tab:pink','tab:cyan','tab:orange','tab:purple']

fig,ax = plt.subplots(figsize=(10,5))
ax.bar(lowest_5.index,lowest_5, color=colors3)
ax.set_title('Top 5 States with Lowest Ratings',fontsize=20)
ax.set_xlabel('States')
ax.set_ylabel('Overall rating')
# plt.savefig('top5_states_low_rate.png')

plt.show()


# In[65]:


# check how many employees submitted feedback in NE since its rating is so low

len(df_cleaned.loc[df_cleaned.State=='NE'])


# In[ ]:


# There is actually only 1 employee in NE submitted feedback to Glassdoor.

# Looks like employees in KS and WI have the highest employee satisfaction and employees in NE (only 1 submitted review) are the unhappiest. Of note, the location data indicates 'Author Location', we assumed these employees are based locally in the company's branches in that particular state. If there are substantial remote employees, this conclusion may not be accurate.


# In[67]:


# plot overall rating by full-time/part-time employee

rate_by_fte = df_cleaned.groupby('Full-Time')['Overall Rating'].mean()

fig, ax = plt.subplots()
ax.bar(['Part_Time', 'Full_Time'], rate_by_fte,color=['tab:pink','tab:cyan'])
ax.set_title('Overall Ratings by Full/Part-time Employee', fontsize=20)
ax.set_ylabel('Overall rating')
# plt.savefig('rating_by_fulltime_parttime.png')

plt.show()


# In[72]:


# plot overall rating by current/former employee

rat_by_emp_type=df_cleaned.groupby('Current Employee')['Overall Rating'].mean()

color_emp_type=['tab:pink','tab:cyan']
fig,ax=plt.subplots()
ax.bar(['Former Employee', 'Current Employee'], rat_by_emp_type,color=['tab:pink','tab:cyan'])
ax.set_title('Overall Ratings by Current/Former Employee', fontsize=20)
ax.set_ylabel('Overall rating')
plt.show()


# In[73]:


# plot the most frenquest reviewer job titles

top_20_job = df_cleaned['Job Title'].value_counts().nlargest(20)

plt.figure(figsize=(12,7))
sns.countplot(y='Job Title',data=df_cleaned, order=top_20_job.index)
sns.set_context('talk')
plt.title('Most Frequest Employee Job Titles', fontsize=20)
# figure.savefig('most_freq_job_title.png',bbox_inches = 'tight')

plt.show()


# In[74]:


top_20job_review = df_cleaned.loc[df_cleaned['Job Title'].isin(top_20_job.index), ['Job Title','Overall Rating']]
top_20job_mean_review = top_20job_review.groupby('Job Title')['Overall Rating'].mean().sort_values(ascending = False)

# plot the reviewers in the top 20 job families' overall rating

plt.figure(figsize=(12,7))
sns.barplot(y=top_20job_mean_review.index, x=top_20job_mean_review, hue_order=top_20job_mean_review)
sns.set_context('talk')
plt.title('Overall Rating by Job Family', fontsize=20)
# figure.savefig('rating_by_job_family.png',bbox_inches = 'tight')

plt.show()


# In[ ]:


df_cleaned.to_csv(r'C://Users/Namrata/Desktop/Corporate Projects/company-review-analysis-master/dataset/df_cleaned1.csv')

