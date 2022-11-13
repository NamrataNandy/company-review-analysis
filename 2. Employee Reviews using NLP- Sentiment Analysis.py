#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

get_ipython().system('pip install textblob')
from textblob import TextBlob #For processing textual data


# In[62]:


df=pd.read_csv(r'C://Users/Namrata/Desktop/Corporate Projects/company-review-analysis-master/dataset/df_cleaned.csv', index_col=[0], parse_dates=['Comment Datetime'])


# In[63]:


df.head()


# In[64]:


# Sentiment analysis of summary column

pol=lambda x:TextBlob(x).sentiment.polarity
sub=lambda x: TextBlob(x).sentiment.subjectivity

df['polarity']=df['Summary'].apply(pol)
df['subjectivity']=df['Summary'].apply(sub)


# In[65]:


df.reset_index(inplace=True)


# In[66]:


df.head()


# In[67]:


df[['Summary','polarity','subjectivity']]


# In[68]:


df.loc[6383,'Comment Datetime']

print(df.loc[6383,'Con'])


# In[69]:


df.head()


# In[70]:


# Plot polarity and subjectivity

def polarity_subjectivity(pol,sub,column):
    ax,fig=plt.subplots(figsize=(10,10))
    plt.scatter(x=pol, y=sub,data=df,c=pol)
    plt.vlines(x=0,ymin=0,ymax=1,linestyles='dashed',colors='r')
    plt.title("Polarity vs Subjectivity",fontsize=20)
    plt.xlabel("Polarity",fontsize=10)
    plt.ylabel("Subjectivity",fontsize=10)
    plt.xticks([-1,0,1],['Negative','Neutral','Positive'])
    plt.show()


# In[71]:


polarity_subjectivity(pol='polarity',sub='subjectivity',column='Summary')


# ### Findings
# It is clear that there are more positive comments than negative as we see more data points on the right side of the red vertical line. This verified what we have observed from the precious EDA.
# The more polarized the comment is (either positive or negative), the more subjective it is.
# We also notice that reviewers who give positive comments based more on facts (lower subjectivity) and reviewers who give negative comments based more on opinion.
# 

# In[72]:


# plot most freq 10 job titles for sentiment analysis
df['Job Title'].value_counts().nlargest(10)


# In[73]:


# combine similar categories for job title
df['Job Title']=df['Job Title'].str.replace('Financial Services Representative', 'Financial Representative')
df['Job Title'] = df['Job Title'].str.replace('Software Engineer/Developer', 'Software Engineer')


# In[74]:


most_10_job=df['Job Title'].value_counts().nlargest(10).index.to_list()


# In[75]:


# Get the mean polarity and subjectivity of the top 10 job titles
pol_sub_job_title=df.groupby('Job Title')['polarity','subjectivity'].mean()
pol_sub_job_title_top10=pol_sub_job_title.loc[most_10_job]


# In[76]:


pol_sub_job_title_top10


# In[77]:


# plot the polarity and subjectivity of the most frequent 10 job titles

plt.rcParams['figure.figsize'] = [10, 8]

for i in pol_sub_job_title_top10.index:
    x = pol_sub_job_title_top10.loc[i,'polarity']
    y = pol_sub_job_title_top10.loc[i,'subjectivity']
    plt.scatter(x, y, color='blue')
    plt.text(x, y, i, fontsize=10)

plt.title("Most Frequent 10 Job Titles' Sentiment", fontsize=20)
plt.xlabel('Average Polarity', fontsize=15)
plt.ylabel('Average Subjectivity', fontsize=15)

plt.show()


# Findings:
# 
# Financial Representatives, Project Managers and reviewers who do not disclose their titles (Anonymous Employee) tend to give the most negative comments in 'Summary'. In the next section (keywords extraction and topic modeling), we will dive into these groups and see what makes them unhappy.
# System Analysts and Software Engineers have the most positive comments.

# In[81]:


# sentiment analysis on column 'Pro' and column 'Con'

from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df['p_polarity'] = df['Pro'].apply(pol)
df['p_subjectivity'] = df['Pro'].apply(sub)

df['c_polarity'] = df['Con'].apply(pol)
df['c_subjectivity'] = df['Con'].apply(sub)


# In[83]:


polarity_subjectivity(pol='p_polarity',sub='p_subjectivity',column='Pro')


# In[84]:


polarity_subjectivity(pol='c_polarity',sub='c_subjectivity',column='Con')


# Findings:
# 
# It is not surprising to see the polarity of Pro mostly falls in the positive part because we know these are the nice things employees say about this company.
# It is surprising to see even in the Con column, half of the data points' polarity falls on the positive side. This is strong evidence that most employees love this company and they don't have many complaints.

# In[86]:


df.to_csv(r'C://Users/Namrata/Desktop/Corporate Projects/company-review-analysis-master/dataset/df_cleaned2.csv')


# In[ ]:




