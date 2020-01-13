#!/usr/bin/env python
# coding: utf-8

# # Twitter Engagement Code
# ## Code to create a twitter object, extract relevant tweets, clean and do some frequemcy and sentiment analysis through the nltk package

# Create a pickle object to securely record the twitter authentication keys obtained from the twitter developer account

# In[1]:


import pickle
import os


# In[2]:


#use the generated keys from twitter, this code checks if these already exist, if not create a secret credentails file otherwise
# load previous credentials, credentials should not be visible in files for security reasons

if not os.path.exists('secret_twitter_credentials.pkl'):
    Twitter={}
    Twitter['Consumer Key'] = ''
    Twitter['Consumer Secret'] = ''
    Twitter['Access Token'] = ''
    Twitter['Access Token Secret'] = ''
    with open('secret_twitter_credentials.pkl','wb') as f:
        pickle.dump(Twitter, f)
else:
    Twitter=pickle.load(open('secret_twitter_credentials.pkl','rb'))


# In[3]:


# ensure that the Twitter package is installed
get_ipython().system('pip install twitter')


# In[4]:


get_ipython().system('pip install nltk')


# In[5]:


import numpy as np
import pandas as pd
import string
import nltk

nltk.download('stopwords')


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ##  Authorizing an application to access Twitter account data

# In[7]:


import twitter
#auth is an aithentication object
auth = twitter.oauth.OAuth(Twitter['Access Token'],
                           Twitter['Access Token Secret'],
                           Twitter['Consumer Key'],
                           Twitter['Consumer Secret'])

#twitter apu object, part of the Twitter Class
twitter_api = twitter.Twitter(auth=auth)

# Nothing to see by displaying twitter_api except that it's now a
# defined variable

type(twitter_api)


# # Look at global and local trends from Twitter
# 
# Twitter identifies locations using the Yahoo! Where On Earth ID.

# In[8]:


WORLD_WOE_ID = 1
US_WOE_ID = 23424977
CA_WOE_ID = 23424775

# Added for Toronto
LOCAL_WOE_ID=4118


# In[9]:



# Prefix ID with the underscore for query string parameterization.
# Without the underscore, the twitter package appends the ID value
# to the URL itself as a special case keyword argument.

world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)
us_trends = twitter_api.trends.place(_id=US_WOE_ID)
canada_trends = twitter_api.trends.place(_id=CA_WOE_ID)
local_trends = twitter_api.trends.place(_id=LOCAL_WOE_ID)


# In[10]:


trends=local_trends
print(type(trends))
print(list(trends[0].keys()))
#print(trends[0]['trends'])


# Using pretty printy JSON

# In[11]:


# Look at names of the local trends
for i in local_trends[0]['trends']:
    print(i['name'])


# In[12]:


import json

print((json.dumps(local_trends[0], indent=1)))


# Lets look at the current interections between trends in Toronto and the world overall

# In[13]:


# Create a dict to store sets
trends_set = {}
trends_set['world'] = set([trend['name'] for trend in world_trends[0]['trends']])

trends_set['canada'] = set([trend['name'] for trend in canada_trends[0]['trends']]) 

trends_set['toronto'] = set([trend['name'] for trend in local_trends[0]['trends']]) 


# In[14]:


trends_set['world']


# In[15]:


for loc in ['world','toronto','canada']:
    print(('-'*10,loc))
    print((','.join(trends_set[loc])))


# Lets look at an interection between these trends, these topics are trending both locally and globally!

# In[16]:


print(( '='*10,'intersection of world and toronto'))
print((trends_set['world'].intersection(trends_set['toronto'])))


# In[17]:


print(( '='*10,'intersection of canada and toronto'))
print((trends_set['canada'].intersection(trends_set['toronto'])))


# # Streaming Tweets
# Now lets stream tweets regarding a trending topic on twitter

# In[83]:


q = 'Toronto' 

number = 100

# https://developer.twitter.com/en/docs/tweets/search/FAQ

search_results = twitter_api.search.tweets(q=q, count=number, lang = 'en')

statuses = search_results['statuses']


# In[84]:


search_results.keys()


# In[85]:


# These are the list of keys stored in each dict of the status results
# I'm interested in extracting what people are saying which is in 'text' field 
statuses[0].keys()


# In[86]:


statuses[1]


# In[87]:


# First lets remove any duplicated results

all_text = []
filtered_statuses = []
for s in statuses:
    if not s["text"] in all_text:
        filtered_statuses.append(s)
        all_text.append(s["text"])
statuses = filtered_statuses     


# In[88]:


# number of unique statuses
len(statuses)


# In[89]:


status_list = [s['text'] for s in statuses]


# In[90]:


status_list


# In[91]:


# Store tweet texts in a pandas dataframe
Topic = "Recent Tweets about " + q
search_df = pd.DataFrame({Topic:status_list})
search_df


# In[92]:


# Create a utility function for cleaning of tweets text

def clean_text(tweet):
    ''' Takes in text and removes any words in the useless list'''
    useless = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    t = tweet.split()
    new_tweet = ""
    garbage = ['RT', q,'I','the','it','a','The','It']
    for word in t:
        if not word in useless:
            if word not in garbage:
                new_tweet = new_tweet + word + " "
    
    return new_tweet


# In[93]:


search_df['Clean Tweet'] = search_df[Topic].apply(clean_text)
search_df


# In[94]:


# Create a counter for the most common words 
# Counter takes in a list an creates a collection.counter object

from collections import Counter

words = []

for item in search_df['Clean Tweet']:
    for word in item.split():
        words.append(word)

c = Counter(words)

print(c.most_common()[:10]) # top 10

type(c)


# In[95]:


Top_words = c.most_common()[:20]
Top_words


# In[96]:


Top_words.pop(0)


# In[97]:


Top_words


# In[98]:


df2 = pd.DataFrame(Top_words)
df2


# In[99]:


df3 = df2.pivot_table(values=1,index=0)
df3.columns = ['Count']
df3


# In[100]:


df3.plot.bar(figsize=(15,10),color='rgb')
plt.title('Most Commont Words Frequency')
plt.ylabel('Word Frequency')
plt.xlabel('Word Frequency')


# In[102]:


# Lets look at the associated hastags for the searched query

hashtags = [ hashtag['text'] 
             for status in statuses
                 for hashtag in status['entities']['hashtags'] ]


# In[103]:


hashtags


# In[104]:


c2 = Counter(hashtags)
print(c2.most_common()[:15]) # top 10
print()


# In[ ]:




