#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required libraries
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


movies_data = pd.read_csv('movies[1].csv')


# In[3]:


#Printing the first 5 rows of the dataframe
movies_data.head()


# In[4]:


#Getting the number of rows and columns in the data frame
movies_data.shape


# In[5]:


#Selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[6]:


#Replacing the null values with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[7]:


#Combining all 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[8]:


print(combined_features)


# In[9]:


#Converting the text data to feature vectors
vectorizer = TfidfVectorizer()


# In[10]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[11]:


print(feature_vectors)


# In[12]:


#Getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)


# In[13]:


print(similarity)


# In[14]:


print(similarity.shape)


# In[15]:


#Getting the movie name from the user
movies_name = input(' Enter your favourite movie name : ')


# In[16]:


#Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[24]:


#Finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movies_name,list_of_all_titles)
print(find_close_match)


# In[25]:


close_match = find_close_match[0]
print(close_match)


# In[30]:


#Finding the index of the movie with title
index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_movie)


# In[32]:


#Getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_movie]))
print(similarity_score)


# In[34]:


len(similarity_score)


# 

# In[36]:


#Sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)


# In[41]:


#print the name of similar movies based on the index
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if(i<30):
        print(1,'_',title_from_index)
        i==1


# In[43]:


#MOVIE RECOMMENDATION SYSTEM
movies_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movies_name,list_of_all_titles)

close_match = find_close_match[0]

index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if(i<30):
        print(1,'_',title_from_index)
        i==1


# In[ ]:




