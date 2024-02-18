#!/usr/bin/env python
# coding: utf-8

# In[15]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
warnings.filterwarnings("ignore")


# # performing EDA(exploratory data analysis)

# # BOOKS

# In[16]:


try:
    books = pd.read_csv("D:\\DS new project\\Books.csv", encoding='ISO-8859-1')
except Exception as e:
    print(f"An error occurred: {e}")


# In[17]:


books


# In[18]:


books.info()


# In[19]:


books.describe()


# In[20]:


#checking of duplicate values


# In[21]:


books.loc[books.duplicated()]


# In[22]:


top_books = books['Book-Title'].value_counts().head(10)


# In[23]:


top_books


# In[24]:


top_books.index


# In[25]:


sns.barplot(x=top_books.values, y=top_books.index, palette='muted')


# In[26]:


books['Year-Of-Publication'].value_counts()


# In[27]:


books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')


# In[28]:


books.info()


# In[29]:


plt.figure(figsize=(15,7))
sns.countplot(y='Book-Author',data=books,order=pd.value_counts(books['Book-Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[30]:


plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=books,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[31]:


books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('str')
a = list(books['Year-Of-Publication'].unique())
a = set(a)
a = list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[32]:


books['Book-Author'].fillna('other',inplace=True)


# In[33]:


books.loc[books.Publisher.isnull(),:]


# In[34]:


books.Publisher.fillna('other',inplace=True)


# In[35]:


books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# In[36]:


books.isna().sum()


# In[37]:


books.isna().sum()


# # USERS

# In[38]:


users = pd.read_csv("D:\\DS new project\\Users.csv",encoding='latin-1')


# In[39]:


users


# In[40]:


# shape of the data

users.shape


# In[41]:


users.info()


# In[42]:


users.describe()


# In[43]:


# Checking duplicated values

users.loc[users.duplicated()]


# In[44]:


# Checking Missing values

cols = users.columns 
colours = ['blue', 'yellow'] # yellow is missing. blue is not missing.
sns.heatmap(users[cols].isnull(), cmap=sns.color_palette(colours))


# In[45]:


# Checking Null values

users.loc[users['Age'].isnull()]


# In[46]:


mean = users['Age'].mean()
print(mean)


# In[47]:


# mean impuation on null values

users['Age'] = users['Age'].fillna(users['Age'].mean())
users.isnull().sum()


# In[48]:


users.head()


# OUT LAYERS

# In[49]:


# distplot

sns.distplot(users['Age'])


# In[50]:


# Histogram

users['Age'].hist()
plt.show()


# In[51]:


# Boxplot 

plt.boxplot(users['Age'])
plt.xlabel('Boxplot')
plt.ylabel('Age')
plt.show()


# In[52]:


# Locations

print(users.Location.unique())


# In[53]:


# Location data is not suitable to interpret the information

for i in users:
    users['Country'] = users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')  


# In[54]:


users.Country.nunique()


# In[55]:


# Dropping the Location

users.drop('Location',axis=1,inplace=True)


# In[56]:


users.head()


# In[57]:


users.isnull().sum()


# In[58]:


users['Country']=users['Country'].astype('str')


# In[59]:


a = list(users.Country.unique())
a = set(a)
a = list(a)
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[60]:


users['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[61]:


print(users.Country.nunique())


# In[62]:


plt.figure(figsize=(15,7))
sns.countplot(y='Country', data=users, order=pd.value_counts(users['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# In[63]:


users.isna().sum()


# RATINGS

# In[86]:


ratings =pd.read_csv("D:\\DS new project\\Ratings.csv",encoding='latin-1')


# In[87]:


ratings


# In[88]:


ratings['User-ID'].value_counts()


# In[89]:


ratings['User-ID'].unique().shape


# In[90]:


x = ratings['User-ID'].value_counts() > 200
x[x]


# In[91]:


y = x[x].index
y


# In[92]:


ratings = ratings[ratings['User-ID'].isin(y)]


# In[93]:


ratings


# In[94]:


plt.figure(figsize=(10,6), dpi=100)
ratings['Book-Rating'].value_counts().plot(kind='bar')
plt.title('Ratings Frequency',  fontsize = 16, fontweight = 'bold')
plt.show()


# CONSOLIDATING OF DATASET

# In[118]:


ratings_with_books = ratings.merge(books,on = 'ISBN')
ratings_with_books


# In[119]:


num_rating = ratings_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()


# In[120]:


num_rating.head()


# In[121]:


num_rating.rename(columns={'Book-Rating':'num_of_rating'},inplace=True)


# In[122]:


num_rating.head()


# In[123]:


final_ratings=ratings_with_books.merge(num_rating,on = 'Book-Title')


# In[124]:


final_ratings.head()


# In[125]:


final_ratings.shape


# In[126]:


final_ratings.drop_duplicates(['User-ID','Book-Title'],inplace=True)


# In[127]:


final_ratings


# In[128]:


final_ratings = final_ratings.rename({'User-ID' : 'userid','Book-Title' : 'booktitle','Book-Rating' : 'bookrating'},axis=1)


# In[129]:


final_ratings.drop_duplicates(['userid','booktitle'],inplace=True)


# In[130]:


final_ratings


# In[ ]:





# # MODEL BUILDING

# #Collaborative Filtering
# 

# In[131]:


# Now let us create the pivot table

pivot_table = final_ratings.pivot_table(index='userid',
                                   columns='booktitle',
                                   values='bookrating')


# In[132]:


pivot_table 


# In[133]:


# Filling Null values

pivot_table.fillna(0, inplace=True)


# In[134]:


pivot_table


# In[135]:


# Calculating Cosine Similarity between Users

user_sim = 1 - pairwise_distances(pivot_table.values,metric='cosine')
user_sim


# In[136]:


#Store the results in a dataframe

user_sim_df = pd.DataFrame(user_sim)
user_sim_df


# In[137]:


user_sim_df.index = final_ratings.userid.unique()
user_sim_df.columns = final_ratings.userid.unique()


# In[138]:


user_sim_df


# In[139]:


user_sim_df.iloc[0:15, 0:15]


# In[140]:


# Filling Diagonal values to prevent self similarity

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:15, 0:15]


# In[141]:


# Most Similar Users

user_sim_df.idxmax(axis=1)[0:15]


# In[142]:


print(user_sim_df.max(axis=1).sort_values(ascending=False).head(10))


# In[143]:


user_sim_df.sort_values((44728),ascending=False).iloc[0:5,0:15]


# In[144]:


final_ratings[(final_ratings['userid']==13552)|(final_ratings['userid']==183995)].head(10)


# In[145]:


user1 = final_ratings[(final_ratings['userid']==13552)]
user1


# In[146]:


user2 = final_ratings[(final_ratings['userid']==183995)]
user2


# In[147]:


pd.merge(user1,user2,on='booktitle',how='outer')


# In[148]:


book_read_by_user1 = list(set(user1['booktitle']))
book_read_by_user2 = list(set(user2['booktitle']))

for book_name in book_read_by_user1:
    if book_name not in book_read_by_user2:
        print("Recommendation : ", book_name)


# In[149]:


book_read_by_user1 = list(set(user1['booktitle']))
book_read_by_user2 = list(set(user2['booktitle']))

for book_name in book_read_by_user2:
     if book_name not in book_read_by_user1:
        print("Recommendation : ", book_name)  


# In[150]:


top_n = 5
most_similar_users_ids = {}

for user_id_val in user_sim_df.columns:
    
    # Sort the user IDs by similarity score in descending order
    similar_ids = user_sim_df[user_id_val].sort_values(ascending=False).index.tolist()
    
    # Remove the user's own ID from the list
    similar_ids.remove(user_id_val)
    
    # Store the top N  similar user IDs in the dictionary
    most_similar_users_ids[user_id_val] = similar_ids[:top_n]
    
most_similar_users_ids


# In[151]:


def get_top_n_similar_users(userid, topn=5):
    
    # Sort the user IDs by similarity score in descending order
    similar_ids = user_sim_df[userid].sort_values(ascending=False).index.tolist()
    
    # Remove the user's own ID from the list
    similar_ids.remove(userid)
    
    # Return the top N similar user IDs
    return similar_ids[:topn]

# Example
userid = 26535  
topn = 5  

similar_users = get_top_n_similar_users(userid, topn)

print("Top", topn, "similar users for user", userid, ":", similar_users)


# In[152]:


def get_top_rated_books_for_user(userid, topn=5):
    
    # Filter the final_ratings DataFrame for the given user
    user_ratings = final_ratings[final_ratings['userid'] == userid]
    
    # Sort the user's ratings by book rating in descending order
    user_top_rated_books = user_ratings.sort_values(by='bookrating', ascending=False).head()
    
    return user_top_rated_books

# Example
userid = 43806  
topn = 5  

users_top_rated_books = get_top_rated_books_for_user(userid, topn)

print("Users Top", topn, "rated books for user", userid, ":")
print(users_top_rated_books)


# In[154]:


def recommend_books_to_user(userid, topn=5):
    
    # Get the most similar users
    similar_users = get_top_n_similar_users(userid, topn)
    
    recommended_books = []
    
    for sim_user in similar_users:
        
        # Filter books rated by the similar user
        sim_user_ratings = final_ratings[final_ratings['userid'] == sim_user]
        
        # Find the top-rated books by the similar user
        top_rated_books = sim_user_ratings.sort_values(by='bookrating', ascending=False).head(topn)
        
        # Get the titles of the top-rated books
        new_recommendations = top_rated_books['booktitle'].tolist()
        
        # Append all new recommendations to the list
        recommended_books.extend(new_recommendations)
    
    # Remove duplicates and limit to the specified number of recommendations
    recommended_books = list(set(recommended_books))[:topn]
    
    return recommended_books

# Example
userid = 12538 
top_n = 3  

recommended_books = recommend_books_to_user(userid, topn=top_n)

print("Book recommendations for user", userid, ":", recommended_books)


# In[155]:


recommend_books_to_user(3363,7)


# # MODEL EVALUATION

# In[ ]:


#First, let us evaluate the model using Precision, Recall & F1-Scores


# In[160]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

train_data, test_data = train_test_split(final_ratings, test_size=0.2, random_state=42)

# Rebuilding the pivot table for training data
train_pivot_table = train_data.pivot_table(index='userid',
                                           columns='booktitle',
                                           values='bookrating')
train_pivot_table.fillna(0, inplace=True)

# Function to get recommendations for a user based on the trained model
def recommend_books_to_user_eval(userid, topn=5):
    similar_users = get_top_n_similar_users(userid, topn)
    
    recommended_books = []
    
    for sim_user in similar_users:
        sim_user_ratings = train_data[train_data['userid'] == sim_user]
        top_rated_books = sim_user_ratings.sort_values(by='bookrating', ascending=False).head(topn)
        new_recommendations = top_rated_books['booktitle'].tolist()
        recommended_books.extend(new_recommendations)
    recommended_books = list(set(recommended_books))[:topn]
    
    return recommended_books

# Evaluate the recommendation system on the test data
precision_scores = []
recall_scores = []
f1_scores = []

for userid in test_data['userid'].unique():
    
    # Get actual books rated by the user in the test set
    actual_books = test_data[test_data['userid'] == userid]['booktitle'].tolist()
    
    # Get recommended books using the recommendation function
    recommended_books = recommend_books_to_user_eval(userid, topn=5)
    
    # Check if both actual and recommended books lists have the same length
    if len(actual_books) == len(recommended_books):
        
        # Calculate precision, recall, and F1-score
        precision = precision_score(actual_books, recommended_books, average='micro')
        recall = recall_score(actual_books, recommended_books, average='micro')
        f1 = f1_score(actual_books, recommended_books, average='micro')
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

# Calculate the average scores
average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1-score:", average_f1)


# #The scores that we got are very less. It suggests that Recommendation model is not performing well for Test data.
# 
# #Let us try to improve the model using matrix factorization with Singular Value Decomposition (SVD). It is the part of Surprise library which is popular for Recommendation Systems.

# In[161]:


from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(final_ratings[['userid', 'booktitle', 'bookrating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Singular Value Decomposition (SVD) algorithm
model = SVD(n_factors=50, random_state=42) 

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Function to get top N recommendations for a user
def get_top_n_recommendations(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    # Sort the predictions for each user and get top N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get top N recommendations for a specific user
userid = 12538
user_top_n = get_top_n_recommendations(predictions, n=5).get(userid, [])

print("Top 5 recommendations for user", userid, ":")
for book, predicted_rating in user_top_n:
    print("Book:", book, "| Predicted Rating:", predicted_rating)


# In[ ]:




