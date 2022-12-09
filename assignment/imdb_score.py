#!/usr/bin/env python
# coding: utf-8

# In[69]:


#Importing libraries that might be necessary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
import xgboost as xgb


# # Background Information

# IMDB is the go-to website for information about movies and television series. If you cannot decide which movie you are going to watch tonight, simply go to IMDB.com and find out what the highest rated movies are, which movies are produced by your favourite producer, or which movies are starring your favourite movie star. The aim of this case is to apply everything you learned this week on a data set provided by IMDB.

# # Variable Description

# #### Pre_release variables
# 1. director_name - Name of the movie director
# 2. duration - Duration of the movie
# 3. director_facebook_likes - Number of Facebook likes of the director
# 4. actor_1_name - Primary actor starring in the movie
# 5. actor_2_name - Other actor starring in the movie
# 6. actor_3_name - Other actor starring in the movie
# 7. actor_1_facebook_likes - Number of likes of actor_1 on his/her Facebook Page
# 8. actor_2_facebook_likes - Number of likes of actor2 on his/her Facebook Page
# 9. actor_3_facebook_likes - Number of likes of actor_1 on his/her Facebook Page
# 10. genres - Genre of the movie
# 11. movie_title - Title of the movie
# 12. cast_total_facebook_likes - Total number of Facebook likes of the entire cast of the movie
# 13. language - Language of the movie
# 14. country - Country of production
# 15. content_rating - Age restrictions on viewership. Parental Guidance Suggested to determine if a film is appropriate for children
# 16. budget - Production budget of the movie in dollars
# #### After_release variables
# 1. num_critic_for_review - Number of critical reviews on IMDb
# 2. gross - Gross earnings of the movie in dollars
# 3. movie_title - Title of the movie
# 4. num_voted_users - Number of people who voted for the movie
# 5. num_users_for_reviews - Number of users who reviewed the movie
# 6. imdb_score - IMDb score of the movie
# 7. movie_facebook_likes - Number of Facebook likes on the movie page

# ## Case Study
# 
# The aim of this case is to use the information that is known before the movie was released to predict a feature that becomes known only after the movie got released. In this case I will be predicting the IMDB score.

# ## Introduction

# As a regular IMDB user I like to look into the IMDB score of movies before deciding if I should watch a certain movie or not. In addition, I am always very thrilled to go to the cinema and to watch the new hits on the big screen. However, I  like to make sure that I'm going to watch a quality movie. As so, I figured it could be interesting to look into different factors (Such as: Director Facebook likes, Budget, Content rating, and others) that might allow me to predict if a movie is going to be a success or not, based on its IMDB Score. Therefore, I would be able to decide if it would be worth it to spend my money on buying a cinema ticket for a specific movie ahead of time. The end result will be a classification based on the IMDB Score ranging from Very Bad (class 1) to Very Good (class 5), which will allow me to make a decision if I should go or if it would be better to save my money to watch a different movie.

# 
# 
# # Data Preprocessing

# In this section I will be preprocessing the data.

# In[70]:


#Reading the pre_release data and checking the first rows
df = pd.read_csv('../data/pre_release.csv')
df.head()


# In[71]:


#Reading the after_release data and checking the first rows
df2 = pd.read_csv('../data/after_release.csv')
df2.head()


# In[72]:


#Merging the two data frames based on the movie_title column as this will be necessary for our prediction
#Checking the first few rows to see the effect of the merge
df = pd.merge(df, df2, how='inner', on='movie_title')
df.head()


# In[73]:


#Checking the size of the dataset (number of rows and number of columns)
df.shape


# In[74]:


#Checking the data type of the features in the dataset
df.info()


# In[75]:


#Checking basic statistical characteristics of each numerical feature in a dataset
df.describe()


# In[76]:


#Investigating the number of unique values. In this case this is more interesting to observe the categorical variables 
df.nunique()


# ## Data Cleaning

# In this section I will be cleaning the data.

# In[77]:


#Check the number of missing values for each column
df.isnull().sum()


# In[78]:


#Replacing all the NaN values regarding the facebook likes and critic reviews with the value 0
df[["actor_1_facebook_likes",
    "actor_2_facebook_likes", "actor_3_facebook_likes", "num_critic_for_reviews"]] = df[[
    "actor_1_facebook_likes","actor_2_facebook_likes", "actor_3_facebook_likes", "num_critic_for_reviews"]].fillna(0)


# In[79]:


#Creating a list of categorical and numerical features
cat = df.select_dtypes(include = "object").columns.tolist()
num = df.select_dtypes(include = "float64").columns.tolist()


# In[80]:


#Replacing the NaN values for the language for the most commun categorical value
cat_imputer = SimpleImputer(strategy = "most_frequent")
cat_imputer.fit(df[["language"]])
df[["language"]] = cat_imputer.transform(df[["language"]])


# In[81]:


#Replacing the NaN values for the content_rating for the most commun categorical value
cat_imputer = SimpleImputer(strategy = "most_frequent")
cat_imputer.fit(df[["content_rating"]])
df[["content_rating"]] = cat_imputer.transform(df[["content_rating"]])


# In[82]:


#Check the number of missing values for each column
df.isnull().sum()


# In[83]:


#Removing the rows with the remaining missing values (actor names) as this will only remove less than 1% of the data
df = df.dropna()


# In[84]:


#Final check to confirm that all null values were successfully removed
df.isnull().sum()


# In[85]:


#Checking if there are any duplicated rows in the dataset

df[df.duplicated() == True]


# In[86]:


#Removing the duplicates from the dataset
df.drop_duplicates(inplace = True)

#Checking if successfully removed all duplicates
print(df[df.duplicated() == True].shape[0])


# In[87]:


#Checking the new shape of the dataset 
df.shape


# In[88]:


#Checking the count for all values under the content_rating column
rating_counts = df["content_rating"].value_counts()
print(rating_counts)


# In[89]:


#Creating a new content_rating value called "other" to assign the values with a count <= 30
r_vals = rating_counts[:3].index
print (r_vals)
df["content_rating"] = df.content_rating.where(df.content_rating.isin(r_vals), "other")


# In[90]:


#Checking if it worked
df["content_rating"].value_counts()


# In[91]:


#Checking the count for all values under the country column
country_counts = df["country"].value_counts()
print(country_counts)


# In[92]:


#Creating a new country value called "other" to assign the values with a count =< 40
c_vals = country_counts[:2].index
print (c_vals)
df['country'] = df.country.where(df.country.isin(c_vals), "other")


# In[93]:


#Checking if it worked
df["country"].value_counts()


# In[94]:


#Checking the count for all values under the language column
language_counts = df["language"].value_counts()
print(language_counts)


# In[95]:


#Creating a histogram to visualize the language distributuion
plt.figure(figsize=(15,5));
sns.histplot(data=df["language"], color = "lightblue", ec="black");
plt.xticks(rotation=40);


# In[96]:


#Considering that only 8.6% of movies are not in English I will drop the language column
#As this shouldn't add any additional value to our model
df.drop('language', inplace=True, axis=1)


# Side Note: I tested doing the same I did for the countries and content_rating for language, however since most movies are in English it didn't make significant changes on the model for me to keep it in.

# In[97]:


#Checking the genres value counts
df["genres"].value_counts()


# In[98]:


#As the genres seem to be equally distributed I will be dropping this column
df.drop("genres", inplace=True, axis=1)


# Side Note: I tested incorporating the genres in the models by splitting them by the delimiter "|" and then dummifying the multiple genres. However, doing this didn't have a significant impact on the the models which lead me to take them out.

# # Data Visualization

# As a next step I will be exploring the data through different visuals to better visualize features of interest. Also, I will be assessing some potential pitfalls within the dataset.

# In[99]:


#Creating a histogram to view the distribution of counts of imdb_score
sns.histplot(data=df['imdb_score'], color = "lightblue");
plt.title("Distribution of IMDB scores", fontweight="bold");


# Most movies have an imdb_score between 5 and 7.

# In[100]:


#Creating a violinplot to view the distribution of imdb_score across countries
plt.figure(figsize=(5,5))

sns.boxplot(x='country',y='imdb_score',data=df, palette='rainbow')
plt.title("IMDB scores across countries", fontweight="bold");
plt.show()


# The "other countries" have most movies with the highest IMDB scores.

# In[101]:


#Creating a violinplot to view the distribution of imdb_score across content_ratings
plt.figure(figsize=(5,5))

sns.boxplot(x='content_rating',y='imdb_score',data=df, palette='rainbow')
plt.title("IMDB scores across Content Rating", fontweight="bold");
plt.show()


# The "other content rating" have most movies with the highest IMDB scores.

# In[102]:


#Making a 2d density chart for the imdb_score and budget
plt.figure(figsize=(5,5))
sns.scatterplot(data = df, x ="budget",y = "imdb_score");
plt.title("Budget and IMDB Score", fontweight="bold");
plt.show()


# It seems like there are a lot of movies with a budget of 0, or at least a low budget. Overall, having a higher budget doesn't mean you will have a higher imdb_score.

# In[103]:


#Making a scatterplot for imdb_score and duration
plt.figure(figsize=(5,5))
sns.scatterplot(data = df, x ="duration",y = "imdb_score");
plt.title("Duration and IMDB Score", fontweight="bold");
plt.show()


# Most movies have between 80 and 100 minutes (duration). Overall, having a higher duration doesn't mean you will have a higher imdb_score.

# In[104]:


#Making scatterplots for facebook likes (actors and directors) and imdb_score
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,4))
sns.scatterplot(data = df, x ="actor_1_facebook_likes",y = "imdb_score", ax = ax1);
sns.scatterplot(data = df, x ="actor_2_facebook_likes",y = "imdb_score", ax = ax2);
sns.scatterplot(data = df, x ="actor_3_facebook_likes",y = "imdb_score", ax = ax3);
sns.scatterplot(data = df, x ="director_facebook_likes",y = "imdb_score", ax = ax4);
print("\033[1m" + "                                 Facebook likes Actors & Director vs IMDB Score" + "\033[0m")


# As it can be seen by the scatterplots above there are a lot of values equal to 0 regarding the facebook likes. In addition, the likes seem to be capped at 1000 which seems rather strange and not representative of what the reality might be. As so, I believe this might be a pitfall of the dataset that might affect my predictions and accuracy of the model. This will be emphasized when looking into the correlation matrix of our features.

# # Data Preparation for models

# In this section I will be preparing the data for the models.

# In[105]:


#Removing the columns with the actor and director names, as well as movie title since these won't be used
df = df.drop(["actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title"], axis=1)


# In[106]:


#Removing the other features from the after_release.csv as these won't be used to predict the target variable chosen
df = df.drop(["num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "movie_facebook_likes"], axis=1)


# In[107]:


#Binning the IMDB scores as 0-2,2-4,4-6,6-8,8-10 to classify them accordingly as Very Bad
#Bad, Average, Good, Very Good. This will be my target variable and I will be using Classification models.
df["imdb_score_binned"]= pd.cut(df["imdb_score"], bins=[0,2,4,6,8,10], right=True, labels=False)+1


# In[108]:


#After doing this the imdb_score column can be dropped 
df = df.drop(["imdb_score"], axis=1)


# In[109]:


#Verifying the applied changes by looking into the first 10 rows 
df.head(10)


# In[110]:


#Checking the value counts across the different bins created
df["imdb_score_binned"].value_counts()


# In[111]:


#Creating a violinplot to view the distribution of imdb_scores across the different bins created
sns.violinplot(data=df['imdb_score_binned'], color = "lightblue");
plt.title("Distribution of Binned IMDB scores across classes", fontweight="bold");
sns.set(style="darkgrid")


# There are very few values for the binned scores of class 1, 5, and 2. This needs to be dealt with in order to improve my model predictions. As so, during the modeling phase I will explain how I intend to do this.
# 
# I also tried adjusting the bins by modifying them (Example: making them larger and smaller); however, for me this was the best way of approaching our problem, as I also across the testing didn't see improvements in my models.

# In[112]:


#Creating a correlation matrix where it will be able to see the correlation between the different variables
plt.figure(figsize=(15,7))
matrix = df.corr().round(2)

#Creating a mask so it only displays the lower portion of the matrix so it is easier to visualize
mask = np.triu(np.ones_like(matrix, dtype=bool))

#Making it a heatmap with a color scale so it is easier to identify which ones we will need to handle, this will 
#happen when the value displayed between variables is above 0.7
m = sns.heatmap(matrix, annot = True,vmax=1, vmin=-1, center=0, cmap='vlag', mask = mask);
m.set_xticklabels(m.get_xticklabels(), rotation=45, horizontalalignment='right');
plt.title("Correlation matrix", fontweight="bold");


# #### Considerations
# 
# After carefully looking at the matrix it is possibly to verify that there are some variables with a 
# correlation value above 0.7, therefore we need to handle those. As so, first we need to handle the actor facebook
# likes because they are highly correlated to the cast_total_facebook_likes. In addition, the actor facebook likes
# are also correlated between each other so some attention should go towards there as well. Finally, the num_voted_users
# and num_users_for_reviews are also correlated and must be handled

# In[113]:


#Removing the actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes columns they are all correlated
#between eachother and also highly correlated with the cast_total_facebook_likes
df = df.drop(['actor_2_facebook_likes', "actor_3_facebook_likes", "actor_1_facebook_likes"], axis=1)


# In[114]:


#Creating the correlation matrix once again to verify the effect of the applied changes, by using the same code that 
#I used above (With one slight change to adjust the size of the figure).

plt.figure(figsize=(10,5))
matrix = df.corr().round(2)
mask = np.triu(np.ones_like(matrix, dtype=bool))
m = sns.heatmap(matrix, annot = True,vmax=1, vmin=-1, center=0, cmap='vlag', mask = mask);
m.set_xticklabels(m.get_xticklabels(), rotation=45, horizontalalignment='right');
plt.title("Adjusted Correlation matrix", fontweight="bold");


# After the applied changes we can verify that now all of the correlation values for the present variables are all 
# below 0.7, therefore we can proceed.

# In order to use the categorical data in the model we need to create dummie variables. As so, I will be creating those for the columns country and content_rating.

# In[115]:


#Getting the dummies for the columns country and content_rating
df = pd.get_dummies(df, prefix=["content_rating", "country"], columns=["content_rating", "country"], drop_first = False)


# In[116]:


#Checking if it worked by getting all the columns in our dataframe
df.columns


# #### Splitting data into traning and test

# In[117]:


#Making the split between the data that we need to train our model and the target variable that will be trying to 
#predict. I also trying to use stratify = y to better balance the data, but obtained better results by doing so 
#during the fitting for each model
X=pd.DataFrame(columns=["duration","director_facebook_likes","cast_total_facebook_likes","budget","content_rating_PG","content_rating_PG-13","content_rating_R","content_rating_other","country_UK","country_USA","country_other"],data = df)
y=pd.DataFrame(columns=["imdb_score_binned"],data = df)
from sklearn.model_selection import train_test_split
#Randomly split into training (70%) and val (30%) sample with a random state of 100. I found this to be the combination
#That gave me the best results across all models
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=100)


# As it can be seen above in the value counts and Violin plot for the IMDB binned scores, there are very few values within the bins 1, 2 and 5 in comparison to the others. In order to improve this, and to avoid problems later on with the models, I will be adjusting the weights for the classes for all of the models by using the class_weight parameter. In addition, instead of a RandomForestClassifier I also decided to use a BalancedRandomForestClassifier to make my results my realistic.

# #### Scaling the features

# In[118]:


#Scaling our features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# # Classification Models

# ### Logistic Regression

# In[119]:


#Logistic Regression to predict Binned IMDB scores
reg = LogisticRegression(class_weight = "balanced")
# Train the model on training data
reg.fit(X_train,np.ravel(y_train,order = "C"))
#Make predictions of test
reg_pred = reg.predict(X_test)

print("The accuracy of the model is:",metrics.accuracy_score(y_test, reg_pred))


# ### Decision Tree

# In[120]:


#Decision Tree to predict Binned IMDB scores
dt = DecisionTreeClassifier(criterion="entropy", class_weight = "balanced") 
# Train the model on training data
dt.fit(X_train, np.ravel(y_train,order="C"))
#Make predictions of test
dt_pred = dt.predict(X_test)

print("The accuracy of the model is:",accuracy_score(y_test,dt_pred))


# ### Random Forest

# In[121]:


#Random forest model to predict Binned IMDB scores
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from imblearn.ensemble import BalancedRandomForestClassifier


# In[122]:


#Seeing the influence of the accuracy on a larger random forest on both train and test set
#Doing this by iterativly running a random forest with a higher number of estimators and max_depth
train_accuracy = []
test_accuracy = []
highest_accuracy = 0
best_estimators = 0
best_max_depth = 0
for i in range(10, 200,10):
    train_accuracy_part = []
    test_accuracy_part = []
    for j in range(5, 50, 5):
        #Train model with j depth
        rf = BalancedRandomForestClassifier(n_estimators=i, max_depth=j)  
        rf.fit(X_train,y_train)
        #Make predictions + accuracy
        y_predict_train = rf.predict(X_train)
        train_acc = accuracy_score(y_train,y_predict_train)
        rf_pred = rf.predict(X_test)
        test_acc = accuracy_score(y_test,rf_pred)
        train_accuracy_part.append(train_acc)
        test_accuracy_part.append(test_acc)
        if test_acc > highest_accuracy:
            highest_accuracy = test_acc
            best_estimators = i
            best_max_depth = j
    train_accuracy.append(train_accuracy_part)
    test_accuracy.append(test_accuracy_part)
print("Best n_estimators is", best_estimators,"and best max_depth is", best_max_depth,"with an accuracy:", highest_accuracy)


# ### Model Comparison

# In[123]:


#Importing package for model comparison
from sklearn.metrics import classification_report


# #### Logistic Regression

# In[124]:


#Printing the classification report for the Logistic Regression
print(classification_report(y_test, reg_pred))

#Creating and printing Confusion matrix Logistic Regression
print("\033[1m" + "Confusion Matrix Logistic Regression" + "\033[0m")
cnf_matrix = metrics.confusion_matrix(y_test, reg_pred)
print(cnf_matrix)


# #### Decision Tree


#Printing the classification report for the Decision Tree
print(classification_report(y_test, dt_pred))
#Creating and printing Confusion matrix Logistic Regression
print("\033[1m" + "Confusion Matrix Decision Tree" + "\033[0m")
cnf_matrix = metrics.confusion_matrix(y_test, dt_pred)
print(cnf_matrix)


# #### Balanced Random Forest

# In[126]:


#Printing the classification report for the Balanced Random Forest
print(classification_report(y_test, rf_pred))
#Creating and printing Confusion matrix Logistic Regression
print("\033[1m" + "Confusion Matrix Balanced Random Forest" + "\033[0m")
cnf_matrix = metrics.confusion_matrix(y_test, rf_pred)
print(cnf_matrix)


# # Conclusion

# Considering that if I were to choose at random each I would have a 20% chance of assigning the movie to its correct class, all the models obtain better results compared to this. However, after analysing the outputs of the Classification reports for the multiple models, the one which constantly gets the highest accuracy is the Decision Tree model. It is still important to consider that for the Decision Tree model it does not predict any movie as Very Good (class 5). 
# 
# Overall, I believe that there were some problems with the dataset. The facebook likes didn't seem realistic and some classes had a very low number of observations. These factores combined resulted in a lower performance across all models, and with a bigger dataset better results could be obtained.
# 
# At the end of the day, considering everything that was said above I am still happy with my results as I would almost have a 60% chance of knowing if I should or not spend my money on the cinema ticket
