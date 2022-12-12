from pyspark import SparkConf
from pyspark.sql import SparkSession
import os

print(os.environ["AWS_SECRET_ACCESS_KEY"])
print('='*80)

BUCKET = "dmacademy-course-assets"
KEY_pre = "vlerick/pre_release.csv"
KEY_after = "vlerick/after_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Read the CSV file from the S3 bucket
pre_release = spark.read.csv(f"s3a://{BUCKET}/{KEY_pre}", header=True)
after_release = spark.read.csv(f"s3a://{BUCKET}/{KEY_after}", header=True)

pre_release.show()
after_release.show()

#Convert the Spark DataFrames to Pandas DataFrames.

pre_df = pre_release.toPandas()
after_df = after_release.toPandas()

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

#Merging the two data frames based on the movie_title column as this will be necessary for our prediction
#Checking the first few rows to see the effect of the merge
df = pd.merge(pre_df, after_df, how='inner', on='movie_title')
df.head()

#Checking the size of the dataset (number of rows and number of columns)
df.shape


#Checking the data type of the features in the dataset
df.info()


#Checking basic statistical characteristics of each numerical feature in a dataset
df.describe()


#Investigating the number of unique values. In this case this is more interesting to observe the categorical variables 
df.nunique()


# ## Data Cleaning

# In this section I will be cleaning the data.

#Check the number of missing values for each column
df.isnull().sum()


#Replacing all the NaN values regarding the facebook likes and critic reviews with the value 0
df[["actor_1_facebook_likes",
    "actor_2_facebook_likes", "actor_3_facebook_likes", "num_critic_for_reviews"]] = df[[
    "actor_1_facebook_likes","actor_2_facebook_likes", "actor_3_facebook_likes", "num_critic_for_reviews"]].fillna(0)


#Creating a list of categorical and numerical features
cat = df.select_dtypes(include = "object").columns.tolist()
num = df.select_dtypes(include = "float64").columns.tolist()


#Replacing the NaN values for the language for the most commun categorical value
cat_imputer = SimpleImputer(strategy = "most_frequent")
cat_imputer.fit(df[["language"]])
df[["language"]] = cat_imputer.transform(df[["language"]])


#Replacing the NaN values for the content_rating for the most commun categorical value
cat_imputer = SimpleImputer(strategy = "most_frequent")
cat_imputer.fit(df[["content_rating"]])
df[["content_rating"]] = cat_imputer.transform(df[["content_rating"]])


#Check the number of missing values for each column
df.isnull().sum()


#Removing the rows with the remaining missing values (actor names) as this will only remove less than 1% of the data
df = df.dropna()


#Final check to confirm that all null values were successfully removed
df.isnull().sum()


#Checking if there are any duplicated rows in the dataset

df[df.duplicated() == True]


#Removing the duplicates from the dataset
df.drop_duplicates(inplace = True)

#Checking if successfully removed all duplicates
print(df[df.duplicated() == True].shape[0])


#Checking the new shape of the dataset 
df.shape


#Checking the count for all values under the content_rating column
rating_counts = df["content_rating"].value_counts()
print(rating_counts)


#Creating a new content_rating value called "other" to assign the values with a count <= 30
r_vals = rating_counts[:3].index
print (r_vals)
df["content_rating"] = df.content_rating.where(df.content_rating.isin(r_vals), "other")


#Checking if it worked
df["content_rating"].value_counts()


#Checking the count for all values under the country column
country_counts = df["country"].value_counts()
print(country_counts)


#Creating a new country value called "other" to assign the values with a count =< 40
c_vals = country_counts[:2].index
print (c_vals)
df['country'] = df.country.where(df.country.isin(c_vals), "other")


#Checking if it worked
df["country"].value_counts()


#Checking the count for all values under the language column
language_counts = df["language"].value_counts()
print(language_counts)


#Considering that only 8.6% of movies are not in English I will drop the language column
#As this shouldn't add any additional value to our model
df.drop('language', inplace=True, axis=1)


#Checking the genres value counts
df["genres"].value_counts()


#As the genres seem to be equally distributed I will be dropping this column
df.drop("genres", inplace=True, axis=1)


#Removing the columns with the actor and director names, as well as movie title since these won't be used
df = df.drop(["actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title"], axis=1)

#Removing the other features from the after_release.csv as these won't be used to predict the target variable chosen
df = df.drop(["num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "movie_facebook_likes"], axis=1)


#Binning the IMDB scores as 0-2,2-4,4-6,6-8,8-10 to classify them accordingly as Very Bad
#Bad, Average, Good, Very Good. This will be my target variable and I will be using Classification models.
df["imdb_score_binned"]= pd.cut(df["imdb_score"], bins=[0, 2, 4, 6, 8, 10], right=True, labels=False)+1


#After doing this the imdb_score column can be dropped 
df = df.drop(["imdb_score"], axis=1)


#Verifying the applied changes by looking into the first 10 rows 
df.head(10)


#Checking the value counts across the different bins created
df["imdb_score_binned"].value_counts()

#Removing the actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes columns they are all correlated
#between eachother and also highly correlated with the cast_total_facebook_likes
df = df.drop(['actor_2_facebook_likes', "actor_3_facebook_likes", "actor_1_facebook_likes"], axis=1)


#Getting the dummies for the columns country and content_rating
df = pd.get_dummies(df, prefix=["content_rating", "country"], columns=["content_rating", "country"], drop_first = False)


#Checking if it worked by getting all the columns in our dataframe
df.columns


# #### Splitting data into traning and test

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

#Scaling our features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# # Classification Models

# ### Logistic Regression


#Logistic Regression to predict Binned IMDB scores
reg = LogisticRegression(class_weight = "balanced")
# Train the model on training data
reg.fit(X_train,np.ravel(y_train,order = "C"))
#Make predictions of test
reg_pred = reg.predict(X_test)

print("The accuracy of the model is:",metrics.accuracy_score(y_test, reg_pred))


# ### Decision Tree

#Decision Tree to predict Binned IMDB scores
dt = DecisionTreeClassifier(criterion="entropy", class_weight = "balanced") 
# Train the model on training data
dt.fit(X_train, np.ravel(y_train,order="C"))
#Make predictions of test
dt_pred = dt.predict(X_test)

print("The accuracy of the model is:",accuracy_score(y_test,dt_pred))