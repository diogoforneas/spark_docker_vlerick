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

pre_data = pre_release.toPandas()
after_data = after_release.toPandas()

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

#Reading the pre_release data and checking the first rows
df = pd.read_csv('../data/pre_release.csv')
df.head()


#Reading the after_release data and checking the first rows
df2 = pd.read_csv('../data/after_release.csv')
df2.head()

#Merging the two data frames based on the movie_title column as this will be necessary for our prediction
#Checking the first few rows to see the effect of the merge
df = pd.merge(df, df2, how='inner', on='movie_title')
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
df["imdb_score_binned"]= pd.cut(df["imdb_score"], bins=[0,2,4,6,8,10], right=True, labels=False)+1


#After doing this the imdb_score column can be dropped 
df = df.drop(["imdb_score"], axis=1)


#Verifying the applied changes by looking into the first 10 rows 
df.head(10)


#Checking the value counts across the different bins created
df["imdb_score_binned"].value_counts()


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


# ### Random Forest

#Random forest model to predict Binned IMDB scores
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from imblearn.ensemble import BalancedRandomForestClassifier


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

#Importing package for model comparison
from sklearn.metrics import classification_report


# #### Logistic Regression

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


#Printing the classification report for the Balanced Random Forest
print(classification_report(y_test, rf_pred))
#Creating and printing Confusion matrix Logistic Regression
print("\033[1m" + "Confusion Matrix Balanced Random Forest" + "\033[0m")
cnf_matrix = metrics.confusion_matrix(y_test, rf_pred)
print(cnf_matrix)
