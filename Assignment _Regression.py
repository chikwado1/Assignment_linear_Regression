#!/usr/bin/env python
# coding: utf-8

# In[137]:


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[4]:


pip install pandas statsmodels


# In[50]:


pip install pandas


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics


# In[ ]:





# In[ ]:





# In[138]:


# Read data from a CSV file (replace with the correct file path)
df = pd.read_csv(r'C:\Users\chikw\Downloads\social_media_data.csv')

# Display the first few rows of the DataFrame to understand its structure
print(df.head())


# In[ ]:





# In[139]:


# Display a concise summary of the DataFrame
social_media_data.info()


# In[140]:


# View the last 5 rows also to understand the dataset

df.tail()


# In[141]:


# If you want to see 20 rows you add 20 into the bracket

df.head(100)


# In[ ]:





# In[142]:


# To see all the columns in the dataset 

df.columns


# In[ ]:





# In[143]:


# to view some statistic information about the numeric data in the dataset

df.describe()


# In[ ]:





# In[144]:


# check the data type of all the columns

df.dtypes


# In[145]:


#We can also check to see if there are still missing values

df.isna().sum().sum()


# In[146]:


# List all column names to identify the target variable
social_media_data.columns


# In[66]:


import pandas as pd

# Specify the path to your CSV file
file_path = r'C:\Users\chikw\Downloads\social_media_data.csv'  # Use raw string

# Load the CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the data to confirm it's loaded correctly
print(data.head())


# In[ ]:





# In[148]:


# Define the feature and the target variable
X = social_media_data[['User Engagement History']]
y = social_media_data['Likes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict the target variable using the test data
y_pred = linear_model.predict(X_test)

# Calculate and display the coefficients
intercept = linear_model.intercept_
coefficient = linear_model.coef_[0]

print(f"Intercept: {intercept}")
print(f"Coefficient: {coefficient}")

# Plot the scatter plot of the test data
plt.scatter(X_test, y_test, color='r', alpha=0.5, label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='b', label='Regression Line')

# Add title and labels
plt.title("Simple Linear Regression: User Engagement History vs. Likes")
plt.xlabel("User Engagement History")
plt.ylabel("Likes")

# Add a legend
plt.legend()

# Display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[149]:


# Define the features and the target variable
X = social_media_data[['User Engagement History', 'Comments', 'Shares', 'Post Frequency']]
y = social_media_data['Likes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict the target variable using the test data
y_pred = linear_model.predict(X_test)

# Calculate and display the coefficients
intercept = linear_model.intercept_
coefficients = linear_model.coef_

print(f"Intercept: {intercept}")
print(f"Coefficients: {coefficients}")

# Plot the scatter plot of the test data for the first variable 'User Engagement History'
plt.scatter(X_test['User Engagement History'], y_test, color='r', alpha=0.5, label='Actual Data')

# Plot the regression line for the first variable 'User Engagement History'
plt.plot(X_test['User Engagement History'], y_pred, color='b', label='Regression Line')

# Add title and labels
plt.title("Multiple Linear Regression: Predicting Likes")
plt.xlabel("User Engagement History")
plt.ylabel("Likes")

# Add a legend
plt.legend()

# Display the plot
plt.show()


# In[ ]:





# In[150]:


# Define the features and the target variable
X = social_media_data[['User Engagement History', 'Comments', 'Shares', 'Post Frequency']]
y = social_media_data['Likes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict the target variable using the test data
y_pred = linear_model.predict(X_test)

# Calculate and display the coefficients
intercept = linear_model.intercept_
coefficients = linear_model.coef_

print(f"Intercept: {intercept}")
print(f"Coefficients: {coefficients}")

# Plot the scatter plot of the test data for the different variables
plt.scatter(X_test['User Engagement History'], y_test, color='r', alpha=0.5, label='User Engagement History')
plt.scatter(X_test['Comments'], y_test, color='g', alpha=0.5, label='Comments')
plt.scatter(X_test['Shares'], y_test, color='b', alpha=0.5, label='Shares')
plt.scatter(X_test['Post Frequency'], y_test, color='y', alpha=0.5, label='Post Frequency')

# Plot the regression line for the first variable 'User Engagement History'
plt.plot(X_test['User Engagement History'], y_pred, color='r', linestyle='--', label='Regression Line (User Engagement History)')

# Add title and labels
plt.title("Multiple Linear Regression: Predicting Likes")
plt.xlabel("Independent Variables")
plt.ylabel("Likes")

# Add a legend
plt.legend()

# Display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[151]:


# Plot the scatter plot of the test data for the different variables
plt.figure(figsize=(14, 8))

colors_actual = ['r', 'g', 'b', 'y']
colors_predicted = ['darkred', 'darkgreen', 'darkblue', 'gold']
shapes_actual = ['o', 's', 'D', '^']
shapes_predicted = ['v', '<', '>', 'p']
variables = ['User Engagement History', 'Comments', 'Shares', 'Post Frequency']

for i, variable in enumerate(variables):
    plt.scatter(X_test[variable], y_test, color=colors_actual[i], alpha=0.5, label=f'{variable} - Actual', marker=shapes_actual[i])
    plt.scatter(X_test[variable], y_pred, color=colors_predicted[i], alpha=0.5, edgecolor='k', label=f'{variable} - Predicted', marker=shapes_predicted[i])
    sorted_indices = X_test[variable].argsort()
    plt.plot(X_test[variable].iloc[sorted_indices], y_pred[sorted_indices], color=colors_predicted[i], linestyle='--', label=f'{variable} - Regression Line')

# Add title and labels
plt.title("Multiple Linear Regression: Predicting Likes")
plt.xlabel("Independent Variables")
plt.ylabel("Likes")

# Add a legend
plt.legend(loc='upper left', bbox_to_anchor=(1,1))

# Display the plot
plt.tight_layout()
plt.show


# In[ ]:





# In[ ]:




