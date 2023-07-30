import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# loading the csv data to a pandas dataframe
gold_data = pd.read_csv('gold_price_data.csv')

gold_data.head()

gold_data.tail()

# number of rows and columns
gold_data.shape

# getting the information about the data
gold_data.info()

gold_data.isnull().sum()

# getting the statistical information about the data
gold_data.describe()

# finding the correlation between the data columns in the dataset like positive and negative correlation
correlation = gold_data

print(correlation['GLD'])

# checking the distribution of GLD price
sns.histplot(gold_data['GLD'], color='green')

# splitting the features and the target
X = gold_data.drop(['Date', 'GLD'],axis=1)
Y = gold_data['GLD']

print(Y)

# splitting into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# training the model using random forest 
regressor = RandomForestRegressor(n_estimators=100)


regressor.fit(X_train, Y_train)


# prediction on test data
test_data_prediction = regressor.predict(X_test)




print(test_data_prediction)



# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)



# compare the actual values and predicted values
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green',label="predicted value")
plt.title("Actual Price vs Predicted Price")
plt.xlabel('Number of values')
plt.ylabel("GLD Price")
plt.legend()
plt.show()
