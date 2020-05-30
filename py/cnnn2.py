
import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

df = pd.read_csv('CSV (8).csv')
#This will convert the date which otherwise would hve been a string to a date time object.  We will now get a date-time ind

import datetime

li= []
for i in range(0, 100):
    l =df['Date'][i].split('-')
    d = ' '
    d = d.join(l)
    d = datetime.datetime.strptime(d, '%d %m %Y').strftime('%m-%d-%Y')
    date_object = datetime.datetime.strptime(d, '%m-%d-%Y').date()
    date_object
    li.append(date_object)

df['Date']=li
df.set_index('Date')
df['new']= ((df['High']+df['Low'])/2)
df.drop('High',axis=1,inplace=True)
df.drop('Low',axis=1,inplace=True)
df=df.set_index('Date')

df.plot()

# a variable for predicting 'n' days out in the future
forecast_out = 15
# Create one more column shifted 'n' units up
df['Prediction'] = df[['new']].shift(-forecast_out)

#Create the independent dataset(X)
#Convert the dataframe to NUmpy array
X = np.array(df.drop(['Prediction'], 1))
#REmove the last 'n' rows
X = X[:-forecast_out]

#Create the dependent dataset(y)
# Convert dataframe to numpy array (All of the values including NaN)
y = np.array(df['Prediction'])

#Get all of the y values except the last n rows
y = y[:-forecast_out]
print(y)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Create and train the Polynomial REgression Model
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
#plt.scatter(X, y, color = 'red')
#plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
#plt.title('Truth or Bluff (Polynomial Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

#Testing Model
pr_confidence = lin_reg_2.score(poly_reg.fit_transform(x_test), y_test)
print(pr_confidence)

predictionss = lin_reg_2.predict(poly_reg.fit_transform(x_test))

plt.plot(y_test)
plt.plot(predictionss)
plt.show()

# set x_forecast equal to the last 10 rows of the original dataset
x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
print(x_forecast)

lr_prediction = lin_reg_2.predict(poly_reg.fit_transform(x_forecast))
print(lr_prediction)

plt.plot(lr_prediction)

forecast_index = pd.date_range(start='2020-04-25',periods=15)

forecast_df = pd.DataFrame(data=lr_prediction,index=forecast_index,
                           columns=['Forecast'])

df.drop('Prediction',axis=1,inplace=True)

df.plot()
forecast_df.plot()

ax = df.plot()
forecast_df.plot(ax=ax)
#######

import pickle
with open('trained-model.pkl','wb') as f:
    pickle.dump(lin_reg_2 ,f)

with open('trained-model.pkl','rb') as f:
    mp = pickle.load(f)

message = "enter msg"
data = [message]
vect = cv.transform(data).toarray()
my_prediction = lin_reg_2 .predict(vect)
print(my_prediction)
