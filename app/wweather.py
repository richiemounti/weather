import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import pickle

''' 
load the dataset
'''
filename = "C:/Users/loi/Videos/weather/app/data/weather.csv"

df = pd.read_csv(filename)

df.describe()

label = df.Play
features_attr = ["Outlook", "Temp", "Humidity", "Windy"]
features = df[features_attr]

'''
problem  statement
consider a function y = f(x), y would be the label
'''
y = label

predictors = ['Outlook', 'Temp', 'Humidity', 'Windy']

X_raw = df[predictors]

'''
check for missing values
'''
my_imputer = Imputer()
X = my_imputer.fit_transform(X_raw)

'''
splitting the data into a training and a test set
'''
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

'''
fuction f in f(x)
'''
decision_tree_regressor = DecisionTreeRegressor()

'''
tell the regressor to act on predictors, x and give us a f(x).This is done by a fit method
'''
decision_tree_model = decision_tree_regressor.fit(train_X, train_y)
predicted_price = decision_tree_regressor.predict(val_X)


''' saving model to disk '''
pickle.dump(decision_tree_model, open('model.pkl', 'wb'))

''' loading model to compare the results '''
model = pickle.load(open('model.pkl', 'rb'))

'''
calculate the percentage accuracy of the calculated model
'''
print("Accuracy is : {0}".format(accuracy_score(val_y, predicted_price)*100))