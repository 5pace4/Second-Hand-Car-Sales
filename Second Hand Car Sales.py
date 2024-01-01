# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loading dataset 
dataset = pd.read_csv("E:\Project\Machine Learing\paid\car_sales_data.csv")

dataset.head()

dataset.shape
# taking care of missing values

dataset.isnull().values.any()

dataset.describe()

dataset['Manufacturer'].unique()

dataset['Model'].unique()

dataset['Model'].unique().size

dataset['Fuel type'].unique()

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 6:].values

# spliting dataset into the Traing data set and  Testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size  = 1/3, random_state = 0)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor_engine_size  = LinearRegression()
regressor_Year_of_manufacture  = LinearRegression()
regressor_Mileage = LinearRegression()


regressor_engine_size.fit(X_train_engine, Y_train)
regressor_Year_of_manufacture.fit(X_train_manufacture, Y_train)
regressor_Mileage.fit(X_train_mileage, Y_train)

# Calculating the R-square value for each numerical features's model

engine = regressor_engine_size.score(X_test[:, 2:3], Y_test)
manufacture = regressor_Year_of_manufacture.score(X_test[:, 4:5], Y_test)
mileage = regressor_Mileage.score(X_test[:, 5:6], Y_test)

print(f"The R-square Score in Linear Regression using\n\nEngine size = {engine}\nYear of manufacture = {manufacture}\nMileage = {mileage}")

# PolynomialFeatures with degree 2 (adjust the degree)

from sklearn.preprocessing import PolynomialFeatures

poly_engine = PolynomialFeatures(degree=2)
poly_manufacture = PolynomialFeatures(degree=2)
poly_mileage = PolynomialFeatures(degree=2)


poly_regressor_engine  = LinearRegression()
poly_regressor_manufacture  = LinearRegression()
poly_regressor_mileage = LinearRegression()


poly_regressor_engine.fit(poly_engine.fit_transform(X_train_engine), Y_train)
a = poly_regressor_engine.score(poly_engine.fit_transform(X_test[:, 2:3]), Y_test)

poly_regressor_manufacture.fit(poly_manufacture.fit_transform(X_train_manufacture), Y_train)
b = poly_regressor_engine.score(poly_engine.fit_transform(X_test[:, 4:5]), Y_test)

poly_regressor_mileage.fit(poly_mileage.fit_transform(X_train_mileage), Y_train)
c = poly_regressor_engine.score(poly_engine.fit_transform(X_test[:, 5:6]), Y_test)


print(f"The R-square Score in Polynomial Regression using\n\nEngine size = {engine}\nYear of manufacture = {manufacture}\nMileage = {mileage}")

X_multiple= dataset[['Engine size', 'Year of manufacture', 'Mileage']]
y_multiple = dataset['Price']

X_train_mul, X_test_mul, y_train_mul, y_test_mul = train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=0)

regressor1 = LinearRegression()
regressor2 = LinearRegression()

regressor1.fit(X_train_mul, y_train_mul)

poly_mul = PolynomialFeatures(degree = 2)
regressor2.fit(poly_mul.fit_transform(X_train_mul), y_train_mul)

pa = regressor1.score(X_test_mul, y_test_mul)

pb = regressor2.score(poly_mul.fit_transform(X_test_mul), y_test_mul)

print(f"R-squre Score in Linear Regression = {pa}\nR-square Score in Polynomial Regression = {pb}")

# libraries 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Extract features (X) and target variable (y)
X = dataset.drop('Price', axis=1)
y = dataset['Price'] 


# Perform categorical encoding on the 'Manufacturer', 'Model' and 'Fuel type' columns

encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = pd.DataFrame(encoder.fit_transform(X[['Manufacturer', 'Model', 'Fuel type']]))
# Concatenate the encoded features with the original dataset
X = pd.concat([X, X_encoded], axis=1)

X

X = X.drop(['Manufacturer', 'Model', 'Fuel type'], axis = 1)

# convert all features name string

X.columns = X.columns.astype(str)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


# Train the model on the training data
rf_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(x_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(r2)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(r2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# standardize
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.fit_transform(x_test)

# build ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(128, activation = 'relu', input_shape = (x_train_scaled.shape[1],)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# compiling the model

model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error'
)

# fit the model using the training data 

model.fit(x_train_scaled, y_train, epochs = 50, batch_size = 32, validation_split = 0.1)

# make prediction using the testing data

y_pred = model.predict(x_test_scaled)
y_pred

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R-square score in ANN model: {r2}')


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Drop the original categorical columns
data = dataset.drop(['Manufacturer', 'Model', 'Fuel type'], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data

# Apply k-Means clustering with different k values

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Evaluate clustering
    inertia = kmeans.inertia_
    silhouette = silhouette_score(scaled_data, clusters)
    davies_bouldin = davies_bouldin_score(scaled_data, clusters)

    print(f'Clusters: {k}, Inertia: {inertia}, Silhouette: {silhouette}, Davies-Bouldin: {davies_bouldin}')


scaled_data.shape

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)

# Subsample data
from sklearn.utils import shuffle
subsample_size = 5000  # You can adjust this based on your memory constraints
subsampled_data = shuffle(scaled_data, random_state=42)[:subsample_size]

agg_labels = agg.fit_predict(subsampled_data)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(subsampled_data)

# Evaluate clustering for Hierarchical Clustering
agg_silhouette = silhouette_score(subsampled_data, agg_labels)
agg_davies_bouldin = davies_bouldin_score(subsampled_data, agg_labels)

# Evaluate k means clustering
kmeans_silhouette = silhouette_score(subsampled_data, clusters)
kmeans_davies_bouldin = davies_bouldin_score(subsampled_data, clusters)

print('\nHierarchical Clustering:')
print(f'Silhouette: {agg_silhouette}, Davies-Bouldin: {agg_davies_bouldin}')

print('\nK-means Clustering:')
print(f'Silhouette: {kmeans_silhouette}, Davies-Bouldin: {kmeans_davies_bouldin}')

