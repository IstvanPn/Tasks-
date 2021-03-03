import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013],[2014],[2015],[2016],[2017],[2018],[2019],[2022],[2021],[2022],[2023]]
y_train = [[550],[600],[650],[700],[700],[750],[760],[780],[790],[900],[990],[1020],[950],[930],[920],[890],[880],[900]]

# Testing set
x_test = [[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013],[2014],[2015],[2016],[2017],[2018],[2019],[2022],[2021],[2022],[2023]]
y_test = [[550],[600],[650],[700],[700],[750],[760],[780],[790],[900],[990],[1020],[950],[930],[920],[890],[880],[900]]


# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(2005, 2025, 1000)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
X_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

#a method that lets us make a polynomial model:

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Price of affordable Smartphones')
plt.xlabel('Years')
plt.ylabel('Price in Rands')
plt.axis([2003, 2026, 0, 1500])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print(x_train)
print(X_train_quadratic)
print(x_test)
print(X_test_quadratic)
