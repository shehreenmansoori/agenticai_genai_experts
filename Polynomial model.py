import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\SHIROZ\Downloads\emp_sal.csv")

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x), color ='blue')
plt.title('Linear regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_model_predict = lin_reg.predict([[6.5]])
print(lin_model_predict)

#polynomial model

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x_poly= poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x,y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color ='blue')
plt.title('Truth or bluff (poly regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

poly_model_pred= lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)  
