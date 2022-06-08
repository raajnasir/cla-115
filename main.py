import pandas as pd
import plotly.express as px

df = pd.read_csv("data2.csv")

temperature_list = df["Temperature"].tolist()
melted_list = df["Melted"].tolist()

fig = px.scatter(x=temperature_list, y=melted_list)
fig.show()

import numpy as np
temperature_array = np.array(temperature_list)
melted_array = np.array(melted_list)


#Slope and intercept using pre-built function of Numpy

m, c = np.polyfit(temperature_array, melted_array, 1)

y = []
for x in temperature_array :
     y_value = m*x + c
     y.append(y_value)

#plotting the graph
fig = px.scatter(x=temperature_array, y=melted_array)
fig.update_layout(shapes = [
    dict(
        type = 'line',
        y0 = min(y), y1 = max(y), 
        x0 = min(temperature_array), x1 = max(temperature_array)
    )
])
fig.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.reshape(temperature_list, (len(temperature_list), 1))
Y = np.reshape(melted_list, (len(melted_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line formula 
X_test = np.linspace(0, 5000, 10000)
melting_chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, melting_chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

#do hit and trial by changing the vlaue of X_test here.
plt.axvline(x=X_test[6843], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(3400, 3450)
plt.show()
print(X_test[6843])