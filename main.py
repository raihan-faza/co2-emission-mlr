import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

"""
Importing dataset, checking for null
"""

df = pd.read_csv("co2.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.isnull().sum())
print(y)


"""
checking for linearity using pair plot
"""
plt.figure(figsize=(20, 20))
sns.pairplot(df)
plt.show()


"""
splitting training and validation data
"""
X = df[["Engine Size(L)"]]
y = df.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.2, random_state=20)

"""
train model
"""
model = LinearRegression()
model.fit(X_train, y_train)

"""
predict using model
"""
y_pred = model.predict(X_val)

"""
eval model
"""
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(model.score(X_train, y_train))
print(f"mse:{mse}\nr2:{r2}")


"""
plot regression line
"""
plt.figure(figsize=(10, 10))
plt.scatter(y_val, y_pred, alpha=0.4, color="purple", label="Actual Values")
plt.plot(
    [y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "--", lw=2, label="(y=x)"
)
plt.xlabel("Actual Values")
plt.ylabel("Estimated Values")
plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
