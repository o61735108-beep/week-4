import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
encoding = pd.read_csv("day16_encoding.csv")
df16 = pd.DataFrame(encoding)

le_city = LabelEncoder()
df16["city_label"] = le_city.fit_transform(df16["city"])
print("Classes:", le_city.classes_)
df16_ohe = pd.get_dummies(df16, columns=["city"], prefix="city")
print(df16_ohe.head())
ohe = OneHotEncoder( handle_unknown="ignore")
city_encoded = ohe.fit_transform(df16[["city"]])
print("Encoded shape:", city_encoded.shape, "Categories:", ohe.categories_)