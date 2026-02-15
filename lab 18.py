import pandas as pd
from sklearn.preprocessing import   OneHotEncoder

df18s = pd.read_csv("day18_binning.csv")

df18s["age_bin_3"] = pd.cut(df18s["age"], 3)
print(df18s[["age", "age_bin_3"]].head())
bin_edge = [0, 18, 35, 50, 100]
labelz = ["child", "YoungAdult", "adult", "Senior"]
df18s["age_bin_width"] = pd.cut(df18s["age"], bins=bin_edge, labels=labelz, right=False)
print(df18s["age_bin_width"].value_counts())
df18s["age_bins_quantiles"] = pd.qcut(df18s["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
print(df18s["age_bins_quantiles"].value_counts())

age_edges = [0, 13, 18, 65, 120]
age_labels = ["child", "Teen", "adult", "Senior"]
df18s["age_group"] = pd.cut(df18s["age"], bins=age_edges,  labels=age_labels, right=False)
print(df18s["age_group"].value_counts())

ohe = OneHotEncoder(handle_unknown="ignore")
age_encoded = ohe.fit_transform(df18s[["age"]])
print("Encoded shape:", age_encoded.shape, "Categories:", ohe.categories_)
