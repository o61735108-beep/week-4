import pandas as pd
df17 = pd.read_csv("day17_scaling.csv")

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
df17_mm = df17.copy()
df17_mm[["CRIM_mm", "RM_mm"]] = mm_scaler.fit_transform(df17[["CRIM", "RM"]])
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
df17_std = df17.copy()
df17_std[["CRIM_std", "RM_std"]] = std_scaler.fit_transform(df17[["CRIM", "RM"]])
print(df17_std[["CRIM_std", "RM_std"]].agg(["mean", "std"]))
rob_scaler = RobustScaler()
df17_rob = df17.copy()
df17_rob[["CRIM_rob", "RM_rob"]] = rob_scaler.fit_transform(df17[["CRIM", "RM"]])
