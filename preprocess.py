import numpy as np
import pandas as pd

import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# for type1 dataset
output_folder = r"D:\Datasets\Mining_output"
train_df = pd.read_csv(r"D:\Users\wt\Downloads\production_quality_prediction\train_left.csv")
test_df = pd.read_csv(r"D:\Users\wt\Downloads\production_quality_prediction\test_left.csv")
train_df.drop("date", axis=1, inplace=True)
test_df.drop("date", axis=1, inplace=True)

extract_col = list(range(0,21,1))
extract_col = extract_col+[23]






scaler = StandardScaler()
scaler.fit(train_df.iloc[:, :22])
train_dfX = pd.DataFrame(scaler.transform(train_df.iloc[:, :22]), index=train_df.index,
                         columns=train_df.columns[:-1])
train_dfX["% Silica Concentrate"] = train_df["% Silica Concentrate"]
val_dfX = pd.DataFrame(scaler.transform(val_df.iloc[:, :22]), index=val_df.index,
                       columns=val_df.columns[:-1])
val_dfX["% Silica Concentrate"] = val_df["% Silica Concentrate"]
test_dfX = pd.DataFrame(scaler.transform(test_df.iloc[:, :22]), index=test_df.index,
                        columns=test_df.columns[:-1])
test_dfX["% Silica Concentrate"] = test_df["% Silica Concentrate"]

print("the split train-val-test is ", len(train_dfX), len(val_dfX), len(test_dfX))
train_dfX.to_csv(os.path.join(output_folder, 'train.csv'))
val_dfX.to_csv(os.path.join(output_folder, 'val.csv'))
test_dfX.to_csv(os.path.join(output_folder, 'test.csv'))
