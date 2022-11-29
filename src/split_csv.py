import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"E:\SQA\TencentCorups\withReverberationTrainDevMOS.csv")
train, val, xlabel, ylabel = train_test_split(
    data, range(data.shape[0]), test_size=0.2, random_state=20)
train.insert(loc=0, column='db', value='TRAIN')
train.to_csv('./with_train.csv', index=False)
val.insert(loc=0, column='db', value='VAL')
val.to_csv('./with_val.csv', index=False)
