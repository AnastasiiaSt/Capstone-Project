import pandas as pd
from pandas_profiling import ProfileReport

from plotnine import (
    ggplot, aes, geom_point, geom_smooth, labs, theme_xkcd
)


# Read the datasets from the files

train_set = pd.read_csv(r'C:\Users\anast\Documents\GitHub\Capstone-Project\data\train.csv')
test_set = pd.read_csv(r'C:\Users\anast\Documents\GitHub\Capstone-Project\data\test.csv')

print('Training dataset size:', train_set.shape)
print('Training dataset columns:', train_set.columns.to_list())

print('Test dataset size:', test_set.shape)
print('Test dataset columns:', test_set.columns.to_list())

train_y = train_set['Cover_Type']
train_X = train_set.drop(['Cover_Type'], axis = 1)

test_X = test_set

# Perform EDA using pandas profiling library

report_train = ProfileReport(train_set)

report_train.to_file(r'C:\Users\anast\Documents\GitHub\Capstone-Project\capstone_project\profile_report_train.html')



