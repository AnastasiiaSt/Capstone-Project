import pandas as pd
from pandas_profiling import ProfileReport
import os

# Read the datasets from the files

path = r'C:\Users\anast\Documents\GitHub\Capstone-Project\data'

def get_data(path):
    print('runnnnn')
    file_name_train = 'train.csv'
    file_name_test = 'test.csv'
    file_train = os.path.join(path, file_name_train)
    file_test = os.path.join(path, file_name_test)
    train_set = pd.read_csv(file_train)
    test_set = pd.read_csv(file_test)

    print('Training dataset size:', train_set.shape)
    print('Training dataset columns:', train_set.columns.to_list())
    print('Test dataset size:', test_set.shape)
    print('Test dataset columns:', test_set.columns.to_list())

    train_y = train_set['Cover_Type']
    train_X = train_set.drop(['Cover_Type'], axis = 1)
    test_X = test_set
    return train_X, train_y, test_X

get_data(path = path)

# Perform EDA using pandas profiling library

def profile_report(dataset):
    report = ProfileReport(dataset)
    #report.to_file(r'C:\Users\anast\Documents\GitHub\Capstone-Project\capstone_project\profile_report_train.html')
    return report




