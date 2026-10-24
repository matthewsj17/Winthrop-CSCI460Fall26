######################################################

#  Student Name: Jake Matthews

#  Course Code: CSCI 460 -- Fall 2026

#  Assignment Due Date: 10/25/26

#  GitHub Link: https://github.com/matthewsj17/Winthrop-CSCI460Fall26/tree/main/Assignment1

######################################################

# Task I: Data Download
import pandas as pd
import numpy as np
import sklearn
path = 'bank-full.csv'
df = pd.read_csv(path, sep=';')
# df.isna().sum()
# no nans
# df.dropna(inplace=True)

# Mapping Categoricals
# ‘job’, 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome',
d = {"yes":1,'no':0}
df['job']= df['job'].map({'admin.':0, 'blue-collar':1, 'entrepreneur':2,
               'housemaid':3, 'management':4, 'retired':5,
               'self-employed':6, 'services':7, 'student':8,
               'technician':9, 'unemployed':10, 'unknown':11})
df['marital']= df['marital'].map({'divorced':0, 'single':1, 'married':2}, )
df['education']= df['education'].map({'unknown':0,'primary':1,'secondary':2,'tertiary':3})
df['default'] = df['default'].map(d)
df['housing'] = df['housing'].map(d)
df['loan'] = df['loan'].map(d)
df['contact'] = df['contact'].map({'unknown':0, 'cellular':1,'telephone':2})
df['month'] = df['month'].map({'jan':1,'feb':2, 'mar':3,
                               'apr':4,'may':5,'jun':6,
                               'jul':7,'aug':8,'sep':9,
                               'oct':10,'nov':11,'dec':12})
df['poutcome'] = df['poutcome'].map({'unknown':0,'failure':1,'success':2,'other':3})
df['y'] = df['y'].map(d)

y = df['y']
X = df.drop('y', axis=1)


# Task II: Train and Test a Naive Bayes Model

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score, 
    classification_report, 
) 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True, stratify=y)

model = GaussianNB()
model.fit(X_train, y_train) 

def runProg(seed):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=seed, shuffle=True, stratify=y)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test) 
    f1 = f1_score(y_pred, y_test, average="weighted") 
    return [accuracy, f1]


# create seed values, attempting to do 'true random'
import time
rng = np.random.default_rng(seed=int(time.time())) #seed for random gen is the UTC timestep.
random_array = rng.integers(low=0, high=10000, size=10) # ensures these are different everytime
# Iterate 10 TImes, collect results into an array.
arr = []
for i in random_array: 
    arr.append(runProg(i)) #which ensures these are different everytime.

task2df = pd.DataFrame(arr, columns=['Accuracy','F1 Score'])
task2df.index.name = 'Testing Iteration'
task2df.to_csv('task2results.csv')

# Task III: report impact of splitting the data 

def ttsProg(seed, testSize):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=testSize, random_state=seed, shuffle=True, stratify=y)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test) 
    f1 = f1_score(y_pred, y_test, average="weighted") 
    return [accuracy, f1]

testSizes = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

bigDict = {}

for test in testSizes:
    tempArr = []
    for i in range(10):
        tempArr.append(ttsProg(i, test))
    bigDict[f'Test_{test}'] = tempArr

bigDict

task3DF = pd.DataFrame.from_dict(bigDict)
task3DF
task3accuracy = pd.DataFrame()
task3std = pd.DataFrame()
for column in task3DF.columns:
    acc = []
    std = []
    for row in task3DF[column]:
        acc.append(row[0])
        std.append(row[1])

    task3accuracy[column] = acc
    task3std[column] = std
# Extract Data and Create Meaningful INformation From it.  
# test90arr = bigDict['Test_0.9']
# task3DF.to_csv('task3results.csv')
task3std.to_csv('std3results.csv')
task3accuracy.to_csv('acc3results.csv')
