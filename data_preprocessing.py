# This script pre-processes the data.

print("\033[2J")

import warnings
import seaborn as sb
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# *** SETTINGS **************
X = Y = X_train = X_test = Y_train = Y_test= FEATURES = None
INPUT_FILE = 'DATA_FULL.csv'    
OUTPUT_FILE_TRAIN = 'DATA_TRAIN.csv'
OUTPUT_FILE_TEST = 'DATA_TEST.csv'
OUTPUT_FILE_MEANS = 'MEANS.csv'
OUTPUT_FILE_VARIANCES = 'VARIANCES.csv'

def readData():       
    global X, Y, FEATURES
    dataFrame = pd.read_csv(INPUT_FILE, sep=';', decimal='.')                
    (rowsNumber, columnsNumber) = dataFrame.shape            
    X = dataFrame.iloc[:, 0:columnsNumber-1]
    Y = dataFrame.iloc[:, columnsNumber-1]    
    FEATURES = list(X.columns)      

def splitData():    
    global X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)  

def rescaleFeatures():
    global X_train, X_test, Y_train, Y_test, FEATURES
    scaler = StandardScaler()        
    # *** Train set
    X_train = scaler.fit_transform(X_train)    
    #The mean values and variances for each feature of the learning set are saved 
    # in order to transform the new data based on them in an identical way during prediction
    means = pd.DataFrame(scaler.mean_)
    means.to_csv (OUTPUT_FILE_MEANS, sep=';', index = None, header=None)
    variances = pd.DataFrame(scaler.var_)
    variances.to_csv (OUTPUT_FILE_VARIANCES, sep=';', index = None, header=None)    
    X_train = pd.DataFrame(X_train)
    X_train.columns = FEATURES
    Y_train = Y_train.values
    # *** Test Set
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test)    
    X_test.columns = FEATURES
    Y_test = Y_test.values
   
# Removal of features with fixed values
def removingConstantFeatures():    
    global X_train, X_test
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    len(X_train.columns[constant_filter.get_support()])    
    constant_columns = [column for column in X_train.columns
        if column not in X_train.columns[constant_filter.get_support()]]    
    print('\nNumber of features with a constant value:', len(constant_columns))
    print('Features with a constant value:\n', constant_columns)    
    X_train.drop(labels=constant_columns, axis=1, inplace=True)
    X_test.drop(labels=constant_columns, axis=1, inplace=True)    
    print('Number of features after removal of features with fixed value:', X_train.shape[1])    

# Removal of features with near-constant values
def removingQuasiConstantFeatures():
    global X_train, X_test
    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(X_train)
    len(X_train.columns[qconstant_filter.get_support()])
    qconstant_columns = [column for column in X_train.columns
        if column not in X_train.columns[qconstant_filter.get_support()]]            
    print('\nNumber of features with near-constant value:', len(qconstant_columns))
    print('Features with a near-constant value:\n', qconstant_columns)
    X_train.drop(labels=qconstant_columns, axis=1, inplace=True)
    X_test.drop(labels=qconstant_columns, axis=1, inplace=True)    
    print('Number of features after removal of features with near-constant value:', X_train.shape[1])

# Removal of duplicate features
def removingDuplicateFeatures():    
    global X_train, X_test
    X_train_T = X_train.T    
    print('\nNumber of features with duplicate values:', X_train_T.duplicated().sum())
    unique_features = X_train_T.drop_duplicates(keep='first').T        
    duplicated_features = [dup_col for dup_col in X_train.columns if dup_col not in unique_features.columns]
    X_train = unique_features    
    print('Features with duplicate values:\n', duplicated_features)
    print('Number of features after removing features with duplicate values:', X_train.shape[1])    

# Removal of features correlated with each other
def removingCorrelatedFeatures():        
    global X_train, X_test
    # Features correlated with other features more strongly than the set threshold are removed
    correlationThreshold = 0.9
    correlated_features = set()
    correlation_matrix = X_train.corr(method='pearson')
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlationThreshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    X_train.drop(labels=correlated_features, axis=1, inplace=True)
    X_test.drop(labels=correlated_features, axis=1, inplace=True)
    print('\nNumber of features correlated with each other:', len(correlated_features), '\t(Pearson\'s linear correlation coefficient r >', correlationThreshold, ') - very strong relationship')
    print('Features correlated with each othe:\n', correlated_features)
    print('Number of features after removal of correlated features:', X_train.shape[1])
    plotCorrelationMatrix()    
    
def plotCorrelationMatrix():
    global X_train
    correlationMatrix = X_train.corr(method='pearson')
    sb.heatmap(correlationMatrix, xticklabels=correlationMatrix.columns, yticklabels=correlationMatrix.columns,
               cmap='RdBu_r', annot=True, linewidth=0.5)    

def saveData():   
    X_train['F'] = Y_train
    X_train.to_csv (OUTPUT_FILE_TRAIN, sep=';', index = None, float_format='%g')  
    X_test['F'] = Y_test
    X_test.to_csv (OUTPUT_FILE_TEST, sep=';', index = None, float_format='%g')  
    print('\nFiles created:', OUTPUT_FILE_TRAIN, OUTPUT_FILE_TEST)
    
def main():        
    warnings.filterwarnings("ignore")    
    readData()    
    splitData()               
    rescaleFeatures()            
    removingConstantFeatures() # Removal of features with fixed values
    removingQuasiConstantFeatures()  # Removal of features with near-constant values
    removingDuplicateFeatures() # Removal of duplicate features
    removingCorrelatedFeatures() # Removal of mutually correlated features
    saveData()

if __name__ == "__main__":
    main()
