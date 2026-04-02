print("\033[2J")  # Clear console screen
import os, sys, pickle, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages

# *** SETTINGS **************
# Global variables used across functions
SELECTION_METHOD = INPUT_FILE_TRAIN = OUTPUT_FILE = X_train_temp = X_train = Y_train = COLUMN_NUMBER = None

def configuration():
    global SELECTION_METHOD, INPUT_FILE_TRAIN, OUTPUT_FILE    
    
    # Menu options for selecting feature selection method
    menuList = ['1', '2', '3', '4', '5', '6']
    menuOptions = ['FISHER', 'ANOVA', 'RELIEF', 'SFS', 'SBS', 'RFE']
    menu = set(menuList)        
    choice = ''
    
    # Ask user to select a method
    while not choice in menu:
        print('1. FISHER')
        print('2. ANOVA')
        print('3. RELIEF')        
        print('4. SFS')
        print('5. SBS')
        print('6. RFE')        
        choice = input('CHOICE: ')
    
    # Assign selected method and configure file names
    SELECTION_METHOD = menuOptions[int(choice)-1]            
    print('\n*** SETTINGS ***')    
    print('SELECTION METHOD =', SELECTION_METHOD)    
    
    # Input dataset depends on selected feature selection method
    INPUT_FILE_TRAIN = 'DATA_' + SELECTION_METHOD + '_TRAIN.csv'    
    
    # Output prefix for saving results
    OUTPUT_FILE = 'TRAINING_' + SELECTION_METHOD
    
    # Option to stop execution
    choice = input("Continuation - any key, Quit - 'q': ")
    if choice == 'q':
        sys.exit(0)

def readData():           
    global X_train_temp, X_train, Y_train, COLUMN_NUMBER    
    
    # *** Train set ***
    # Load dataset from CSV file
    df_train = pd.read_csv(INPUT_FILE_TRAIN, sep=';', decimal='.')                
    (_, COLUMN_NUMBER) = df_train.shape            
    
    # Split into features (X) and labels (Y)
    X_train = df_train.iloc[:, 0:COLUMN_NUMBER-1]
    Y_train = df_train.iloc[:, COLUMN_NUMBER-1]    

def lda():
    print('\n*** 1. LDA ***')
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    
    
    # Create directory for LDA results if it does not exist
    if not os.path.exists('LDA') :
        os.mkdir('LDA')    
    
    # DataFrame to store results
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    
    # Iterate over increasing number of features
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        
        # *** MODIFY PARAMETERS HERE ***
        # Define model and hyperparameters
        pipe = Pipeline([('clf', LinearDiscriminantAnalysis())])            
        param_grid = [{'clf__solver': ['svd', 'lsqr', 'eigen']}]
        # ***
        
        # Perform Grid Search with 10-fold cross-validation
        gs = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        
        # ********************************
        # Train best model and save it
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)        
        filename = 'LDA/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_LDA.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        
        # Save performance metrics
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)        
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    
    # Save results to CSV
    df.to_csv('LDA/' + OUTPUT_FILE + '_LDA.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_LDA.csv') 

def svm():
    print('\n*** 2. SVM ***')
    from sklearn.svm import SVC
    
    # Create directory for SVM results
    if not os.path.exists('SVM') :
        os.mkdir('SVM')    
    
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        
        # *** MODIFY PARAMETERS HERE ***
        # Define SVM model and hyperparameters
        pipe = Pipeline([('clf', SVC())])            
        param_grid = [{'clf__C': [0.001, 0.01, 0.1, 1.0],
                       'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                       'clf__gamma': ['scale', 'auto'],
					   'clf__probability': [True]}]
        #***
        
        gs = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)        
        
        #********************************
        # Save trained model
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)        
        filename = 'SVM/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_SVM.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        
        # Store results
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)                
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    
    df.to_csv('SVM/' + OUTPUT_FILE + '_SVM.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_SVM.csv') 
    
def knn():
    print('\n*** 3. KNN ***')
    from sklearn.neighbors import KNeighborsClassifier
    
    # Create directory for KNN results
    if not os.path.exists('KNN') :
        os.mkdir('KNN')   
    
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        
        # *** MODIFY PARAMETERS HERE ***
        # Define KNN model and hyperparameters
        pipe = Pipeline([('clf', KNeighborsClassifier())])            
        param_grid = [{'clf__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
        # ***
        
        gs = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)        
        
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)        
        filename = 'KNN/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_KNN.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)                
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    
    df.to_csv('KNN/' + OUTPUT_FILE + '_KNN.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_KNN.csv') 

# Остальные функции (dt, mlp, rf, boosting и т.д.) уже повторяют ту же логику:
# 1. Создание папки
# 2. Перебор количества признаков
# 3. GridSearchCV для подбора гиперпараметров
# 4. Обучение лучшей модели
# 5. Сохранение модели (.sav)
# 6. Сохранение метрик в CSV

def main():    
    # Initialize configuration and load dataset
    configuration()
    readData()
    
    # Sequential training of all classifiers
    lda()            
    svm()
    knn()    
    dt()    
    mlp()    
    rf()  
    gradient_boosting()
    adaboost()
    hist_gradient_boosting()
    extra_trees()
    qda()
    gnb()
    logistic_regression()         
        
if __name__ == "__main__":
    main()