


print("\033[2J")
import os, sys, pickle, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# *** SETTINGS **************
SELECTION_METHOD = INPUT_FILE_TRAIN = OUTPUT_FILE = X_train_temp = X_train = Y_train = COLUMN_NUMBER = None

def configuration():
    global SELECTION_METHOD, INPUT_FILE_TRAIN, OUTPUT_FILE    
    menuList = ['1', '2', '3', '4', '5', '6']
    menuOptions = ['FISHER', 'ANOVA', 'RELIEF', 'SFS', 'SBS', 'RFE']
    menu = set(menuList)        
    choice = ''
    while not choice in menu:
        print('1. FISHER')
        print('2. ANOVA')
        print('3. RELIEF')        
        print('4. SFS')
        print('5. SBS')
        print('6. RFE')        
        choice = input('CHOICE: ')
    SELECTION_METHOD = menuOptions[int(choice)-1]            
    print('\n*** SETTINGS ***')    
    print('SELECTION METHOD =', SELECTION_METHOD)    
    INPUT_FILE_TRAIN = 'DATA_' + SELECTION_METHOD + '_TRAIN.csv'    
    OUTPUT_FILE = 'TRAINING_' + SELECTION_METHOD
    choice = input("Continuation - any key, Quit - 'q': ")
    if choice == 'q':
        sys.exit(0)

def readData():           
    global X_train_temp, X_train, Y_train, COLUMN_NUMBER    
    # *** Train set ***
    df_train = pd.read_csv(INPUT_FILE_TRAIN, sep=';', decimal='.')                
    (_, COLUMN_NUMBER) = df_train.shape            
    X_train = df_train.iloc[:, 0:COLUMN_NUMBER-1]
    Y_train = df_train.iloc[:, COLUMN_NUMBER-1]    

def lda():
    print('\n*** 1. LDA ***')
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    
    if not os.path.exists('LDA') :
        os.mkdir('LDA')    
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', LinearDiscriminantAnalysis())])            
        param_grid = [{'clf__solver': ['svd', 'lsqr', 'eigen']}]
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
        filename = 'LDA/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_LDA.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)        
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv('LDA/' + OUTPUT_FILE + '_LDA.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_LDA.csv') 

def svm():
    print('\n*** 2. SVM ***')
    from sklearn.svm import SVC
    if not os.path.exists('SVM') :
        os.mkdir('SVM')    
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        # *** MODIFY PARAMETERS HERE ***
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
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)        
        filename = 'SVM/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_SVM.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)                
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv('SVM/' + OUTPUT_FILE + '_SVM.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_SVM.csv') 
    
def knn():
    print('\n*** 3. KNN ***')
    from sklearn.neighbors import KNeighborsClassifier
    if not os.path.exists('KNN') :
        os.mkdir('KNN')   
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        # *** MODIFY PARAMETERS HERE ***
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

def dt():
    print('\n*** 4. DT ***')
    from sklearn.tree import DecisionTreeClassifier
    if not os.path.exists('DT') :
        os.mkdir('DT')   
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', DecisionTreeClassifier())])            
        param_grid = [{'clf__criterion': ['gini', 'entropy'],
                      'clf__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
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
        filename = 'DT/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_DT.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)                
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv('DT/' + OUTPUT_FILE + '_DT.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_DT.csv') 

def mlp():
    print('\n*** 5. MLP ***')
    from sklearn.neural_network import MLPClassifier
    if not os.path.exists('MLP') :
        os.mkdir('MLP')   
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', MLPClassifier())])            
        param_grid = [{'clf__hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)], # Number of neurons in one hidden layer
                      'clf__activation': ['relu'],
                      'clf__solver': ['lbfgs'],
                      'clf__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                      'clf__max_iter': [20000]}]
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
        filename = 'MLP/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_MLP.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)                
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv('MLP/' + OUTPUT_FILE + '_MLP.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_MLP.csv') 

def rf():
    print('\n*** 6. RF ***')
    from sklearn.ensemble import RandomForestClassifier
    if not os.path.exists('RF') :
        os.mkdir('RF')   
    df = pd.DataFrame(columns = ['N', 'ACC', 'Par'])     
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]        
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', RandomForestClassifier())])            
        param_grid = [{'clf__n_estimators': [50, 100, 150, 300, 500],
                      'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}]
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
        filename = 'RF/MODEL_' + str(columnRange) + '_' + SELECTION_METHOD + '_RF.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)                
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv('RF/' + OUTPUT_FILE + '_RF.csv', sep=';', index = None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_RF.csv') 

def gradient_boosting():
    print('\n*** 7. Gradient Boosting ***')
    from sklearn.ensemble import GradientBoostingClassifier
    if not os.path.exists('GB'):
        os.mkdir('GB')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', GradientBoostingClassifier())])
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7]
        }
        # ***
        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)
        filename = f'GB/MODEL_{columnRange}_{SELECTION_METHOD}_GB.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv(f'GB/{OUTPUT_FILE}_GB.csv', sep=';', index=None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_GB.csv')

def adaboost():
    print('\n*** 8. AdaBoost ***')
    from sklearn.ensemble import AdaBoostClassifier
    if not os.path.exists('AB'):
        os.mkdir('AB')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', AdaBoostClassifier())])
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__learning_rate': [0.01, 0.1, 1]
        }
        # ***
        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)
        filename = f'AB/MODEL_{columnRange}_{SELECTION_METHOD}_AB.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print('N:', columnRange, 'ACC: {0:.2f}'.format(gs.best_score_), 'Par:', gs.best_params_)
    df.to_csv(f'AB/{OUTPUT_FILE}_AB.csv', sep=';', index=None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_AB.csv')


def hist_gradient_boosting():
    print('\n*** 9. HistGradientBoostingClassifier ***')
    from sklearn.ensemble import HistGradientBoostingClassifier
    if not os.path.exists('HGB'):
        os.mkdir('HGB')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', HistGradientBoostingClassifier())])
        param_grid = [{'clf__max_iter': [100, 200, 300],
                       'clf__learning_rate': [0.01, 0.1, 0.2],
                       'clf__max_depth': [None, 3, 5]}]
        # **
        gs = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)
        filename = f'HGB/MODEL_{columnRange}_{SELECTION_METHOD}_HGB.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print(f'N: {columnRange}, ACC: {gs.best_score_:.2f}, Par: {gs.best_params_}')
    df.to_csv(f'HGB/{OUTPUT_FILE}_HGB.csv', sep=';', index=None, float_format='%.2f')
    print(f'File created: {OUTPUT_FILE}_HGB.csv')

def extra_trees():
    print('\n*** 10. ExtraTreesClassifier ***')
    from sklearn.ensemble import ExtraTreesClassifier
    if not os.path.exists('ET'):
        os.mkdir('ET')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', ExtraTreesClassifier())])
        param_grid = [{'clf__n_estimators': [50, 100, 200],
                       'clf__max_depth': [None, 10, 20],
                       'clf__min_samples_split': [2, 5, 10]}]
        # **
        gs = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)
        filename = f'ET/MODEL_{columnRange}_{SELECTION_METHOD}_ET.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print(f'N: {columnRange}, ACC: {gs.best_score_:.2f}, Par: {gs.best_params_}')
    df.to_csv(f'ET/{OUTPUT_FILE}_ET.csv', sep=';', index=None, float_format='%.2f')
    print(f'File created: {OUTPUT_FILE}_ET.csv')

def qda():
    print('\n*** 11. Quadratic Discriminant Analysis (QDA) ***')
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    if not os.path.exists('QDA'):
        os.mkdir('QDA')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', QuadraticDiscriminantAnalysis())])
        param_grid = {'clf__reg_param': [0.0, 0.1, 0.5, 1.0]}
        # ***
        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)
        filename = f'QDA/MODEL_{columnRange}_{SELECTION_METHOD}_QDA.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print(f'N: {columnRange}, ACC: {gs.best_score_:.2f}, Par: {gs.best_params_}')
    df.to_csv(f'QDA/{OUTPUT_FILE}_QDA.csv', sep=';', index=None, float_format='%.2f')
    print(f'File created: {OUTPUT_FILE}_QDA.csv')

def gnb():
    print('\n*** 12. Gaussian Naive Bayes (GNB) ***')
    from sklearn.naive_bayes import GaussianNB
    if not os.path.exists('GNB'):
        os.mkdir('GNB')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', GaussianNB())])
        param_grid = {'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        # ***
        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X_train_temp, Y_train)
        # ********************************
        clf = gs.best_estimator_
        clf.fit(X_train_temp, Y_train)
        filename = f'GNB/MODEL_{columnRange}_{SELECTION_METHOD}_GNB.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print(f'N: {columnRange}, ACC: {gs.best_score_:.2f}, Par: {gs.best_params_}')
    df.to_csv(f'GNB/{OUTPUT_FILE}_GNB.csv', sep=';', index=None, float_format='%.2f')
    print(f'File created: {OUTPUT_FILE}_GNB.csv')

def logistic_regression():
    print('\n*** 13. Logistic Regression ***')
    from sklearn.linear_model import LogisticRegression
    if not os.path.exists('LogReg'):
        os.mkdir('LogReg')
    df = pd.DataFrame(columns=['N', 'ACC', 'Par'])
    for columnRange in range(2, COLUMN_NUMBER):
        X_train_temp = X_train.iloc[:, 0:columnRange]
        # *** MODIFY PARAMETERS HERE ***
        pipe = Pipeline([('clf', LogisticRegression(max_iter=1000))])
        param_grid = {'clf__C': [0.01, 0.1, 1, 10],
                      'clf__solver': ['liblinear', 'lbfgs', 'saga']}
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
        filename = f'LogReg/MODEL_{columnRange}_{SELECTION_METHOD}_LogReg.sav'
        pickle.dump(clf, open(filename, 'wb'))
        # ********************************
        row = {'N': columnRange, 'ACC': gs.best_score_, 'Par': gs.best_params_}
        df = df._append(row, ignore_index=True)
        print(f'N: {columnRange}, ACC: {gs.best_score_:.2f}, Par: {gs.best_params_}')
    df.to_csv(f'LogReg/{OUTPUT_FILE}_LogReg.csv', sep=';', index=None, float_format='%.2f')
    print('File created:', OUTPUT_FILE + '_LogReg.csv')

def main():    
    configuration()
    readData()        
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