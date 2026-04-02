# This script extracts texture features from MaZda *.PAR files
# and converts them into a single CSV dataset for machine learning.
# Note: All *.PAR files must be located in the working directory.

print ("\033[2J")  # Clear console output (optional)

import csv, os, pandas as pd

# *** SETTINGS ***
OUTPUT_FILE = 'DATA_FULL.csv'  # Final output dataset
CLASS_SIZE_0 = 1               # Number of samples in class 0
CLASS_SIZE_1 = 1               # Number of samples in class 1
CLASS_SIZE_2 = 1               # Number of samples in class 2


def headerExtraction():
    print('\n*** HEADER EXTRACTION ***')
    
    # Read feature names (first column) from a sample PAR file
    # Skip metadata rows and extract only feature labels
    data = pd.read_csv('class0-1.par', sep='\t', decimal='.', header=None, skiprows=18, usecols=[0])
    
    # Transpose to make features as columns
    data = data.T
    
    # Save header to temporary CSV file
    data.to_csv('Header.csv', sep=';', index = None, header=None)
    
    print('Header.csv file created')


def dataExtraction(CLASS, CLASS_SIZE):
    # Extract feature values from PAR files for a given class
    for i in range(1, CLASS_SIZE+1):
        
        # Construct PAR file name (e.g., class0-1.par)
        fileName = CLASS + str(i) + '.par'
        
        # Read feature values (second column), skipping metadata
        data = pd.read_csv(fileName, sep = '\t', decimal='.', header=None, skiprows=18, usecols=[1])
        
        # Transpose so that each row corresponds to one sample
        data = data.T
        
        # Save extracted features to temporary CSV file
        data.to_csv(CLASS + str(i)+ '.csv', sep=';', index = None, header=None, float_format='%g')
        
        print(CLASS + str(i) + '.csv' + 'file created')


def dodajKlase(writer, CLASS, CLASS_SIZE):
    # Append class data from temporary CSV files into the final dataset
    for i in range(1, CLASS_SIZE+1):
        
        fileName = CLASS + str(i) + '.csv'
        
        # Read temporary file and write its content to the final CSV
        with open(fileName, 'rt') as readingFile:
            read = csv.reader(readingFile)
            for row in read:
                writer.writerow(row)
        
        # Remove temporary file after merging
        os.remove(fileName)
        
        print(CLASS + str(i) + '.csv' + 'file deleted')


def dataMerging():
    # Merge header and all class data into a single CSV file
    print('\n*** MERGING DATA AND TEMPORARY FILES DELETING ***')
    
    with open(OUTPUT_FILE, 'w', newline= '') as recordFile:
        writer = csv.writer(recordFile)
        
        # Write header to the output file
        with open('Header.csv', 'rt') as readingFile:
            read = csv.reader(readingFile)
            for row in read:
                writer.writerow(row)
        
        # Delete temporary header file
        os.remove('Header.csv')
        
        print('Header.csv file deleted')
        
        # Merge data for each class
        print('*** CLASS  0 ***')
        dodajKlase(writer, 'class0-', CLASS_SIZE_0)
        
        print ('*** CLASS 1 ***')
        dodajKlase(writer, 'class1-', CLASS_SIZE_1)
        
        print ('*** CLASS 2 ***')
        dodajKlase(writer, 'class2-', CLASS_SIZE_2)


def addClassLabels():
    # Add class labels (target variable F) to the dataset
    dane = pd.read_csv(OUTPUT_FILE, sep=';', decimal='.')
    
    # Assign labels based on row indices
    dane.loc[0:CLASS_SIZE_0-1, 'F'] = 0
    dane.loc[CLASS_SIZE_0:CLASS_SIZE_0+CLASS_SIZE_1-1, 'F'] = 1
    dane.loc[CLASS_SIZE_0+CLASS_SIZE_1:CLASS_SIZE_0+CLASS_SIZE_1+CLASS_SIZE_2-1, 'F'] = 2
    
    # Save updated dataset
    dane.to_csv(OUTPUT_FILE, sep=';', index = None, float_format='%g')


def main():
    # Step 1: Extract header (feature names)
    headerExtraction()
    
    # Step 2: Extract data for each class
    print('\n*** DATA EXTRACTION ***\n*** TEMPORARY FILES CREATION – CLASS 0 ***')
    dataExtraction('class0-', CLASS_SIZE_0)
    
    print('*** TEMPORARY FILES CREATION = CLASS 1 ***')
    dataExtraction('class1-', CLASS_SIZE_1)
    
    print('*** TEMPORARY FILES CREATION – CLASS 2 ***')
    dataExtraction('class2-', CLASS_SIZE_2)
    
    # Step 3: Merge all data
    dataMerging()
    
    # Step 4: Add class labels
    addClassLabels()
    
    print('\n', OUTPUT_FILE, 'file created')


if __name__== "__main__":
    main()