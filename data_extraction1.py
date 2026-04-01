print ("\033[2J")
import csv, os, pandas as pd
#*** SETTINGS ***
OUTPUT_FILE = 'DATA_FULL.csv'
CLASS_SIZE_0 = 1
CLASS_SIZE_1 = 1
CLASS_SIZE_2 = 1
def headerExtraction():
	print('\n*** HEADER EXTRACTION ***')
	data = pd.read_csv('class0-1.par', sep='\t', decimal='.', header=None, skiprows=18, usecols=[0])
	data = data.T
	data.to_csv('Header.csv', sep=';', index = None, header=None)
	print('Header.csv file created')
def dataExtraction(CLASS, CLASS_SIZE):
	for i in range(1, CLASS_SIZE+1):
		fileName = CLASS + str(i) + '.par'
		data = pd.read_csv(fileName, sep = '\t', decimal='.', header=None, skiprows=18, usecols=[1])
		data = data.T
		data.to_csv(CLASS + str(i)+ '.csv', sep=';', index = None, header=None, float_format='%g')
		print(CLASS + str(i) + '.csv' + 'file created')
def dodajKlase(writer, CLASS, CLASS_SIZE):
	for i in range(1, CLASS_SIZE+1):
		fileName = CLASS + str(i) + '.csv'
		with open(fileName, 'rt') as readingFile:
			read = csv.reader(readingFile)
			for row in read:
				writer.writerow(row)
		os.remove(fileName)
		print(CLASS + str(i) + '.csv' + 'file deleted')
def dataMerging():
	print('\n*** MERGING DATA AND TEMPORARY FILES DELETING ***')
	with open(OUTPUT_FILE, 'w', newline= '') as recordFile:
		writer = csv.writer(recordFile)
		with open('Header.csv', 'rt') as readingFile:
			read = csv.reader(readingFile)
			for row in read:
				writer.writerow(row)
		os.remove('Header.csv')
		print('Header.csv file deleted')
		print('*** CLASS  0 ***')
		dodajKlase(writer, 'class0-', CLASS_SIZE_0)
		print ('*** CLASS 1 ***')
		dodajKlase(writer, 'class1-', CLASS_SIZE_1)
		print ('*** CLASS 2 ***')
		dodajKlase(writer, 'class2-', CLASS_SIZE_2)
def addClassLabels():
	dane = pd.read_csv(OUTPUT_FILE, sep=';', decimal='.')
	dane.loc[0:CLASS_SIZE_0-1, 'F'] = 0
	dane.loc[CLASS_SIZE_0:CLASS_SIZE_0+CLASS_SIZE_1-1, 'F'] = 1
	dane.loc[CLASS_SIZE_0+CLASS_SIZE_1:CLASS_SIZE_0+CLASS_SIZE_1+CLASS_SIZE_2-1, 'F'] = 2
	dane.to_csv(OUTPUT_FILE, sep=';', index = None, float_format='%g')
def main():
	headerExtraction()
	print('\n*** DATA EXTRACTION ***\n*** TEMPORARY FILES CREATION – CLASS 0 ***')
	dataExtraction('class0-', CLASS_SIZE_0)
	print('*** TEMPORARY FILES CREATION = CLASS 1 ***')
	dataExtraction('class1-', CLASS_SIZE_1)
	print('*** TEMPORARY FILES CREATION – CLASS 2 ***')
	dataExtraction('class2-', CLASS_SIZE_2)
	dataMerging()
	addClassLabels()
	print('\n', OUTPUT_FILE, 'file created')
if __name__== "__main__":
    main()
