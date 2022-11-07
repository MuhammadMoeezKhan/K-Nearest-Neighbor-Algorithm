#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:21:57 2022
@author: moeezkhan
"""

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial import distance
import sys as system
import timeit

# =============================================================================
# BaseEstimator is the base class for making an “estimator” – the generic term scikit-learn uses for classifiers and other types of models (some of which we’ll make later on).
# ClassifierMixin gives us more functionality specific to classifiers.
# =============================================================================

import numpy as np
from sklearn import model_selection
from sklearn.metrics import make_scorer
# =============================================================================
# to help perform k-fold-validation
# =============================================================================

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------------------------------------------------------------
# GIVEN: For use in all testing for the purpose of grading
def testMain():
    '''
    This function runs all the tests we'll use for grading. Please don't change it!
    When certain parts need to be graded, uncomment those parts only.
    Please keep all the other parts commented out for grading.
    '''
    pass

    #print("========== testAlwaysOneClassifier ==========")
    #testAlwaysOneClassifier()

    #print("========== testFindNearest() ==========")
    #testFindNearest()

    #print("========== testOneNNClassifier() ==========")
    #testOneNNClassifier()

    #print("========== testCVManual(OneNNClassifier(), 5) ==========")
    #testCVManual(OneNNClassifier(), 5)

    #print("========== testCVBuiltIn(OneNNClassifier(), 5) ==========")
    #testCVBuiltIn(OneNNClassifier(), 5)

    #print("========== compareFolds() ==========")
    #compareFolds()

    #print("========== testStandardize() ==========")
    #testStandardize()

    #print("========== testNormalize() ==========")
    #testNormalize()

    #print("========== comparePreprocessing() ==========")
    #comparePreprocessing()

    print("========== visualization() ==========")
    visualization()

    print("========== testKNN() ==========")
    testKNN()

    print("========== paramSearchPlot() ==========")
    paramSearchPlot()

    print("========== paramSearchPlotBuiltIn() ==========")
    paramSearchPlotBuiltIn()
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Reading in the data" step
def readData(numRows=None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids",
                 "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows=numRows)

    # Step#3
    # Need to mix this up before doing CV
    # Helps randomise row selection, for better prediction
    wineDF = wineDF.sample(frac=1, random_state=50).reset_index(drop=True)

    return wineDF, inputCols, outputCol
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Testing AlwaysOneClassifier" step
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0

    return accuracy
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def operationsOnDataFrames():
    d = {'x': pd.Series([1, 2], index=['a', 'b']),
         'y': pd.Series([10, 11], index=['a', 'b']),
         'z': pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')

    cols = ['x', 'z']

    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')

    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def testStandardize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    standardize(df, colsToStandardize)
    print("After standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, colsToStandardize].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, colsToStandardize].std(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Normalization on a DataFrame" step
def testNormalize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    normalize(df, colsToStandardize)
    print("After normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, colsToStandardize].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, colsToStandardize].min(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------



#==============================================================================
# Step#4
# create algorithim class for simple testing set 
class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    def __innit__(self):
        pass

    #fit the testing model to find the explain the data and estimate its structure
    #since testing model very simple, does not need to fit the training set
    def fit(self, inputDF, outputSeries):
        return self
    
    #predict the answer(target attribute) using the testing set and the testing model
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return 1
        
        #if df, return series of all df rows with "1"
        else:
            numElements = testInput.shape[0]
            return pd.Series(np.ones(numElements), index=testInput.index, dtype="int64")
            
         
            
# Step#5
# seperate the testing and training sets
def testAlwaysOneClassifier():
    df, inputCols, outputCol = readData()
    
    numCols = df.shape[1]
    
    testInputDF = df.iloc[ :10, 1:numCols]
    testOutputSeries = df.iloc[ :10, 0]
    
    trainInputDF = df.iloc[ 10:, 1:numCols]
    trainOutputSeries = df.iloc[ 10: , 0]
    
    #print("testInputDF:", testInputDF, sep='\n', end='\n\n') 
    #print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    #print("trainInputDF:", trainInputDF, sep='\n', end='\n\n') 
    #print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')
    
    test_instance = AlwaysOneClassifier() 
    
    test_instance.fit(trainInputDF, trainOutputSeries)
    print("\n---------- Test one example")
    print("Correct answer: " + str(testOutputSeries[0]))
    
    firsrTestRow = testInputDF.iloc[0, :] 
    print("Predicted answer: " + str(test_instance.predict(firsrTestRow)))
    
    print("\n\n---------- Test the entire test set")
    print("Correct answers: ")
    print(testOutputSeries)
    
    print("\nPredicted answers: ")
    predictedResults = test_instance.predict(testInputDF)
    print(predictedResults)
    
    print("\n") 
    print("Accuracy: " + str(accuracyOfActualVsPredicted(testOutputSeries, predictedResults)))
    
    
#============================================================================== 
# Step#7
# find the nearest Series(row) using iteration
def findNearestLoop(df, testRow):
    minEucDifference = system.maxsize
    minEucDifferenceIdx = 0
    
    for index, row in df.iterrows():
        
        currEucDifference = distance.euclidean(row, testRow)
        
        if currEucDifference < minEucDifference:
            minEucDifference = currEucDifference
            minEucDifferenceIdx = index
        
    return minEucDifferenceIdx

    #difference approach:
    #for rowID in range(df.shape[0])
    #row = df.iloc[rowID, :]

#print("\nFinding The Closest Row:")
# test Step#7
df, inputCols, outputCol = readData() 
#print( "Closest DF Row Label: "  + str(findNearestLoop(df.iloc[100:107, :], df.iloc[90, :])))
    


# Step#8
# find the nearest Series(row) using high-order functions 
def findNearestHOF(df, testRow):
    eucDistance = df.apply( lambda row : distance.euclidean(row, testRow), axis = 1 )
    return eucDistance.idxmin()
    
# test Step#8
#print("Closest DF Row Label: "  + str(findNearestHOF(df.iloc[100:107, :], df.iloc[90, :])))



# Step#9
def testFindNearest():
    print("\n\nTime Analysis: ")
    startTime1 = timeit.default_timer()
    print("================ findNearestLoop")
    for i in range(100):
        findNearestLoop(df.iloc[100:107, :], df.iloc[90, :])
    print( str(timeit.default_timer() - startTime1) + " seconds" )
    
    
    startTime2 = timeit.default_timer()
    print("================ findNearestHOF")
    for i in range(100):
        findNearestHOF(df.iloc[100:107, :], df.iloc[90, :])
    print( str(timeit.default_timer() - startTime2) + " seconds" )
    
    
#==============================================================================
# Alterting Testing Sets   
#change: fit according to a linear regression mode, figure out line of best fit
#change: preduct according to a linear regression mode, use line of best fit to make predictions
    #or
#change: fit according to a neural network mode,figure out the best parameters. 
#change: preduct according to a neural network mode, use the fit test model's (the parameters) to make predictions.


# Step#10
# change: fit simply “remembers” the training set by storing it in fields.
# change: predict does the same thing using the new test model
class OneNNClassifier(BaseEstimator, ClassifierMixin):
    
    def __innit__(self):
        self.inputDF = None
        self.outputSeries = None

    #no fitting algo required for 1NN
    def fit(self, inputDF, outputSeries): 
        self.inputDF= inputDF
        self.outputSeries = outputSeries
    
    #find the nearest point for one row
    def __predictOne(self, testInput):
        return self.outputSeries[findNearestHOF(self.inputDF, testInput)]
    
    #find the nearest point for two cases - series and df input
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.__predictOne(testInput)
        else:
            predictedSeries = testInput.apply( lambda testRow : self.__predictOne(testRow), axis = 1 )
            return predictedSeries



# Step#11 and Step#12
def testOneNNClassifier():
    df, inputCols, outputCol = readData()
    
    numCols = df.shape[1]
    
    testInputDF = df.iloc[ :10, 1:numCols]
    testOutputSeries = df.iloc[ :10, 0]
    
    trainInputDF = df.iloc[ 10:, 1:numCols]
    trainOutputSeries = df.iloc[ 10: , 0]
    
    #print("testInputDF:", testInputDF, sep='\n', end='\n\n') 
    #print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    #print("trainInputDF:", trainInputDF, sep='\n', end='\n\n') 
    #print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')
    
    test_instance = OneNNClassifier() 
    
    #Step#11
    test_instance.fit(trainInputDF, trainOutputSeries)
    print("\n\n---------- OneNNClassifier Tests ----------")
    print("\n---------- Test one example")
    print("Correct answer: " + str(testOutputSeries[0]))
    
    secondTestRow = testInputDF.iloc[2, :] 
    print("Predicted answer: " + str(test_instance.predict(secondTestRow)))
    
    #Step#12
    print("\n\n---------- Test the entire test set")
    print("Correct answers: ")
    print(testOutputSeries)
    
    print("\nPredicted answers: ")
    predictedResults = test_instance.predict(testInputDF)
    print(predictedResults)
    
    print("\n") 
    print("Accuracy: " + str(accuracyOfActualVsPredicted(testOutputSeries, predictedResults)))
  
    
#==============================================================================
#Step#16 --> k-fold cross validation
#k = #folds and verbose = boolean flag that tells whether details should be printed
def cross_val_score_manual(model, inputDF, outputSeries, k, verbose):
    
    #size of each fold equals the number of rows divided by #of folds
    #if not even distribution, reminder rows get put in ...
    numRows = inputDF.shape[0]
    foldSize = numRows / k
    
    results = []
    for iteration in range(k):                                      #make k-folds
        
        testStart = int(iteration * foldSize)
        testEnd = int((iteration + 1) * foldSize)


        testInputDF = inputDF.iloc[testStart : testEnd , :] 
        testOutputSeries = outputSeries.loc[testInputDF.index]

        trainingP1 = inputDF.iloc[0 : testStart , :]
        trainingP2 = inputDF.loc[testEnd : , :]

        trainInputDF = pd.concat([trainingP1, trainingP2])          #concatenate seperated subsets
        trainOutputSeries = outputSeries.loc[trainInputDF.index]
    

        if (verbose): #print data structure info
            print("================================")
            print("Iteration:", iteration)
            print("Train input:\n", list(trainInputDF.index)) 
            print("Train output:\n", list(trainOutputSeries.index)) 
            print("Test input:\n", testInputDF.index)
            print("Test output:\n", testOutputSeries.index)
            print("================================") 
       
        model.fit(trainInputDF, trainOutputSeries)                  #fit on the training set
        predictedResults = model.predict(testInputDF)               #predict on the testing set
        results.append((accuracyOfActualVsPredicted(testOutputSeries, predictedResults)))
        
    return results


def testCVManual(model, k): 
    df, inputCols, outputCol = readData()
    inputDF = df.loc[ : , inputCols]
    outputSeries = df.loc[ : , outputCol]
    
    accuracies = cross_val_score_manual(model, inputDF, outputSeries, k, True)
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))
    
    

#Step#17 --> Use Python libraries to perform k-fold 
def testCVBuiltIn(model,k):
    df,inputCols,outputCol = readData()
    
    inputDF = df.loc[:,inputCols]  
    outputSeries = df.loc[:,outputCol]
    
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries,cv=k, scoring=scorer)
    print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    
    
#Step#18 --> Comparing Results for different numbers of folds
def compareFolds():
    testCVBuiltIn(OneNNClassifier(),3)   #prints the accuracies
    testCVBuiltIn(OneNNClassifier(),10)  #prints the accuracies
    
    
    
#==============================================================================
#Step#20 --> Standardization on a DataFrame
def standardize(df, cols):
    df.loc[:, cols] = (df.loc[:, cols] - df.loc[:,cols].mean()) / df.loc[:,cols].std()


#Step#21 --> Normalization on a Dataframe
def normalize(df, cols):
    df.loc[:,cols] = (df.loc[:,cols]- df.loc[:,cols].min()) / (df.loc[:,cols].max() - df.loc[:,cols].min())


#Step#22 --> Comparison of Preprocessing Approaches
def comparePreprocessing():
    df,inputCols,outputCol = readData()
    
    copy_for_normalization = df.copy()
    copy_for_standardization = df.copy()
    
    normalize(copy_for_normalization, inputCols)
    standardize(copy_for_standardization , inputCols)
    
    #---------------------------------
    print('\nOriginal Dataset')
    inputDF = df.loc[:,inputCols]  
    outputSeries = df.loc[:,outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF, outputSeries,cv=10, scoring=scorer)
    #print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    
    #---------------------------------------------
    print('\nNormalized Dataset')
    normalizedInputDF =  copy_for_normalization.loc[:,inputCols]  
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(OneNNClassifier(), normalizedInputDF, outputSeries,cv=10, scoring=scorer)
    #print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    
    #---------------------------------------------
    print('\nStandardized Dataset')
    standarizedInputDF =  copy_for_standardization.loc[:,inputCols]  
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(OneNNClassifier(), standarizedInputDF, outputSeries,cv=10, scoring=scorer)
    #print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))


"""
Responses:
    
    a) original dataset mean accuracy : 0.7477
       normalized dataset mean accuracy: 0.9496
       standardized dataset mean accuracy: 0.9552
       
       Normalization and Standardization drastically improve accuracy since they do not allow erranous values
       from massively disrupting the output. Each variable is given equal weights/importance, 
       preventing the effectiveness of the model from being influenced by a single variable simply because it has larger values.
     
        
    b) 1NN --> 96.1%
       Z-Transformed Data: The standardization procedure known as Z transformation facilitates comparison of results from various z-distributions.
       These transformations combine distinct distributions into a standardized distribution using the distribution's mean and standard deviation, 
       enabling the comparison of measurements with different properties. This is the definition of standardization's formula.


    c) 'leave-one-out' is also a method of testing with multiple training and testing set. It gives a higher accuracy
        because it uses more data for training compared to the k-fold cross validation. This is the reason why the rsults
        reported in wine.names have higher accuracy.

"""


#==============================================================================
#Step#24 --> Introduction to Visualization
def visualization():
    fullDF, inputCols, outputCol = readData()
    standardize(fullDF, inputCols)
    
    sns.displot(fullDF.loc[:, 'Malic Acid'])
    print(fullDF.loc[:, 'Malic Acid'].skew())
    sns.displot(fullDF.loc[:, 'Alcohol'])
    print(fullDF.loc[:, 'Alcohol'].skew())
    
    #a) The skew measure for Alcohol is: -0.051482331077132064 
    #From the plot and the measured value, we can see it is negatively skewed.
    
    sns.jointplot(x='Malic Acid', y='Alcohol', data=fullDF.loc[:, ['Malic Acid', 'Alcohol']], kind='kde')  # combines the distribution estimates of two attributes
    sns.displot(fullDF.loc[:, 'Ash'])
    #print(fullDF.loc[:, 'Ash'].mode()[0])  
    #print(fullDF.loc[:, 'Magnesium'].mode()[0])
    sns.displot(fullDF.loc[:, 'Magnesium'])
    sns.jointplot(x='Ash', y='Magnesium', data=fullDF.loc[:, ['Ash', 'Magnesium']], kind='kde')  # combines the distribution estimates of two attributes '''
    
    #b) The two most likey values for Ash and Magnesium are (-0.315, -0.822) 
    #(We used mode, but this could be seen from graph as well).
    
    
    sns.pairplot(fullDF, hue=outputCol)
    
    #c) 1 
    #d) Accuracy will drop marginally because points on the graph are clustered nicely: All classes are clustered seperately and there is a good spread of the attribute points.
    #e) Accuracy has dropped alot because classes are not clustered seperately. We can see that all classes clustered in the same region, thus removing these attributes causes saturated accuracy loss.

    plt.show()
    

def testSubsets():
    fullDF, inputCols, outputCol = readData()
    standardize(fullDF, inputCols)
    # Checking accuracy for Part d
    inputDF = fullDF.loc[:,['Diluted', 'Proline']]  
    outputSeries = fullDF.loc[:,outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF, outputSeries,cv=3, scoring=scorer)
    print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    
    #The accuracy after dropping most input columns, keeping only Diluted and Proline,
    #has dropped marginally; not very much. The average acuuracy is 0.8426553672316385. 
    
    # Checking accuracy for part e
    inputDF2 = fullDF.loc[:,['Nonflavanoid Phenols', 'Ash']]  
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF2, outputSeries,cv=3, scoring=scorer)
    print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    
    #The accuracy after dropping most input columns, keeping only 'Nonflavanoid Phenols', 'Ash',
    #has dropped alot. The average acuuracy is 0.5447269303201506.
        
#-------------------------------------------------------------------------------------
#Step#26 --> KNNClassifier    
def findNearestHOF1(df,testRow,k):
    s = df.apply(lambda row: distance.euclidean(row,testRow),axis= 1)
    return s.nsmallest(k)
 
    
class kNNClassifier(BaseEstimator,ClassifierMixin):
    
    def __init__(self, inputDF, outputSeries,k=1):
        self.inputDF = None
        self.outputSeries = None
        self.k = k  
        
    
    def fit(self, inputDF, outputSeries):
        self.inputDF = inputDF
        self.outputSeries = outputSeries
    
    def __predOfKNearest(self,testInput):
        kNeighbors = findNearestHOF1(self.inputDF, testInput,self.k)
        kNeighborsClass = self.outputSeries[kNeighbors.index] 
        majorityVote = kNeighborsClass.mode()[0]
        return majorityVote 
    
    def predict(self,testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self. __predOfKNearest(testInput)    # since we want to have same output of the closest row, we pass its label to outtput series, which has the classification number.
        else:
            predictedSeries = testInput.apply(lambda row: self.__predOfKNearest(row),axis=1) # This line get the series of all output classifications of testInput dataframe.
            return predictedSeries
        
    
#------------------------------------------------------------------------------------
#Step#27 --> Cross validation on various kNN models
def testKNN():
    df,inputCols,outputCol = readData()
    copy_for_normalize = df.copy()
    copy_for_standardize = df.copy()
    normalize(copy_for_normalize, inputCols)
    standardize(copy_for_standardize , inputCols)
    
#-----------------------------------------------    
    print(' Unaltered dataset, 1NN, accuracy: ')
    inputDF = df.loc[:,inputCols]  
    outputSeries = df.loc[:,outputCol]
    model = kNNClassifier(None,None,k=1)
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries,cv=10, scoring=scorer)
    #print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
#------------------------------------------------------

    print(' standardized Dataset, 1NN, accuracy: ')
    standarizedInputDF =  copy_for_standardize.loc[:,inputCols]  
    model2 = kNNClassifier(None,None,k=1)
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(model2, standarizedInputDF, outputSeries,cv=10, scoring=scorer)
    #print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
#------------------------------------------------------

    print(' standardized Dataset, 8NN, accuracy: ')
    standarizedInputDF =  copy_for_standardize.loc[:,inputCols]  
    model3 = kNNClassifier(None,None,k=8)
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score(model3, standarizedInputDF, outputSeries,cv=10, scoring=scorer)
    #print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))    
    
    #The 8NN model has higher accuracy because we are using more information for classification, whereas 
    #in 1NN we just classify to the 1 nearest neighbor.
        
#---------------------------------------------------------------------------------------------
#Step#28 --> Tuning the model
def tuning(k): # used this function in paramSearchPlot()
    df,inputCols,outputCol = readData()
    standardize(df , inputCols)
    outputSeries = df.loc[:,outputCol]
    standarizedInputDF =  df.loc[:,inputCols] 
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    accuracies = model_selection.cross_val_score( kNNClassifier(None,None,k), standarizedInputDF, outputSeries,cv=10, scoring=scorer)
    averageAccuracy= np.mean(accuracies)
    return averageAccuracy
    
    

def paramSearchPlot():
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    accuracies = neighborList.map(lambda value: tuning(value))   
    print ( accuracies)      
    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    optimalKValue= neighborList.loc[accuracies.idxmax()]
    print(optimalKValue)
    
#-----------------------------------------------------------------------------------------
#Step#29 --> kNNClassifier
def tuning2(k): # used this function in paramSearchPlotBuiltIn()
    df,inputCols,outputCol = readData()
    standardize(df , inputCols)
    outputSeries = df.loc[:,outputCol]
    stdInputDF =  df.loc[:,inputCols] 
    alg = KNeighborsClassifier(n_neighbors = k)
    cvScores = model_selection.cross_val_score(alg, stdInputDF, outputSeries, cv=10, scoring='accuracy')
    averageAccuracy = np.mean(cvScores)
    return averageAccuracy

def paramSearchPlotBuiltIn():
    df,inputCols,outputCol = readData()
    standardize(df , inputCols)
    outputSeries = df.loc[:,outputCol]
    stdInputDF =  df.loc[:,inputCols]
    alg = KNeighborsClassifier(n_neighbors = 8)
    cvScores = model_selection.cross_val_score(alg, stdInputDF, outputSeries, cv=10, scoring='accuracy')
    print(np.mean(cvScores))  
    
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    accuracies = neighborList.map(lambda value: tuning2(value))   
    #print ( accuracies)      
    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    optimalKValue= neighborList.loc[accuracies.idxmax()]
    print(optimalKValue)