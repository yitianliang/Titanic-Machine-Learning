# Data Analysis
import pandas as pd
import numpy as np
import random as rnd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Missing value visualization
import missingno as msno

def printGeneralStatistics( data ):
 print( data.describe() )					# Statistics
 print( data.describe(include=['O']) )		# Distribution

def printGeneralInformation( data ):
 print( data.columns.values )				# Feature names
 print( data.info )							# Data Types

def setAgeBoundaries (  ):
 for dataset in combine:
     dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
     dataset.loc[(dataset['Age'] > 5 ) & (dataset['Age'] <= 16), 'Age'] = 1
     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 2
     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 3
     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 4
     dataset.loc[ dataset['Age'] > 64, 'Age'] = 5
     #print(train_df['Age'].value_counts()) #check count of age.

def normalizeFamily( ):
 for dataset in combine:
   dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

def pivotingData ( data, entry1, entry2, groupBy, sortBy ):
 return data[[ entry1 , entry2 ]].groupby([groupBy], as_index=False).mean().sort_values(by=sortBy, ascending=False)

def printPivotedData( data ):
 #only categorical values
 print ( pivotingData ( data, 'Pclass','Survived','Pclass','Survived' ) )
 print ( pivotingData ( data, 'Sex','Survived','Sex','Survived' ) )
 print ( pivotingData ( data, 'SibSp','Survived','SibSp','Survived' ) )
 print ( pivotingData ( data, 'Parch','Survived','Parch','Survived' ) )


def visualizeNumericalCorrelation( data, feature1, feature2 ):
 g = sns.FacetGrid(data, col=feature2)
 g.map(plt.hist, feature1, bins=20)
 g.savefig("output.png")

def visualizeScatter( data, feature1, feature2 ):
 g = sns.FacetGrid(data, col='Survived',  hue="Survived")
 g = (g.map(plt.scatter, feature1, feature2, edgecolor="w")
      .add_legend())
 g.savefig("output2.png")

def visualizeSurvivedCorrelation(  feature1, feature2 ):
 grid = sns.FacetGrid(train_df, col='Survived', row=feature2, size=2.2, aspect=1.6)
 grid.map(plt.hist, feature1, alpha=.5, bins=20)
 grid.add_legend()
 grid.savefig("output3.png")

def classifyWithLogisticRegression ( trainingData, results, testData ):
 clf_logreg = LogisticRegression()
 clf_logreg.fit(trainingData, results)
 Y_pred = clf_logreg.predict(testData)
 acc_log = round(clf_logreg.score(trainingData, results) * 100, 2)

 return acc_log

def classifyWithDecisionTree ( trainingData, results, testData ):
 clf_tree = tree.DecisionTreeClassifier()
 clf_tree.fit(trainingData, results)
 Y_pred = clf_tree.predict(testData)
 acc_decision_tree = round(clf_tree.score(trainingData, results) * 100, 2)

 return acc_decision_tree

def classifyWithSVM ( trainingData, results, testData ):
 clf_svm = SVC()
 clf_svm.fit(trainingData,results)
 Y_pred = clf_svm.predict(testData)
 acc_decision_tree = round(clf_svm.score(trainingData, results) * 100, 2)
 return acc_decision_tree

def classifyWithPerceptron ( trainingData, results, testData ):
 clf_perceptron = Perceptron()
 clf_perceptron.fit(trainingData,results)
 Y_pred = clf_perceptron.predict(testData)
 acc_perceptron = round(clf_perceptron.score(trainingData, results) * 100, 2)

 return acc_perceptron

def classifyWithKNeighbors ( trainingData, results, testData ):
 clf_KNN = KNeighborsClassifier()
 clf_KNN.fit(trainingData,results)
 Y_pred = clf_KNN.predict(testData)
 acc_knn = round(clf_KNN.score(trainingData, results) * 100, 2)

 return acc_knn

def classifyWithGaussianNaiveBayes ( trainingData, results, testData ):
 clf_GaussianNB = GaussianNB()
 clf_GaussianNB.fit(trainingData,results)
 Y_pred = clf_GaussianNB .predict(testData)
 acc_gaussian = round( clf_GaussianNB .score(trainingData, results) * 100, 2)

 return acc_gaussian

def classifyWithStochasticGradientDescent ( trainingData, results, testData ):
 sgd = SGDClassifier(max_iter=5, tol=None)
 sgd.fit(trainingData, results)
 Y_pred = sgd.predict(testData)
 sgd.score(trainingData, results)
 acc_sgd = round(sgd.score(trainingData, results) * 100, 2)

 return acc_sgd

def classifyWithLinearSVC ( trainingData, results, testData ):
 linear_svc = LinearSVC()
 linear_svc.fit(trainingData, results)
 Y_pred = linear_svc.predict(testData)
 acc_linear_svc = round(linear_svc.score(trainingData, results) * 100, 2)

 return acc_linear_svc

def classifyWithRandomForest ( trainingData, results, testData ):
 random_forest = RandomForestClassifier(n_estimators=100)
 random_forest.fit(trainingData, results)
 Y_prediction=random_forest.predict(testData)
 random_forest.score(trainingData, results)
 acc_random_forest = round(random_forest.score(trainingData, results) * 100, 2)

 return acc_random_forest


def normalizeSex ( ):
 for dataset in combine:
   dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

def normalizeAges ( ):
 guess_ages = np.zeros((2,3))
 for dataset in combine:
   for i in range(0, 2):
     for j in range(0, 3):
       guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
       age_guess = guess_df.median()
       guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

   for i in range(0, 2):
     for j in range(0, 3):
       dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

   dataset['Age'] = dataset['Age'].astype(int)

def normalizeEmbarked( ):
 freq_port = train_df.Embarked.dropna().mode()[0]

 for dataset in combine:
   dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

 for dataset in combine:
   dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

def normalizeFare():

 for dataset in combine:
   dataset.loc[(dataset['Fare'] < 9), 'Fare'] = 0
   dataset.loc[(dataset['Fare'] >= 9) & (dataset ['Fare'] < 12), 'Fare'] = 1
   dataset.loc[(dataset['Fare'] >= 12) & (dataset ['Fare'] < 15), 'Fare'] = 2
   dataset.loc[(dataset['Fare'] >= 15) & (dataset ['Fare'] < 20), 'Fare'] = 3
   dataset.loc[(dataset['Fare'] >= 20) & (dataset ['Fare'] < 30), 'Fare'] = 4
   dataset.loc[(dataset['Fare'] >= 30) & (dataset ['Fare'] < 55), 'Fare'] = 5
   dataset.loc[(dataset['Fare'] >= 55) & (dataset ['Fare'] < 95), 'Fare'] = 6
   dataset.loc[(dataset['Fare'] >= 95),'Fare'] = 7
   dataset['Fare'] = dataset['Fare'].astype(int)



def normalizeAgeClass( ):
 for dataset in combine:
   dataset['Age*Class*Fare'] = dataset.Age * dataset.Pclass * dataset.Fare
   dataset['Age*Class'] = dataset.Age * dataset.Pclass
   dataset['Age*Fare'] = dataset.Age * dataset.Fare


def normalizeData( ):
 normalizeSex ( )
 normalizeAges( )
 setAgeBoundaries( )
 normalizeFamily( )
 normalizeEmbarked( )
 normalizeFare( )
 normalizeAgeClass( )


def getFareClass(data,cat):
 return data.loc[data['Fare'] == cat]

def ageMissProcession():
    train_df = pd.read_csv("train.csv")  # reload original data

    age_median_psex = train_df.groupby(
        ["Pclass", "Sex"]).Age.median()  # Grouping the median age of men and women in different classes

    train_df.set_index(["Pclass", "Sex"], inplace=True)  # Setting Pclass, make sex as index, Inplace=True means to modify directly on the original data titanic_df

    train_df.Age.fillna(age_median_psex, inplace=True)  # Fill in the missing values with fillna, populate based on index values

    train_df.reset_index(inplace=True)  # Reset index, cancel Sex, change Pclass to index

    (train_df.Age.describe())  # View descriptive statistics for the Age column

def evaluateSubmission():
    print ('A')

def main ( ):
 global train_df
 global test_df
 global combine

 # Training and Testing Data
 train_df = pd.read_csv('train.csv')
 test_df = pd.read_csv('test.csv')
 full_data = [train_df, test_df]

 #train_df.info() #check the input data
 msno.matrix(train_df, figsize=(12, 5))  # 可视化查询缺失值

 #train_df.isnull().sum()  # 查寻整个数据集缺失值的个数

 #data processing
 ageMissProcession()

 (train_df.Embarked.value_counts()) # count Embarked value

 train_df.fillna({"Embarked": "S"}, inplace=True)  # Fill in the missing values of the Embarked column with "S"

 train_df[train_df.Embarked.isnull()].Embarked # View missing value fill effects


 # Drop Useless Features
 train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
 test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
 visualizeNumericalCorrelation(train_df,'Fare','Survived')
 visualizeNumericalCorrelation(train_df,'Age','Survived')

 # output picture
 # visualizeScatter(train_df,'Pclass','Age')
 # visualizeSurvivedCorrelation('Age','Survived')

 test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
 train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
 train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

 # Normalize both data sets
 combine = [train_df, test_df]
 normalizeData( )
 train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
 test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
 train_df = train_df.drop(['FareBand'], axis=1)
 print(train_df['Age'].value_counts())

 print  ("------------------Age--------------------")
 print (pivotingData( train_df, 'Age', 'Survived', 'Age', 'Survived' ))
 print  ("------------------Fare--------------------")
 print (pivotingData( train_df, 'Fare', 'Survived', 'Fare', 'Survived' ))

 #visualizeNumericalCorrelation(getFareClass(train_df,0),'Age','Survived')

 combine = [train_df, test_df]
 printGeneralInformation(train_df)

 # Setting up data
 X_train = train_df.drop(["Survived","PassengerId","Fare","Age","Pclass"], axis=1)
 Y_train = train_df["Survived"]
 X_test  = test_df.drop(["PassengerId","Fare","Age","Pclass"], axis=1).copy()
 X_train.shape, Y_train.shape, X_test.shape

 print (X_train)

 # checking machine learning odd score

 acc_linear_svc=classifyWithLinearSVC(X_train, Y_train, X_test)
 acc_knn=classifyWithKNeighbors(X_train, Y_train, X_test)
 acc_log=classifyWithLogisticRegression(X_train, Y_train, X_test)
 random_forest=classifyWithRandomForest(X_train, Y_train, X_test)


 acc_gaussian=classifyWithGaussianNaiveBayes(X_train, Y_train, X_test)
 acc_perceptron=classifyWithPerceptron(X_train, Y_train, X_test)
 acc_sgd=classifyWithStochasticGradientDescent(X_train, Y_train, X_test)
 acc_decision_tree=classifyWithDecisionTree(X_train, Y_train, X_test)


 # modal score checking
 results = pd.DataFrame({
     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
               'Random Forest', 'Naive Bayes', 'Perceptron',
               'Stochastic Gradient Decent',
               'Decision Tree'],
     'Score': [acc_linear_svc, acc_knn, acc_log,
               random_forest, acc_gaussian, acc_perceptron,
               acc_sgd, acc_decision_tree]})
 result_df = results.sort_values(by='Score', ascending=False,)
 result_df = result_df.set_index('Score')
 print(result_df.head(9))



 # # Use predictive model (ML)
 # prediction = classifyWithRandomForest(X_train, Y_train, X_test)
 #
 # #different level of pclass
 # train_df.pivot_table(index="Pclass", aggfunc="count")
 #
 # plt.figure(figsize=(10, 5))  # create a new structure
 # sns.countplot(x='Pclass', data=train_df)
 # plt.title('Person Count Across on Pclass')
 #
 # plt.show()
 #
 # #Pclass and Survived rate
 # plt.figure(figsize=(10, 5))
 # sns.barplot(data=train_df, x="Pclass", y="Survived", ci=None)
 # plt.show()
 #
 # #Relational between sex and survival rate
 # train_df['Sex'] = train_df['Sex'].map({0:'Male',1:'Famle'})
 # plt.figure(figsize=(8, 5))
 # sns.barplot(data=train_df, x="Sex", y="Survived", ci=None)
 #
 # plt.show()
 #
 # # Comprehensive consideration of Sex, relationship between Pclass and survival rate
 # train_df.pivot_table(values="Survived", index=["Pclass", "Sex"], aggfunc=np.mean)
 #
 # plt.figure(figsize=(10, 5))
 # sns.pointplot(data=train_df, x="Pclass", y="Survived", hue="Sex", ci=None)
 #
 # plt.show()
 #
 # #Relational between age and survival rate
 # train_df.pivot_table(values="Survived", index="Age", aggfunc=np.mean)
 # plt.figure(figsize=(10, 5))
 # sns.barplot(data=train_df, x="Age", y="Survived", ci=None)
 # plt.xticks(rotation=60)
 #
 # plt.show()
 #
 # #Relational between Embarked and Survival rate
 # train_df['Embarked'] = train_df['Embarked'].map({0: 'S', 1: 'C',2:'Q'})
 # sns.catplot('Embarked', 'Survived', data=train_df, height=5, aspect=2)
 # plt.show()
 #
 # #Relational between Age, Sex, and survival rate
 # plt.figure(figsize=(10, 5))
 # sns.pointplot(data=train_df, x="Age", y="Survived", hue="Sex", ci=None,
 #               markers=["^", "o"], linestyles=["-", "--"])
 # plt.xticks(rotation=60)
 #
 # plt.show()
 #
 # #Relationship between Fare and Survived rate
 # train_df.pivot_table(values="Survived", index="Fare", aggfunc=np.mean)
 # #train_df['Fare'] = train_df['Fare'].map({0: '0-8', 1: '9-11',2:'12-14', 3:'15-19', 4:'20-29', 5:'30-54', 6:'55-94', 7:'>95'})
 #
 # plt.figure(figsize=(10, 5))
 # sns.barplot(data=train_df, x="Fare", y="Survived", ci=None)
 # plt.xticks(rotation=0)
 #
 # plt.show()
 #
 # #Relationship between Age, Sex and Survived rate
 # #train_df['Fare'] = train_df['Fare'].map({0: '0-8', 1: '9-11', 2: '12-14', 3: '15-19', 4: '20-29', 5: '30-54', 6: '55-94', 7: '>95'})
 #
 # plt.figure(figsize=(10, 5))
 # sns.pointplot(data=train_df, x="Fare", y="Survived", hue="Sex", ci=None,
 #               markers=["^", "o"], linestyles=["-", "--"])
 # plt.xticks(rotation=60)
 #
 # plt.show()
 #
 # #Build the answer
 #
 # #Build passenagerID
 # submission = pd.DataFrame({
 #   "PassengerId": test_df["PassengerId"],
 #   "Survived": prediction
 #   })
 #
 # # Put it in csv file
 # submission.to_csv('submission_PassenagerID.csv', index=False)
 #
 # #Build Age
 # submission = pd.DataFrame({
 #   "Age": test_df["Age"],
 #   "Survived": prediction
 #   })
 #
 # submission.to_csv('submission_Age.csv', index=False)
 #
 # #Build Pclass
 # submission = pd.DataFrame({
 #   "Pclass": test_df["Pclass"],
 #   "Survived": prediction
 #   })
 #
 # # Put it in csv file
 # submission.to_csv('submission_Pclass.csv', index=False)
 #
 # #Build Sex
 # submission = pd.DataFrame({
 #   "Sex": test_df["Sex"],
 #   "Survived": prediction
 #   })
 #
 # # Put it in csv file
 # submission.to_csv('submission_Sex.csv', index=False)
 #
 # #Build Fare
 # submission = pd.DataFrame({
 #   "Fare": test_df["Fare"],
 #   "Survived": prediction
 #   })
 #
 # # visualization
 # total_survived = train_df['Survived'].sum()
 # total_no_survived = 891 - total_survived
 #
 # train_df['Survived_cat'] = train_df['Survived']. \
 #     map({0: "no-survived", 1: "survived"})
 #
 # plt.figure(figsize=(10, 5))  # create a new structure
 # plt.subplot(121)  # add the first picture
 # sns.countplot(x='Survived_cat', data=train_df)
 # plt.xlabel("Survived distribution")
 # plt.title('Survival count')
 #
 # plt.subplot(122)  # create a new structure
 # plt.pie([total_no_survived, total_survived], labels=['No Survived', 'Survived'], autopct='%1.0f%%')
 # plt.title('Survival rate')
 #
 # plt.show()
 #
 #
 #
 # # Put it in csv file
 # submission.to_csv('submission_PassenageID.csv', index=False)


 evaluateSubmission()

main( )