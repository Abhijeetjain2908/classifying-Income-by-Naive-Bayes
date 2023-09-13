import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\HP\Desktop\machine learning\dataset\adult.csv")
# print(df.shape)
# print(df.head())
# print(df)
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
              'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names
# df.columns
print(df.info())

categorical = [var for var in df.columns if df[var].dtype=='O']
print(categorical)
print(df[categorical].head())

print(df[categorical].isnull().sum())

for var in categorical: 

    print(df[var].value_counts())

for var in categorical: 

    print(df[var].value_counts()/np.cfloat(len(df)))
    print(np.cfloat(len(df)))
print(df.workclass.unique())

print(df.workclass.value_counts())

df['workclass'].replace(' ?', np.NaN, inplace=True)

print(df['workclass'].unique())

# occupation , native_country
print(df.occupation.unique())
print(df.occupation.value_counts())

df['occupation'].replace(' ?', np.NaN, inplace=True)

print(df['occupation'].unique())

print(df.native_country.unique())
print(df.native_country.value_counts())

df['native_country'].replace(' ?', np.NaN, inplace=True)

print(df['native_country'].unique())

print(df[categorical].isnull().sum())

for var in categorical:

       print(var, ' contains ', len(df[var].unique()), ' labels')

# numerical = [var for var in df.columns if df[var].dtype=='int64']
numerical = [var for var in df.columns if df[var].dtype!='O']
print(numerical)
print(df[numerical].head())
print(df[numerical].isnull().sum())

X = df.drop(['income'], axis=1)

y = df['income']

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print(X_train.shape,X_test.shape)

# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

print(categorical)

# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

print(numerical)

print(X_train[categorical].isnull().mean())


for df2 in [X_train, X_test]:

    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)

    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)

    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

print(X_train[categorical].isnull().sum())
print(X_test[categorical].isnull().sum())

import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',

                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

print(X_train.head(), X_test.head())

cols = X_train.columns

#read about scaling method in machine learning

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

print(X_train.head(), X_test.head())

# train a Gaussian Naive Bayes classifier on the training set

from sklearn.naive_bayes import GaussianNB

# instantiate the model

gnb = GaussianNB()

# fit the model

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)

print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

# check class distribution in test set

print(y_test.value_counts())

# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# how many relevant documents were retrieved by the algorithm? 150 documents Æ True positive (Tp). • How many irrelevant documents were retrieved by the algorithm? 100 documents Æ False positive (Fp) (total 250 documents retrieved – 150 relevant documents). • How many relevant documents did the algorithm not retrieve? 50 documents Æ False negative (Fn). • How many irrelevant documents did the algorithm not retrieve? 700 documents Æ True negative (Tn).

# Calculates how many correct results your solution managed to identify. • Apply the formula to the example. • negatives are almost the same. Useful for symmetric data sets where the values of positive and Accuracy = (150+700)/(1000) Accuracy = (Tp+Tn)/(Tp+Tn+Fp+Tn)

# Precision Represents the fraction of retrieved documents that are relevant. • Apply the formula to the example: Precision = 150/250 = 0.60 Precision = 150/(150+100) Precision = Tp/(Tp+Fp

# Recall Represents the fraction of relevant documents that were retrieved. • Apply the formula to the following example: Recall = 150/200 = 0.75 Recall = (150)/(150+50) Recall = Tp/(Tp+Fn

# F-Score (F-measure) • Enables you to tradeoff precision against recall. • The higher the F-score value is, the better the algorithm is. • Here is the formula: • Apply the formula to the example. F = 0.9/1.35= 0.6667 F = (2*0.60*0.75)/(0.60+0.75) F = 2*Precision*Recall/(Precision+Recall     

# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

# True Positives (TP) – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

# True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

# False Positives (FP) – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.

# False Negatives (FN) – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

 
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# plt.show()

# classification matrices

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

TP = cm[0,0]

TN = cm[1,1]

FP = cm[0,1]

FN = cm[1,0]

# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

# print precision score

precision = TP / float(TP + FP)

print('Precision : {0:0.4f}'.format(precision))


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


false_positive_rate = FP / float(FP + TN)

print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))

f1score=2*(precision*recall)/(precision+recall)
print("f1score: {0:0.4f}".format(f1score))

# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:10]

print(y_pred_prob)

# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

print(y_pred_prob_df['Prob of - <=50K'].head(10))
print(y_pred_prob_df['Prob of - >50K'].head(10))

# print the first 10 predicted probabilities for class 1 - Probability of >50K

# gnb.predict_proba(X_test)[0:10, 1]

# store the predicted probabilities for class 1 - Probability of >50K

y_pred1 = gnb.predict_proba(X_test)[:, 1]
# print(y_pred1)

# plot histogram of predicted probabilities

# adjust the font size

plt.rcParams['font.size'] = 12

# plot histogram with 10 bins

plt.hist(y_pred1, bins = 10)

# set the title of predicted probabilities

plt.title('Histogram of predicted probabilities of salaries >50K')

# set the x-axis limit

plt.xlim(0,1)

# set the title

plt.xlabel('Predicted probabilities of salaries >50K')

plt.ylabel('Frequency')

plt.show()



# plot ROC Curve

 

from sklearn.metrics import roc_curve

 

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')

 

plt.figure(figsize=(6,4))

 

plt.plot(fpr, tpr, linewidth=2)

 

plt.plot([0,1], [0,1], 'k--' )

 

plt.rcParams['font.size'] = 12

 

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

 

plt.xlabel('False Positive Rate (1 - Specificity)')

 

plt.ylabel('True Positive Rate (Sensitivity)')

 

plt.show()

# compute ROC AUC

 

from sklearn.metrics import roc_auc_score

 

ROC_AUC = roc_auc_score(y_test, y_pred1)

 

print('ROC AUC : {:.4f}'.format(ROC_AUC))

# calculate cross-validated ROC AUC

from sklearn.model_selection import cross_val_score


Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

# Applying 10-Fold Cross Validation 

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# compute Average cross-validation score   ... by Nikhil Rana [Croma Campus]

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))