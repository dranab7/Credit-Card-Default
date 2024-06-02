# Credit-Card-Default


Project Overview
This project aims to predict credit card default events using a dataset containing 2000 samples, with each sample representing a customer. The dataset includes the following features: Income, Age, Loan, Loan to Income ratio, and Default status. The target variable is Default, indicating whether a customer defaulted on their credit card payment. The analysis and model building follow a systematic approach from data import to model evaluation.

Step-by-Step Implementation
Importing Libraries

python
Copy code
import pandas as pd
We start by importing the necessary library, pandas, which is essential for data manipulation and analysis.

Loading the Data

python
Copy code
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')
default.head()
The dataset is loaded directly from a URL. The first few rows of the dataset are displayed to understand its structure.

Data Inspection

python
Copy code
default.info()
default.describe()
We inspect the dataset to check the data types and identify any missing values. Descriptive statistics provide insights into the distribution and central tendencies of the features.

Defining Target and Features

python
Copy code
y = default['Default']
X = default.drop(['Default'], axis=1)
The target variable (y) is set to Default, and the features (X) are all other columns.

Splitting the Data

python
Copy code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
We split the dataset into training and testing sets, with 70% of the data used for training and 30% for testing.

Model Selection and Training

python
Copy code
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
We select Logistic Regression for this classification task. The model is trained on the training data.

Model Coefficients

python
Copy code
model.intercept_
model.coef_
The model's intercept and coefficients are extracted, providing insights into the influence of each feature on the prediction.

Making Predictions

python
Copy code
y_pred = model.predict(X_test)
Predictions are made on the test data.

Model Evaluation

python
Copy code
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
The model's performance is evaluated using a confusion matrix, accuracy score, and classification report. The confusion matrix shows true positives, true negatives, false positives, and false negatives. Accuracy, precision, recall, and F1-score are calculated to give a comprehensive view of the model's performance.

Results and Analysis
Confusion Matrix:

python
Copy code
array([[506,  13],
       [ 17,  64]])
The confusion matrix indicates that out of 600 test samples, 506 non-defaults and 64 defaults were correctly predicted. There were 13 false positives and 17 false negatives.

Accuracy Score:

python
Copy code
0.95
The model achieved an accuracy of 95%, indicating that it correctly predicted the default status for 95% of the test samples.

Classification Report:

python
Copy code
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       519
           1       0.83      0.79      0.81        81

    accuracy                           0.95       600
   macro avg       0.90      0.88      0.89       600
weighted avg       0.95      0.95      0.95       600
Precision, recall, and F1-score for non-defaults (class 0) are very high, showing excellent model performance for the majority class.
For defaults (class 1), the model also performs well with an F1-score of 0.81, though there is a slight drop in recall (0.79), indicating some room for improvement in identifying all default cases.
Conclusion
This project successfully built a Logistic Regression model to predict credit card default with high accuracy and reliable performance metrics. Despite the class imbalance (more non-defaults than defaults), the model demonstrated strong predictive capabilities. Future improvements could include addressing the class imbalance through techniques such as SMOTE or exploring other classification algorithms to enhance recall for the default class.






