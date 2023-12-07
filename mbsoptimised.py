import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

# Read data
data = pd.read_csv('LoanExport.csv')

# Function for drawing histograms
def draw_histogram(col, plot_title, xlabel, ylabel):
    plt.figure(figsize=(30, 10))
    plt.hist(data[col], edgecolor='black')
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Function for handling outliers
def handle_outlier(col):
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)
    return lower, upper

# Function for label encoding
def label_encoding(col):
    label_encoder = preprocessing.LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])

# Function for one-hot encoding
def one_hot_encoding(cols):
    one_hot_encoded_data = pd.get_dummies(data, columns=cols)
    return one_hot_encoded_data

# Convert date feature to datetime format
data['MaturityDate'] = pd.to_datetime(data['MaturityDate'], format='%Y%m')
data['FirstPaymentDate'] = pd.to_datetime(data['FirstPaymentDate'], format='%Y%m')

# Visualization
# Barplot
plt.figure(figsize=(15, 10))
sns.barplot('OrigLoanTerm', 'MonthsInRepayment', data=data, palette='gist_rainbow')
plt.xlabel('Original Loan Rate')
plt.show()

# Feature engineering
# Drop unnecessary features
drop_features = ['FirstTimeHomebuyer_X', 'FirstTimeHomebuyer_N', 'ServicerName', 'SellerName', 'NumBorrowers',
                 'PropertyType', 'PropertyState', 'ProductType', 'PPM', 'Channel', 'Occupancy', 'MSA', 'MaturityDate',
                 'FirstPaymentDate', 'PostalCode', 'LoanSeqNum']
data.drop(drop_features, axis=1, inplace=True)

# Heatmap for correlation
plt.figure(figsize=(10, 7))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.show()

# Handling outliers for specific columns
outlier_columns = ['OrigUPB', 'Units', 'OrigInterestRate']
for col in outlier_columns:
    lower, upper = handle_outlier(data[col])
    data[col] = np.clip(data[col], lower, upper)

# Drop duplicate values
data.drop_duplicates(inplace=True)

# Feature creation
data['CreditRange'] = data.apply(lambda row: 'excellent' if row['CreditScore'] >= 750 else ('good' if row['CreditScore'] >= 700
                                 else ('fair' if row['CreditScore'] >= 650 else 'poor')), axis=1)

data['LTVRange'] = pd.cut(data['LTV'], bins=[0, 75, 80, 90, np.inf], labels=['low', 'medium', 'high', 'very high'])
data['RepayRange'] = pd.cut(data['OrigInterestRate'], bins=[-np.inf, 4, 6, np.inf], labels=['low', 'medium', 'high'])

# Convert categorical ranges to numerical features
credit_range_dummies = pd.get_dummies(data['CreditRange'], prefix='CreditRange')
label_encoding('LTVRange')
repay_range_dummies = pd.get_dummies(data['RepayRange'], prefix='RepayRange')

# Concatenate numerical features to the original data
raw_data = pd.concat([data, credit_range_dummies, repay_range_dummies], axis=1)

# Drop original categorical ranges
raw_data.drop(['CreditRange', 'RepayRange'], axis=1, inplace=True)

# Drop highly correlated features and target-related features
drop_features = ["MonthsInRepayment", 'MonthsDelinquent']
data.drop(drop_features, axis=1, inplace=True)
raw_data.drop(drop_features, axis=1, inplace=True)

# Split data
x = raw_data.drop(['EverDelinquent'], axis=1)
y = raw_data['EverDelinquent']

# Principal Component Analysis (PCA)
pca = PCA().fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Standardize data for PCA
scaler = StandardScaler()
x_standardized = scaler.fit_transform(x)

# Selecting important features using Mutual Information
selector = SelectKBest(mutual_info_classif, k=15)
x_selected = selector.fit_transform(x, y)
selected_features = x.columns[selector.get_support(indices=True)].tolist()

# Feature importance using Extra Trees Classifier
model = ExtraTreesClassifier()
model.fit(x, y)
feat_imp = pd.Series(model.feature_importances_, index=x.columns)
feat_imp.nlargest(20).plot(kind='barh')
plt.show()

# Feature importance using Mutual Information Regression
mi_scores = make_mi_scores(x, y)
mi_scores[::]

# Select best features for PCA
selector = SelectKBest(mutual_info_regression, k=11)
X_selected = selector.fit_transform(x, y)

# Standardize data for PCA
scaler = StandardScaler()
x_standardized = scaler.fit_transform(x)

# Apply PCA
pca = PCA(n_components=6)
features = pca.fit_transform(x_standardized)

# Modelling
# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=8)

# Oversample using SMOTE
oversampled = SMOTE(random_state=0)
X_train_smote, y_train_smote = oversampled.fit_resample(X_train, y_train)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=2)
rf.fit(X_train_smote, y_train_smote)
y_pred_test_rf = rf.predict(X_test)
testing_accuracy_rf = accuracy_score(y_test, y_pred_test_rf) * 100

# Decision Tree Classifier
model_d = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=44, ccp_alpha=0.8,
                                 min_impurity_decrease=0.6)
model_d.fit(X_train_smote, y_train_smote)
y_pred_test_dt = model
