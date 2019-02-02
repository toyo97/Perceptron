import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')


def multi_cat_transform(X):
    columns = X.columns.values
    for col in columns:
        dummies = X[col].str.get_dummies(sep='; ')
        dummies.columns = [col+'_'+c for c in dummies.columns]
        X = pd.concat([X, dummies], axis=1)
    return X.drop(columns=columns)


# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Read CSV file and import in Pandas DataFrame
data_file = open('data/survey_results_public.csv')
df = pd.read_csv(data_file, index_col=0)

# Early manipulation of DF:
# - remove ExpectedSalary col (not relevant)
# - remove rows without salary value
# - extract target (Salary)
df.drop(['ExpectedSalary'], axis=1, inplace=True)
df = df[pd.notnull(df['Salary'])]
salary_col = df.pop('Salary')

# Binarize target using median threshold (0 for values lt median, 1 for values gt median)
median_salary = salary_col.median()
y = np.array(salary_col.values >= median_salary, dtype=np.int64)

# Separate train-set and test-set with test size fixed above
df_train, df_test, y_train, y_test = train_test_split(df,
                                                      y,
                                                      test_size=TEST_SIZE,
                                                      random_state=RANDOM_STATE)

# Separate features (columns) in numeric-value f., category f. and multiple category f.
# Create numpy boolean index array
# 'O' stands for Object, while numeric values are floats or ints
kinds = np.array([dt.kind for dt in df.dtypes])
is_num = kinds != 'O'

all_cols = df.columns.values

num_cols, cat_cols = all_cols[is_num], all_cols[~is_num]

# High unique values count indicates multiple category feature (to be separated) - plus exceptions
# Threshold (200) found empirically from dataset overview using the following procedure:
# Detail on number of unique values per column
#
# unique_features = df.nunique().sort_values(ascending=False)
# unique_features[:40]
# df[unique_features.index[9:27]]
# df['VersionControl'].value_counts()
#
# same for the three exceptions
multi_cat_cols = [c for c in df[cat_cols] if df[c].nunique() >= 200] + ['Race', 'StackOverflowDevices', 'Gender']
single_cat_cols = [c for c in cat_cols if c not in multi_cat_cols]

# Creation of scikit-learn Pipelines and ColumnTransformer to operate dataset transformation
# Numeric values must be imputed where missing (replacing NaN with median value) and scaled around 0.
num_si_step = ('si', SimpleImputer(missing_values=np.nan, strategy='median'))
num_ss_step = ('ss', StandardScaler())
num_steps = [num_si_step, num_ss_step]

# Category values also must be imputed where missing (with NotAnswered constant category) and encoded
# OneHotEncoder creates k columns of ones and zero where k is the number of unique categories
cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value='NA'))
cat_ohe_step = ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
cat_steps = [cat_si_step, cat_ohe_step]

# Some features are groups of different categories, they are split and encoded using a FunctionTransformer which
# calls the custom function defined at the top of the file
multi_cat_step = ('mce', FunctionTransformer(multi_cat_transform, validate=False))

num_pipe = Pipeline(num_steps)
cat_pipe = Pipeline(cat_steps)
multi_cat_pipe = Pipeline([multi_cat_step])

multi_cat_transformer = ('multi_cat', multi_cat_pipe, multi_cat_cols)
num_transformer = ('num', num_pipe, num_cols)
cat_transformer = ('cat', cat_pipe, single_cat_cols)
transformers = [multi_cat_transformer, num_transformer, cat_transformer]

ct = ColumnTransformer(transformers=transformers)

# Use the transformer to obtain a numeric numpy matrix
X_train = ct.fit_transform(df_train)

print('n_train_samples, n_train_features: {}'.format(X_train.shape))

# Select the Perceptron linear model from scikit-learn library (with some parameters)
clf = Perceptron(max_iter=40, tol=1e-3)

print('\n10-FOLD CROSS VALIDATION')
# Use cross_val_score function from sklearn to get accuracy scores out of a 10-fold cross validation
k_fold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(clf, X_train, y_train, cv=k_fold)
print('10-fold cross validation scores: {}'.format(scores))
print('Mean: {}'.format(scores.mean()))

print('\nGRID SEARCH')
# Perceptron hyperparameters tuning using GridSearch with 10-fold cv
alpha_params = [0.0001, 0.0003, 0.001, 0.003, 0.01]
max_iter_params = [5, 10, 15, 20, 50]

param_grid = {
    'alpha': alpha_params,
    'max_iter': max_iter_params
}

grid_search = GridSearchCV(Perceptron(tol=1e-3), param_grid, scoring='accuracy', cv=k_fold)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('Best params: {}'.format(best_params))
print('Grid search best score: {}'.format(grid_search.best_score_))
results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_alpha', 'param_max_iter', 'mean_test_score', 'std_test_score', 'mean_fit_time']])


print('\nPREDICTION TEST')
# Use best parameters to predict salary in the test set
X_test = ct.transform(df_test)
best_clf = Perceptron(alpha=best_params['alpha'], max_iter=best_params['max_iter'], tol=1e-3)
best_clf.fit(X_train, y_train)
print('Score on test set: {}'.format(best_clf.score(X_test, y_test)))

print('\nLEARNING CURVE')
# Plot the learning curve
X = ct.fit_transform(df)

train_sizes = np.linspace(.1, 1.0, 5)

train_sizes, train_scores, test_scores = learning_curve(best_clf, X, y, cv=k_fold, n_jobs=4, train_sizes=train_sizes)

plt.figure()
plt.title('Learning Curves (Perceptron)')
plt.ylim((0.7, 1.01))
plt.xlabel('Training samples')
plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                 alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                 alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
         label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
         label='Cross-validation score')

plt.legend(loc='best')
plt.show()
