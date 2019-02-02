import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV


def multi_cat_transform(X):
    columns = X.columns.values
    for col in columns:
        dummies = X[col].str.get_dummies(sep='; ')
        dummies.columns = [col+'_'+c for c in dummies.columns]
        X = pd.concat([X, dummies], axis=1)
    return X.drop(columns=columns)


data_file = open('data/survey_results_public.csv')

df = pd.read_csv(data_file, index_col=0)

df.drop(['ExpectedSalary'], axis=1, inplace=True)
df = df[pd.notnull(df['Salary'])]
salary_col = df.pop('Salary')

median_salary = salary_col.median()

y = np.array(salary_col.values >= median_salary, dtype=np.int64)

# unique_features = df.nunique().sort_values(ascending=False)
# unique_features[:40]

# df[unique_features.index[9:27]]

# df['VersionControl'].value_counts()

kinds = np.array([dt.kind for dt in df.dtypes])

all_cols = df.columns.values
is_num = kinds != 'O'
num_cols, cat_cols = all_cols[is_num], all_cols[~is_num]

# high unique values count indicates multiple category feature (to be separated) - plus exceptions
multi_cat_cols = [c for c in df[cat_cols] if df[c].nunique() >= 200] + ['Race', 'StackOverflowDevices', 'Gender']
single_cat_cols = [c for c in cat_cols if c not in multi_cat_cols]

df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


multi_cat_step = ('mce', FunctionTransformer(multi_cat_transform, validate=False))

num_si_step = ('si', SimpleImputer(missing_values=np.nan, strategy='mean'))
num_ss_step = ('ss', StandardScaler())
num_steps = [num_si_step, num_ss_step]

cat_si_step = ('si', SimpleImputer(strategy='constant', fill_value='NA'))
cat_ohe_step = ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
cat_steps = [cat_si_step, cat_ohe_step]


multi_cat_pipe = Pipeline([multi_cat_step])
num_pipe = Pipeline(num_steps)
cat_pipe = Pipeline(cat_steps)


multi_cat_transformer = ('multi_cat', multi_cat_pipe, multi_cat_cols)
num_transformer = ('num', num_pipe, num_cols)
cat_transformer = ('cat', cat_pipe, single_cat_cols)
transformers = [multi_cat_transformer, num_transformer, cat_transformer]

ct = ColumnTransformer(transformers=transformers)

X_train = ct.fit_transform(df_train)

clf = Perceptron(max_iter=40, tol=1e-3)

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
print('10-fold cross validation scores: {}', cross_val_score(clf, X_train, y_train, cv=k_fold))

alpha_params = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
max_iter_params = [5, 10, 15, 20, 50]

param_grid = {
    'alpha': alpha_params,
    'max_iter': max_iter_params
}

grid_search = GridSearchCV(Perceptron(tol=1e-3), param_grid, scoring='accuracy', cv=k_fold)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('Best params: {}', best_params)
print('Grid search best score: {}', grid_search.best_score_)
results = grid_search.cv_results_
print(pd.DataFrame(results))

X_test = ct.transform(df_test)
best_clf = Perceptron(alpha=best_params['alpha'], max_iter=best_params['max_iter'], tol=1e-3)
best_clf.fit(X_train, y_train)
print(best_clf.score(X_test, y_test))
