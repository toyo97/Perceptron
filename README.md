# Perceptron
_Survey data analysis and prediction using Perceptron model on Scikit-Learn_

- [Description](#description)
- [Usage](#usage)
- [Sources](#sources)

## Description
With this simple program we want to analyze the dataset from [Stack Overflow Developer Survey, 2017](https://www.kaggle.com/stackoverflow/so-survey-2017).
In particular we use the [Perceptron](https://en.wikipedia.org/wiki/Perceptron) linear model from SciKit-Learn library,
studying the behaviour of this basic machine learning algorithm for supervised learning.
The aim is to train the model recognising if a developer's salary is lower or greater than the median of all salaries;
then we let it make predictions for unseen data.

This program also allows us to:
- check the accuracy of the predictions with a [10-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation)
- find the best hyperparameters for the model (which are in our case the tolerance and the epochs number) with 
[Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) tuning
- plot the learning curves of the model to verify that it does not overfit given the whole dataset

For any further information about both analysis and script, please read the whole documentation in Report.pdf

## Usage
To reproduce the results shown in the report, just run the Perceptron.py script using a python 3 environment.
Required packages:
- `numpy 1.15.4`
- `pandas 0.24.0`
- `scikit-learn 0.20.2`
- `matplotlib 3.0.2`

(version numbers are those used during coding, but older versions could be sufficient)

## Sources
Most of the work has been done reading the official documentation of SciKit-Learn ([User Guide](https://scikit-learn.org/stable/user_guide.html),
[Tutorials](https://scikit-learn.org/stable/tutorial/index.html) and [API Reference](https://scikit-learn.org/stable/modules/classes.html)).\
For prior data manipulation also Pandas official [API Reference](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) has been useful.\
Other ideas has been found spread over the Internet, for example in the following article which shows how to combine Pandas and Scikit-Learn
to transform a dataset in order to fit a sklearn model: [From Pandas to Scikit-Learn — A new exciting workflow](https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62).