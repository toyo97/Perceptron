import numpy as np
import pandas as pd

data_file = open('data/survey_results_public.csv')

data = pd.read_csv(data_file, index_col=0)

print(data)