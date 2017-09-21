import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

def main():
    data_file = '%s/%s.csv' % ('dataset', 'train')
    df = pd.read_csv(data_file)
    filtered = df.query('Target>0')
    print(filtered.head(20))

    np_array = df.as_matrix()
    drama = np_array[:, 12]
    thriller = np_array[:, 24]
    target = np_array[:, 1]
    count = target[target == 1]
    target_drama_mean = target[drama == 1][target[drama == 1] == 1];
    target_triller_mean = target[thriller == 1][target[thriller == 1] == 1];
    stats = filtered.describe()
    print(stats)

    m = ols('Target ~ Drama + Thriller', df).fit()
    print(m.summary())


    pass

main()
