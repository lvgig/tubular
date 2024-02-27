import pandas as pd
from sklearn.datasets import fetch_california_housing

import tubular
from tubular.imputers import MeanImputer

pd.set_option("mode.copy_on_write", True)

tubular.__version__

# Mean imputer

cali = fetch_california_housing()
cali_df = pd.DataFrame(cali["data"], columns=cali["feature_names"])
cali_df["AveOccup"] = cali_df["AveOccup"].sample(frac=0.99, random_state=1)
cali_df["HouseAge"] = cali_df["HouseAge"].sample(frac=0.95, random_state=2)
cali_df["Population"] = cali_df["Population"].sample(frac=0.995, random_state=3)

imp_1 = MeanImputer(columns=["HouseAge", "AveOccup", "Population"], copy=False, verbose=False)

for loop in range(1000):
  imp_1.fit(cali_df)
  cali_df_2 = imp_1.transform(cali_df)


print('Complete')