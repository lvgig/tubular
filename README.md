# tubular

----

`tubular` implements transformers for pre processing steps commonly used in machine learning pipelines.

The transformers are compatible with scikit-learn [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), having a `transform` method to apply the pre processing step to data and a `fit` method to learn the relevant information from the data, if applicable.

The transformers in `tubular` work with data in [pandas DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

There are a variety of transformers to assist with;

- capping
- imputation
- mapping
- date differencing
- categorical encoding
- numeric operations

Here is a simple example of capping 2 columns at a specified value;

```python
from tubular.capping import CappingTransformer
import pandas as pd
from sklearn.datasets import load_boston

# load the boston dataset
boston = load_boston()
y = boston.target
X = pd.DataFrame(boston.data, columns=boston.feature_names)

# initialise a capping transformer for 2 columns
capper = CappingTransformer(columns=['INDUS', 'RM'], cap_value_max = 20)

# transform the data
X_capped = capper.transform(X)
```

## Installation

tubular can be installed from PyPI simply with;

 `pip install tubular`

## Documentation

To build local documentation, specify the environment variable $SPHINX_BUILD_DIR$, and then
run from the `docs/` directory

```shell
make apidoc
make html
```

## Examples

To help get started there are example notebooks in the [examples](https://github.com/lvgig/tubular/tree/master/examples) folder that show how to use each transformer as well as an example of putting several together in a Pipeline.

## Build and test

The test framework we are using for this project is [pytest](https://docs.pytest.org/en/stable/), to run the tests follow the steps below.

First clone the repo and move to the root directory;

```shell
git clone https://github.com/lvgig/tubular.git
cd tubular
```

Then install tubular in editable mode;

```shell
pip install -e . -r requirements-dev.txt
```

Then run the tests simply with pytest

```shell
pytest
```

## Contribute

`tubular` is under active development, we're super excited if you're interested in contributing! See the `CONTRIBUTING.md` for the full details of our working practices.

For bugs and feature requests please open an [issue](https://github.com/lvgig/tubular/issues).
