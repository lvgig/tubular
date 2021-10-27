# Change log

----

## 0.3.0

- Removed testing folder and updated tests to use helpers from `test-aide` package
- Moved test_data.py to tests folder
- Added `test-aide` to requirements-dev
- Updated example notebooks to use California housing dataset from sklearn instead of Boston house prices dataset

## 0.2.15

- Update NearestMeanResponseImputer to remove fallback to median imputation when no nulls present in a column
- Add github action to run pytest, flake8, black and bandit 
- Modified GroupRareLevelsTransformer to remove the constraint type of rare_level_name being string, instead it must be the same type as the columns selected
- Fix failing NullIndicator.transform tests

## 0.2.14

- Open source release of code on Github

## 0.2.13

- Added base parameter to LogTransformer in numeric module
- Fixed bug in MappingTransformer in mapping module. Changed local variable to be instantiated as deepcopy of input parameter.

## 0.2.12

- Add OrdinalEncoderTransformer in noninal module
- Add check that X has rows in BaseTransformer transform method
- Use kind="stable" in numpy argsort and kind="mergesort" in pandas sort_values for stable sorting
- Fix OutOfRangeNullTransformer not returning self from fit

## 0.2.11

- Change ArbitraryImputer so that columns must be set in init
- Swap out assert_function_call_b and assert_function_call_count_b functions for new assert_function_call and assert_function_call_count context manager versions
- Add support for pd.Index objects in testing.helpers.assert_equal_dispatch

## 0.2.10

- Change assert_function_call and assert_function_call_count to be context managers
- Rename existing assert_function_call and assert_function_call_count functions to assert_function_call_b and assert_function_call_count_b
- Simplify assert_eqaul_dispatch function

## 0.2.9

- Add BaseImputer class and make other imputer classes inherit from BaseImputer
- Change NominalColumnSetOrCheckMixin to be BaseNominalTransformer 
- Rename replacements_ attribute to mappings in MeanResponseTransformer to be consistent with MappingTransformer
- Rename replacements_ attribute to mappings in NominalToIntegerTransformer to be consistent with MappingTransformer
- Uupdate BaseTransformer.transform to call columns_check instead of columns_set_or_check
- Added method to BaseNominalTransformer to checks for mappable rows and raises an error if non-mappable rows are found
- Implement BaseMappingTransformMixin class and inherit from that in MappingTransformer
- Update MeanResponseTransformer and NominalToIntegerTransformer to take their transform method from BaseMappingTransformMixin
- Update GroupRareLevelsTransformer to map columns using pd.where() insteand of np.where()

## 0.2.8

- Rename package from prepro to tubular

## 0.2.7

- Add fit method to calculate weighted quantiles to CappingTransformer
- Add CutTransformer
- Add LogTransformer
- Add ToDatetimeTransformer
- Add SeriesDtMethodTransformer
- Add SeriesStrMethodTransformer
- Add OutOfRangeNullTransformer
- Add ScalingTransformer
- Add SetVaueTransformer
- Add BetweenDatesTransformer

## 0.2.6

- Update build pipelines
- Add date diff transformer (Y, M, D, h, m, s units)
- Rename DateDiffYearTransformer to DateDiffLeapYearTransformer
- Add DataFrameMethodTransformer

## 0.2.5

- Add files in preparation for open source release
- Restructure test helper files
- Move NominalColumnSetOrCheckMixin to tubular/nominal.py
- Remove exception raised decorator
- Add row_by_row_params and index_preserved_params test helpers and update tests to use

## 0.2.4

- Change imports for test helpers

## 0.2.3

- Add mean imputer
- Add mode imputer

## 0.2.2

- Add date diff year transformer

## 0.2.1

- Add nearest mean response imputer
- Add null indicator

## 0.2.0

- Refactor test suite
- Improve documentation
- Add cross column mapping transformers

## 0.1.13

- Fix to one hot encoder to preserve index

## 0.1.12

- Add capping transformer

## 0.1.11

- Add y arg to one hot encoder

## 0.1.10

- Add one hot encoder

## 0.1.9

- Fix for pandas update

## 0.1.8

- Improve speed of mapping transformer

## 0.1.7

- Fix missing y arg in mean response transformer

## 0.1.6

- Add mapping transformer

## 0.1.5

- Add mean response transformer

## 0.1.4

- Improve speed of rare level grouper
- Improve speed of imputers

## 0.1.3

- Add checks on inputs for base transformer
- Add requirements.txt

## 0.1.2

- Fix for rare level grouper

## 0.1.1

- Add version attribute

## 0.1.0

- Add base transformer
- Add nominal to integer transformer
- Add categorical rare level grouper
- Add arbitrary imputer
- Add median imputer
