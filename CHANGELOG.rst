Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into `main` (e.g. with a .dev suffix) but which are not yet in a new release (on pypi) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.


1.3.0 (unreleased)
------------------
Added
^^^^^
- Inheritable tests for generic base behaviours for base transformer in `base_tests.py`, with fixtures to allow for this in `conftest.py`
- Split existing input check into two better defined checks for TwoColumnOperatorTransformer `#183 <https://github.com/lvgig/tubular/pull/183>`_
- Created unit tests for checking column type and size `#183 <https://github.com/lvgig/tubular/pull/183>`_

Changed
^^^^^^^
- Update DataFrameMethodTransformer tests to have inheritable init class that can be used by othe test files.
- Moved BaseTransformer, DataFrameMethodTransformer, BaseMappingTransformer, BaseMappingTransformerMixin, CrossColumnMappingTransformer and Mapping Transformer over to the new testing framework.
- Refactored MappingTransformer by removing redundant init method.
- Updated tests for 
- Refactored ArbitraryImputer by removing redundant fillna call in transform method. This should increase tubular's efficiency and maintainability.
- Refactored ArbitraryImputer and BaseImputer tests in new format.
- Refactored MedianImputer tests in new format.
- Replaced occurrences of pd.Dataframe.drop() with del statement to speed up tubular. Note that no additional unit testing has been done for copy=False as this release is scheduled to remove copy. 

Removed
^^^^^^^
- Functionality for BaseTransformer (and thus all transformers) to take `None` as an option for columns. This behaviour was inconsistently implemented across transformers. Rather than extending to all we decided to remove 
this functionality. This required updating a lot of test files.
- The `columns_set_or_check()` method from BaseTransformer. With the above change it was no longer necessary. Subsequent updates to nominal transformers and their tests were required.
- Set pd copy_on_write to True (will become default in pandas 3.0) which allowed the functionality of the copy method of the transformers to be dropped `#197 <https://github.com/lvgig/tubular/pull/197>`

1.2.2 (2024-02-20)
------------------
Added
^^^^^
- Created unit test for checking if log1p is working and well conditioned for small x `#178 <https://github.com/lvgig/tubular/pull/178>`_

Changed
^^^^^^^
- Changed LogTransformer to use log1p(x) instead of log(x+1) `#178 <https://github.com/lvgig/tubular/pull/178>`_
- Changed unit tests using log(x+1) to log1p(x) `#178 <https://github.com/lvgig/tubular/pull/178>`_

1.2.1 (2024-02-08)
------------------
Added
^^^^^
- Updated GroupRareLevelsTransformer so that when working with category dtypes it forgets categories encoded as rare (this is wanted behaviour as these categories are no longer present in the data) `<#177 https://github.com/lvgig/tubular/pull/177>`_

1.2.0 (2024-02-06)
------------------
Added
^^^^^
- Update OneHotEncodingTransformer to default to returning int8 columns `#175 <https://github.com/lvgig/tubular/pull/175>`_
- Updated NullIndicator to return int8 columns `<#173 https://github.com/lvgig/tubular/pull/173>`_
- Updated MeanResponseTransformer to coerce return to float (useful behaviour for category type features) `#174 <https://github.com/lvgig/tubular/pull/174>`_

1.1.1 (2024-01-18)
------------------

Added
^^^^^
- added type hints `#128 <https://github.com/lvgig/tubular/pull/128>`_
- added some error handling to transform method of nominal transformers  `#162 <https://github.com/lvgig/tubular/pull/162>`_
- added new release pipeline `#161 <https://github.com/lvgig/tubular/pull/161>`_

1.1.0 (2023-12-19)
------------------

Added
^^^^^
- added flake8_bugbear (B) to ruff rules `#131 <https://github.com/lvgig/tubular/pull/131>`_
- added flake8_datetimez (DTZ) to ruff rules `#132 <https://github.com/lvgig/tubular/pull/132>`_
- added option to avoid passing unseen levels to rare in GroupRareLevelsTransformer `#141 <https://github.com/lvgig/tubular/pull/141>`_

Changed
^^^^^^^
- minor changes to comply with flake8_bugbear (B) ruff rules `#131 <https://github.com/lvgig/tubular/pull/131>`_
- minor changes to comply with flake8_datetimez (DTZ) ruff rules `#132 <https://github.com/lvgig/tubular/pull/132>`_
- BaseMappingTransformerMixin chnaged to use Dataframe.replace rather than looping over columns `#135 <https://github.com/lvgig/tubular/pull/135>`_
- MeanResponseTransformer.map_imputer_values() added to decouple from BaseMappingTransformerMixin `#135 <https://github.com/lvgig/tubular/pull/135>`_
- BaseDateTransformer added to standardise datetime data handling `#148 <https://github.com/lvgig/tubular/pull/148>`_

Removed
^^^^^^^
- removed some unnescessary implementation tests `#130 <https://github.com/lvgig/tubular/pull/130>`_
- ReturnKeyDict class removed `#135 <https://github.com/lvgig/tubular/pull/135>`_




1.0.0 (2023-07-24)
------------------

Changed
^^^^^^^
- now compatible with pandas>=2.0.0 `#123 <https://github.com/lvgig/tubular/pull/123>`_
- DateDifferenceTransformer no longer supports 'Y' or  'M' units `#123 <https://github.com/lvgig/tubular/pull/123>`_


0.3.8 (2023-07-10)
------------------

Changed
^^^^^^^
- replaced flake8 with ruff linting.  For a list of rules implemented, code changes made for compliance and further rule sets planned for future see PR  `#92 <https://github.com/lvgig/tubular/pull/92>`_

0.3.7 (2023-07-05)
------------------

Changed
^^^^^^^
- minor change to `GroupRareLevelsTransformer` `test_super_transform_called` test to align with other cases `#90 <https://github.com/lvgig/tubular/pull/90>`_
- removed pin of scikit-learn version to <1.20 `#90 <https://github.com/lvgig/tubular/pull/90>`_
- update `black` version in pre-commit-config `#90 <https://github.com/lvgig/tubular/pull/90>`_

0.3.6 (2023-05-24)
------------------

Added
^^^^^
- added support for vscode dev container with python 3.8, requirments-dev.txt, pylance/gitlens extensions and precommit all preinstalled `#83 <https://github.com/lvgig/tubular/pull/83>`_

Changed
^^^^^^^
- added sklearn < 1.2 dependency `#86 <https://github.com/lvgig/tubular/pull/86>`_

0.3.5 (2023-04-26)
------------------

Added
^^^^^
- added support for handling unseen levels in MeanResponseTransformer `#80 <https://github.com/lvgig/tubular/pull/80>`_

Changed
^^^^^^^
- added pandas < 2.0.0 dependency `#81 <https://github.com/lvgig/tubular/pull/81>`_

Deprecated
^^^^^^^^^^
- DateDifferenceTransformer M and Y units are incpompatible with pandas 2.0.0 and will be removed or changed in a future version `#81 <https://github.com/lvgig/tubular/pull/81>`_

0.3.4 (2023-03-14)
------------------

Added
^^^^^
- added support for passing multiple columns and periods/units parameters to DatetimeSinusoidCalculator `#74 <https://github.com/lvgig/tubular/pull/74>`_
- added support for handling a multi level response to MeanResponseTransformer `#67 <https://github.com/lvgig/tubular/pull/67>`_

Changed
^^^^^^^
- changed ArbitraryImputer to preserve the dtype of columns (previously would upcast dtypes like int8 or float32) `#76 <https://github.com/lvgig/tubular/pull/76>`_

Fixed
^^^^^

- fixed issue with OneHotencodingTransformer use of deprecated sklearn.OneHotEencoder.get_feature_names method `#66 <https://github.com/lvgig/tubular/pull/66>`_

0.3.3 (2023-01-19)
------------------

Added
^^^^^
- added support for prior mean encoding (regularised encodings) `#46 <https://github.com/lvgig/tubular/pull/46>`_

- added support for weights to mean, median and mode imputers `#47 <https://github.com/lvgig/tubular/pull/47>`_

- added classname() method to BaseTransformer and prefixed all errors with classname call for easier debugging `#48 <https://github.com/lvgig/tubular/pull/48>`_

- added DatetimeInfoExtractor transformer in ``tubular/dates.py`` associated tests with ``tests/dates/test_DatetimeInfoExtractor.py`` and examples with ``examples/dates/DatetimeInfoExtractor.ipynb`` `#49 <https://github.com/lvgig/tubular/pull/49>`_

- added DatetimeSinusoidCalculator in ``tubular/dates.py`` associated tests with ``tests/dates/test_DatetimeSinusoidCalculator.py`` and examples with ``examples/dates/DatetimeSinusoidCalculator.ipynb`` `#50 <https://github.com/lvgig/tubular/pull/50>`_

- added TwoColumnOperatorTransformer in ``tubular/numeric.py`` associated tests with ``tests/numeric/test_TwoColumnOperatorTransformer.py`` and examples with ``examples/dates/TwoColumnOperatorTransformer.ipynb`` `#51 <https://github.com/lvgig/tubular/pull/51>`_

- added StringConcatenator in ``tubular/strings.py`` associated tests with ``tests/strings/test_StringConcatenator.py`` and examples with ``examples/strings/StringConcatenator.ipynb`` `#52 <https://github.com/lvgig/tubular/pull/52>`_

- added SetColumnDtype in ``tubular/misc.py`` associated tests with ``tests/misc/test_StringConcatenator.py`` and examples with ``examples/strings/StringConcatenator.ipynb`` `#53 <https://github.com/lvgig/tubular/pull/53>`_

- added warning to MappingTransformer in ``tubular/mapping.py`` for unexpected changes in dtype  `#54 <https://github.com/lvgig/tubular/pull/54>`_

- added new module ``tubular/comparison.py`` containing EqualityChecker.  Also added associated tests with ``tests/comparison/test_EqualityChecker.py`` and examples with ``examples/comparison/EqualityChecker.ipynb`` `#55 <https://github.com/lvgig/tubular/pull/55>`_

- added PCATransformer in ``tubular/numeric.py`` associated tests with ``tests/misc/test_PCATransformer.py`` and examples with ``examples/numeric/PCATransformer.ipynb`` `#57 <https://github.com/lvgig/tubular/pull/57>`_

Fixed
^^^^^
- updated black version to 22.3.0 and flake8 version to 5.0.4 to fix compatibility issues `#45 <https://github.com/lvgig/tubular/pull/45>`_

- removed kwargs argument from BaseTransfomer in ``tubular/base.py`` to avoid silent erroring if incorrect arguments passed to transformers. Fixed a few tests which were revealed to have incorrect arguments passed by change `#56 <https://github.com/lvgig/tubular/pull/56>`_ 


0.3.2 (2022-01-13)
------------------

Added
^^^^^
- Added InteractionTransformer in ``tubular/numeric.py`` , associated tests with ``tests/numeric/test_InteractionTransformer.py`` file and examples with ``examples/numeric/InteractionTransformer.ipynb`` file.`#38 <https://github.com/lvgig/tubular/pull/38>`_


0.3.1 (2021-11-09)
------------------

Added
^^^^^
- Added ``tests/test_transformers.py`` file with test to be applied all transformers `#30 <https://github.com/lvgig/tubular/pull/30>`_

Changed
^^^^^^^
- Set min ``pandas`` version to 1.0.0 in ``requirements.txt``, ``requirements-dev.txt``, and ``docs/requirements.txt`` `#31 <https://github.com/lvgig/tubular/pull/31>`_
- Changed ``y`` argument in fit to only accept ``pd.Series`` objects `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Added new ``_combine_X_y`` method to ``BaseTransformer`` which cbinds X and y `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated ``MeanResponseTransformer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated ``OrdinalEncoderTransformer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated ``NearestMeanResponseImputer`` to use ``y`` arg in ``fit`` and remove setting ``response_column`` in init `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated version of ``black`` used in the ``pre-commit-config`` to ``21.9b0`` `#25 <https://github.com/lvgig/tubular/pull/25>`_
- Modified ``DataFrameMethodTransformer`` to add the possibility of drop original columns `#24 <https://github.com/lvgig/tubular/pull/24>`_

Fixed
^^^^^
- Added attributes to date and numeric transformers to allow transformer to be printed `#30 <https://github.com/lvgig/tubular/pull/30>`_
- Removed copy of mappings in ``MappingTransformer`` to allow transformer to work with sklearn.base.clone `#30 <https://github.com/lvgig/tubular/pull/30>`_
- Changed data values used in some tests for ``MeanResponseTransformer`` so the test no longer depends on pandas <1.3.0 or >=1.3.0, required due to `change <https://pandas.pydata.org/docs/whatsnew/v1.3.0.html#float-result-for-groupby-mean-groupby-median-and-groupby-var>`_ `#25 <https://github.com/lvgig/tubular/pull/25>`_  in pandas behaviour with groupby mean
- ``BaseTransformer`` now correctly raises ``TypeError`` exceptions instead of ``ValueError`` when input values are the wrong type `#26 <https://github.com/lvgig/tubular/pull/26>`_
- Updated version of ``black`` used in the ``pre-commit-config`` to ``21.9b0`` `#25 <https://github.com/lvgig/tubular/pull/25>`_

Removed
^^^^^^^
- Removed ``pytest`` and ``pytest-mock`` from ``requirements.txt`` `#31 <https://github.com/lvgig/tubular/pull/31>`_

0.3.0 (2021-11-03)
------------------

Added
^^^^^
- Added ``scaler_kwargs`` as an empty attribute to the ``ScalingTransformer`` class to avoid an ``AttributeError`` raised by ``sklearn`` `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Added ``test-aide`` package to ``requirements-dev.txt`` `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Added logo for the package `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added ``pre-commit`` to the project to manage pre-commit hooks `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added `quick-start guide <https://tubular.readthedocs.io/en/latest/quick-start.html>`_ to docs `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added `code of conduct <https://tubular.readthedocs.io/en/latest/code-of-conduct.html>`_ for the project `#22 <https://github.com/lvgig/tubular/pull/22>`_

Changed
^^^^^^^
- Moved ``testing/test_data.py`` to ``tests`` folder `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Updated example notebooks to use California housing dataset from sklearn instead of Boston house prices dataset `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Changed ``changelog`` to be ``rst`` format and a changelog page added to docs `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Changed the default branch in the repository from ``master`` to ``main``

Removed
^^^^^^^
- Removed `testing` module and updated tests to use helpers from `test-aide` package `#21 <https://github.com/lvgig/tubular/pull/21>`_

0.2.15 (2021-10-06)
-------------------

Added
^^^^^
- Add github action to run pytest, flake8, black and bandit `#10 <https://github.com/lvgig/tubular/pull/10>`_

Changed
^^^^^^^
- Modified ``GroupRareLevelsTransformer`` to remove the constraint type of ``rare_level_name`` being string, instead it must be the same type as the columns selected `#13 <https://github.com/lvgig/tubular/pull/13>`_
- Fix failing ``NullIndicator.transform`` tests `#14 <https://github.com/lvgig/tubular/pull/14>`_

Removed
^^^^^^^
- Update ``NearestMeanResponseImputer`` to remove fallback to median imputation when no nulls present in a column `#10 <https://github.com/lvgig/tubular/pull/10>`_

0.2.14 (2021-04-23)
-------------------

Added
^^^^^
- Open source release of the package on Github
