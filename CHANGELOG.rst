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

0.3.6 (unreleased)
------------------

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
