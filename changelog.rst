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

0.3.0 (2021-11-03)
------------------

Added
^^^^^
- Added ``scaler_kwargs`` as an empty attribute to the ``ScalingTransformer`` class to avoid an ``AttributeError`` raised by ``sklearn`` `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Added ``test-aide`` package to ``requirements-dev.txt`` `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Added logo for the package `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Added ``pre-commit`` to the project to manage pre-commit hooks `#22 <https://github.com/lvgig/tubular/pull/22>`_

Changed
^^^^^^^
- Moved ``testing/test_data.py`` to ``tests`` folder `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Updated example notebooks to use California housing dataset from sklearn instead of Boston house prices dataset `#21 <https://github.com/lvgig/tubular/pull/21>`_
- Changed ``changelog`` to be ``rst`` format and a changelog page added to docs `#22 <https://github.com/lvgig/tubular/pull/22>`_
- Changed the default branch in the repository from ``master`` to ``main``
- Added quick-start guide to docs `#22 <https://github.com/lvgig/tubular/pull/22>`_

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
