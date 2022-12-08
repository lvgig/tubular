[33mcommit 26907843ee3840bc8985103159ae463b82a81921[m[33m ([m[1;36mHEAD -> [m[1;32mfeature/DatetimeInfoExtractor[m[33m)[m
Author: David Hopkinson <davidhopkinson26@gmail.com>
Date:   Thu Dec 8 17:58:01 2022 +0100

    removed import sys logic from examplem notebook

[33mcommit 17613940f92bc9939846525e60cdfeb82e9fc5fd[m[33m ([m[1;31morigin/feature/DatetimeInfoExtractor[m[33m)[m
Author: David Hopkinson <davidhopkinson26@gmail.com>
Date:   Thu Dec 8 16:08:30 2022 +0100

    updated DatetimeInfoExtractor to work with missing data

[33mcommit 21da2b97323b6b13aaa7b5b02e558bcb298b9aaf[m
Author: David Hopkinson <davidhopkinson26@gmail.com>
Date:   Thu Dec 8 15:54:02 2022 +0100

    updated test_transformers to include DatetimeInfoExtractor

[33mcommit 0dc072792a50b20f20cc424fa7a52bf4d7f620f3[m
Author: David Hopkinson <davidhopkinson26@gmail.com>
Date:   Thu Dec 8 15:48:28 2022 +0100

    changed name of transformer to DatetimeInfoExtractor from
    DateTimeInfoExtractor

[33mcommit e278a9e3c83b240b875c7341fc67faddd61e117b[m
Author: David Hopkinson <davidhopkinson26@gmail.com>
Date:   Thu Dec 8 15:42:24 2022 +0100

    added example notebook for DatetimeInfoExtractor.ipynb

[33mcommit 6424877dffe92d4f9fb6da3d1b8f209fe663fd78[m
Author: David Hopkinson <davidhopkinson26@gmail.com>
Date:   Thu Dec 8 15:41:12 2022 +0100

    added DateTimeInfoExtractor transformer

[33mcommit 68686280e8ffc1ce3856d87793f2d32a861bf9f2[m
Author: davidhopkinson26 <89029024+davidhopkinson26@users.noreply.github.com>
Date:   Thu Dec 8 15:08:50 2022 +0100

    prefix errors with transformer classname (#48)
    
    Prefixed all errors with self.classname() call for easier debugging
    
    Added classname() method to BaseTransformer

[33mcommit 9ea6ddf373631c2ec59ee0c0c79c9b37c158790e[m[33m ([m[1;31morigin/feature/datetime_fixes[m[33m, [m[1;32mfeature/datetime_fixes[m[33m)[m
Author: davidhopkinson26 <89029024+davidhopkinson26@users.noreply.github.com>
Date:   Tue Nov 22 13:31:48 2022 +0000

    Fixes #45: updated black version to 22.3.0 to fix compatibility issue with click (#45)
    
    * updated black version to 22.3.0 to fix compatibility issue with click 8.1.0
    
    * updated to latest version of flake8==5.0.4 in requirement-dev.txt
    
    * updated changelog with black and flake8 fixes
    
    Co-authored-by: david hopkinson <davidhopkinson26@gmail.com>

[33mcommit 0c641c9d759c7841c7026f3644605e1543f12389[m
Author: Claire_Fromholz <33227678+ClaireF57@users.noreply.github.com>
Date:   Wed Jan 26 12:33:44 2022 +0100

    Interaction transformer (#38)
    
    * add interaction transfromer
    
    * updated version number to 0.3.2
    
    * add example notebook

[33mcommit d8f75fb9f2b282dad880245a36dbf27a2e0302cb[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Sat Nov 13 07:54:08 2021 +0000

    add memmbers, inherited members and inheritance options to sphinx docs conf (#35)

[33mcommit 2a2892635ce7d1eee3777bb1e97df2c42e5b6cbf[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Sat Nov 13 07:53:50 2021 +0000

    Feature/dsf notebook filled (#34)
    
    * fill demo notebook
    
    * add matplotlib to conda env file
    
    * add plotting functon

[33mcommit 6bf11fdedfca147050dd2c26a8fb18a9dca97911[m
Author: Ned Webster <49588411+nedwebster@users.noreply.github.com>
Date:   Fri Nov 12 20:37:08 2021 +0000

    removed inplace arg from pandas.cat.add_categories() call (#32)

[33mcommit 608fb6d86752bcfd4bda7af713ab9ec11c93c7b7[m
Author: merve-alanyali <86725085+merve-alanyali@users.noreply.github.com>
Date:   Tue Nov 9 17:33:14 2021 +0000

    Feature/version change 0.3.1 (#31)
    
    * Removed pytest and pytest-mock from requirements.txt
    
    * Fixed pandas min version to be 1.0.0 in requirements.txt, requirements-dev.txt and docs/requirements.txt
    
    * Updated version number from 0.3.0 to 0.3.1
    
    * Updated CHANGELOG.rst
    
    Co-authored-by: merve-alanyali <merve.alanyali@lv.co.uk>

[33mcommit f24e55631b707ab3f0b4d052f5bb07c76ae0c1c4[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Tue Nov 9 15:47:55 2021 +0000

    Feature/separate x y (#26)
    
    * allow only pandas Series objects for y
    
    * update tests now fit only accept y as pd.Series
    
    * set dtype explicitly in empty series used in BaseTransformer test to silence warning
    
    * add new _combine_X_y method to BaseTransformer which cbinds X and y
    
    * update MeanResponseTransformer, OrdinalEncoderTransformer, NearestMeanResponseImputer to use y arg in fit and remove setting response_column in init
    
    * update BaseTransformer to raise TypeErrors instead of ValueErrors when inputs are the wrong type
    
    * update changelog

[33mcommit 76caae7498c1868ecd81bedd1e102a96719f578a[m
Author: lsumption <93920684+lsumption@users.noreply.github.com>
Date:   Tue Nov 9 15:03:21 2021 +0000

    Feature/attribute fix (#30)
    
    * add attributes for string representation.
    
    * change scaler parameter name and add attribute.
    
    * add new attributes to tests
    
    * add tests to be run on all transformers
    
    * remove deep copy of mappings so transformer works with sklearn.base.clone
    
    * update changelog
    
    * fix linting and black
    
    Co-authored-by: lsumpiton <lsumption@dssandpit.onmicrosoft.com>

[33mcommit 019e34a7a27a1b863ff4e55d5d4d1f9515d6f757[m
Author: Ned Webster <49588411+nedwebster@users.noreply.github.com>
Date:   Mon Nov 8 18:11:01 2021 +0000

    Updated test to no longer use ordered arg in pd.cut (#28)

[33mcommit 3299623812d54fc7d465284cf271c63ab8795cb5[m
Author: bissoligiulia <88663909+bissoligiulia@users.noreply.github.com>
Date:   Mon Nov 8 16:02:32 2021 +0100

    Dropcolumns (#24)
    
    * added drop_original to DataFrameMethodTransformer
    
    * update test functions
    
    * update changelog

[33mcommit 7544f04e3bac97594dba703e5485c1567a80a67d[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Sun Nov 7 18:05:33 2021 +0000

    add notebook with code to download data + environment file (#27)

[33mcommit 5077e751c44a1992a8d526418cec7bfcc500353b[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Sun Nov 7 17:37:47 2021 +0000

    update MeanResponseTransformer tests so they do not depend on pandas … (#25)
    
    * update MeanResponseTransformer tests so they do not depend on pandas being above or below v1.3.0
    
    * update changelog

[33mcommit 06e16964c6f009b0db39e79d7c15db86dd1afbf7[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Wed Nov 3 22:10:19 2021 +0000

    Other changes for version 0.3.0 (#22)
    
    * increment version number
    
    * increment version number
    
    * remove top level test running script
    
    * remove pre-commit file
    
    * add pre-commit to requirements-dev
    
    * fix versions of bandit and black in requirements-dev
    
    * remove pytest-benchmark in requirements-dev
    
    * add pre commit config file
    
    * remove tests/outputs/ from folder from gitignore now tests.py script removed
    
    * update documentation
    
    * add logo and add to docs
    
    * add white background to logo
    
    * add logo to readme
    
    * reduce size of logo
    
    * reduce size of logo
    
    * update readme text
    
    * format docs conf.py with black
    
    * update makefile for docs
    
    * update readme to use logo from main branch
    
    * add shields to readme
    
    * change changelog from md to rst so it can be used in the docs
    
    * add changelog to docs
    
    * change changelog to caps
    
    * fix typo in changelog filename
    
    * add code of conduct
    
    * fix module name typos in api page in docs
    
    * add small logo to use in docs sidebar
    
    * fix links in code of conduct with rst format
    
    * update changelog with code of conduct
    
    * add bug issue template
    
    * update the contributing guidelines and add to the docs
    
    * add binder badge to readme
    
    * add quick start guide to docs
    
    * fix wrong link to example notebooks in docs
    
    * change ci pipeline to only run on pushes to certain branches, not all
    
    * change references from master branch to main
    
    * remove boston dataset from last test and test data

[33mcommit ab6b422a3c578194f7b495157d817cbb5a1e2f25[m
Author: shreenapatel <shreena88@gmail.com>
Date:   Fri Oct 29 13:27:03 2021 +0100

    Feature/separate test helpers - updated tests to use helpers from test-aide package (#18)
    
    * removed test helpers and changed any references to test_aide
    
    * added step to env set up in build pipeline which installs test-aide from git repo
    
    * updated imports and references to test-aide helpers to reflect new submodules
    
    * changed test_data references to new script added to tests folder
    
    * replaced index_preserved_params and row_by_row_params with new wrapper function from test-aide, adjusted_dataframe_params
    
    * adding test-aide to dev requirements
    
    * updated test-aide submodule names
    
    * updated version number and change log
    
    * removing boston housing dataset from example notebooks
    
    * removing dependencies on functions in test_data.py
    
    * removing boston housing dataset from example notebooks
    
    * removing boston housing dataset from example notebooks
    
    * removing boston housing dataset from example notebooks
    
    * deleting temporary notebook
    
    * adding example notebook changes to changelog
    
    * added scaler_kwargs as empty attribute to scaling transformer to avoid sklearn attribute error
    
    * update nominal module example notebooks to use diabetes dataset from sklearn (#19)
    
    * Feature/update version nos in example notebooks (#20)
    
    * update base example notebooks
    
    * update capping notebooks
    
    * update dates example notebooks
    
    * update imputers example notebooks
    
    * update mappings example notebooks
    
    * update misc example notebook
    
    * update numeric example notebooks
    
    * update strings example notebook
    
    Co-authored-by: Richard Angell <richardangell37@gmail.com>

[33mcommit c690f25bd0c5452320123b763ed3d466dde58f5a[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Mon Oct 4 19:29:48 2021 +0100

    Feature/tidy for next release (#14)
    
    * format with black
    
    * increment version number to 0.2.15
    
    * update list of contributors
    
    * remove setting output column dtype to int32 in test_NullIndicator.py::TestTransform as no longer needed
    
    * add changelog
    
    * add 1.3.0 as the max version of pandas supported with this version

[33mcommit f5416c67cf359f9f282c16924e68bb6bf75061c9[m
Author: bissoligiulia <88663909+bissoligiulia@users.noreply.github.com>
Date:   Sun Oct 3 20:27:38 2021 +0200

    modified GroupRareLevelsTransformer (#13)
    
    * Modified GroupRareLevelsTransformer to remove the constraint type of rare_level_name being string, instead it must be the same type as the columns selected
    
    Co-authored-by: Giulia Bissoli <TH3KW9@Giulias-MacBook-Pro.local>

[33mcommit e61a1b39c0c87338c30a8b878e5b02320fc04c2f[m
Author: Claire_Fromholz <33227678+ClaireF57@users.noreply.github.com>
Date:   Wed Sep 29 21:42:05 2021 +0200

    NearestImputer Update : Remove median imputer when no null (#10)
    
    * update of nearest mean response imputer to remove median imputation fallback feature
    
    * add github action with bandit, flake8, black and pyest steps to run on pushes and PRs to master and develop
    
    * update requirements.txt and correct flake8/black formatting

[33mcommit dcb9a4fedf95e8a0e2593a2e11006397eea96bf6[m
Author: Paul Larsen <munichpavel@gmail.com>
Date:   Sun Jun 27 21:07:33 2021 +0200

    Restore apidoc Makefile command (#9)
    
    Restore apidoc as at commit https://github.com/lvgig/tubular/pull/3/commits/ee68387f7eb068534e4e814dcaa6e4534c3f5727
    BUT keep source/api/tubular.testing.helpers.rst, as it was not on the remove list from PR https://github.com/lvgig/tubular/pull/3
    
    Co-authored-by: Paul Larsen <paul.larsen1@allianz.com>

[33mcommit 53e277dea2cc869702f2ed49f2b495bf79b92355[m
Author: Richard Angell <richardangell37@gmail.com>
Date:   Wed Jun 2 07:22:53 2021 +0000

    Feature/update docs (#5)
    
    * remove some docs files causing some duplication of contents
    
    * remove apidoc section from makefile
    
    * move docs requirements into docs/requirements.txt
    
    * add path setup section to docs conf.py

[33mcommit a38c397b4173ec4a1a05ce94d2b746e3d2009b26[m
Author: Paul Larsen <munichpavel@gmail.com>
Date:   Tue Jun 1 14:09:18 2021 +0200

    Sphinxify api (#3)
    
    * Add sphinx doc files
    
    Changes to the standard quickstart-generated files are
    
    * Put Sphinx build directory in an environment variable (see .envrc.example)
    * Add api autodoc generated files in docs/source/api
    * Use rtd theme
    * Add documentation build instructions to README
    * Remove link to not-yet-existent read-the-docs page
    
    * Add Makefile task for building apidocs
    
    * keep flags from before
    * remove unwanted rst files after creation
    * update README to reflect new usage
    
    Co-authored-by: Paul Larsen <paul.larsen1@allianz.com>
    Co-authored-by: Richard Angell <richardangell37@gmail.com>

[33mcommit 8da9a21a3cdaaac3b66b740f8ef74023e5037f13[m
Author: Paul Larsen <munichpavel@gmail.com>
Date:   Wed May 19 09:40:48 2021 +0200

    Fix markdown linting issues (#1)
    
    * MD025/single-title/single-h1: Multiple top-level headings in the same document
    * MD009/no-trailing-spaces: Trailing spaces

[33mcommit 40855981b14edbf3506d888e35f25b66dfe2dfb9[m
Author: richardangell <richardangell37@gmail.com>
Date:   Fri Apr 23 12:59:08 2021 +0000

    add files after restarting repo
