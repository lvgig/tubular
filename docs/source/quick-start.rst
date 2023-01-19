Quick Start
====================
|logo|

Welcome to the quick start guide for tubular!

.. |logo| image:: ../../logo.png
   :height: 200px

Installation
--------------------

The easiest way to get ``tubular`` is to install directly from ``pypi``;

   .. code::

     pip install tubular


    Thanks for installing tubular! We hope you find it useful!

Examples
---------------------------------

There are example notebooks available on `Github <https://github.com/lvgig/tubular/tree/main/examples/>`_ that demonstrate the functionality of each transformer.

To open them in `binder <https://mybinder.org/>`_ click `here <https://mybinder.org/v2/gh/lvgig/tubular/HEAD?labpath=examples>`_. Once binder has loaded, click on the directory button in the side bar to the left and navigate to the notebook of interest.

Transformers summary
---------------------------------

Each of the modules in ``tubular`` contains transformers that deal with a specific type of data or problem.

We are always looking for new functionality to improve the package so if you would like to add a new transformer create a pull request to let us know your idea then have a look at the `contributing guide <https://github.com/lvgig/tubular/blob/main/CONTRIBUTING.md>`_.

Base
^^^^

This module contains the `DataFrameMethodTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.base.DataFrameMethodTransformer.html>`_ which allows any `pandas.DataFrame method <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ to be used in a transformer.

So for example if the user wishes to take the product of some columns they can use this transformer with the `pandas.DataFrame.prod <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.prod.html>`_ method to achieve this.

This transformer saves us from implementing many transformations that are available already in pandas in our package.

Capping
^^^^^^^

This module deals with capping of numeric columns. 

The standard `CappingTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.capping.CappingTransformer.html>`_ can apply capping at min and max values for different columns. 

The standard `OutOfRangeNullTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.capping.OutOfRangeNullTransformer.html>`_ works in a similar way but replaces values outside the cap range with ``null`` values rather than the min or max depending on which side they fall. 

Dates
^^^^^

This module contains transformers to deal with datetime columns.

Date differencing is available - accounting for leap years `DateDiffLeapYearTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.dates.DateDiffLeapYearTransformer.html>`_ or not `DateDifferenceTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.dates.DateDifferenceTransformer.html>`_.

The `BetweenDatesTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.dates.BetweenDatesTransformer.html>`_ calculates if one date falls between two others.

The `ToDatetimeTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.dates.ToDatetimeTransformer.html>`_ converts columns to datetime type.

The `SeriesDtMethodTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.dates.SeriesDtMethodTransformer.html>`_ allows the user to use `pandas.Series.dt <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.html>`_ methods in a similar way to `base.DataFrameMethodTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.base.DataFrameMethodTransformer.html>`_.

The `DatetimeInfoExtractor <https://tubular.readthedocs.io/en/latest/api/tubular.dates.DatetimeInfoExtractor.html>`_ allows the user to extract datetime info such as the time of day or month from a datetime field.

The `DatetimeSinusoidCalculator <https://tubular.readthedocs.io/en/latest/api/tubular.dates.DatetimeSinusoidCalculator.html>`_ derives a feature in a dataframe by calculating the sine or cosine of a datetime column.

Imputers
^^^^^^^^

This module contains standard imputation techniques - mean, median mode as well as `NearestMeanResponseImputer <https://tubular.readthedocs.io/en/feature-version_0_3_0/api/tubular.imputers.NearestMeanResponseImputer.html>`_ which imputes with the value which is closest to the ``null`` values in terms of average response.  All of these support weights.

The `NullIndicator <https://tubular.readthedocs.io/en/feature-version_0_3_0/api/tubular.imputers.NullIndicator.html>`_ is used to create binary indicators of where ``null`` values are present in a column.

Mapping
^^^^^^^

This module contains transformers that deal with explicit mappings of values. 

The `MappingTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.mapping.MappingTransformer.html>`_ deals with standard mapping of one set of values to another. 

The `CrossColumnMappingTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.mapping.CrossColumnMappingTransformer.html>`_, `CrossColumnAddTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.mapping.CrossColumnAddTransformer.html>`_ and `CrossColumnMultiplyTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.mapping.CrossColumnMultiplyTransformer.html>`_ apply mapping, addition or multiplication to values in one column based off values in another.

Misc
^^^^

The misc module contains transformers which do not fit into other categories.

`SetValueTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.misc.SetValueTransformer.html>`_ creates a constant column with arbitrary value.

`SetDtype <https://tubular.readthedocs.io/en/latest/api/tubular.misc.SetDtype.html>`_ allows the user to set the dtype of a column.

Nominal
^^^^^^^

This module contains categorical encoding techniques. 

There are respone encoding techniques such as `MeanResponseTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.nominal.MeanResponseTransformer.html>`_, one hot encoding `OneHotEncodingTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.nominal.OneHotEncodingTransformer.html>`_ and grouping of infrequently occuring levels `GroupRareLevelsTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.nominal.GroupRareLevelsTransformer.html>`_.

`MeanResponseTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.nominal.MeanResponseTransformer.html>`_ also supports regularisation of encodings using a prior.

Numeric
^^^^^^^

This module contains numeric transformations - cut `CutTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.numeric.CutTransformer.html>`_, log `LogTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.numeric.LogTransformer.html>`_, and scaling `ScalingTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.numeric.ScalingTransformer.html>`_.

`TwoColumnOperatorTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.numeric.TwoColumnOperatorTransformer.html>`_ allows a user to apply operations to two colmns using methods from `pandas.DataFrame method <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ which require a multiple columns (e.g. add, subtract, multiply etc

It also contains `InteractionTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.numeric.InteractionTransformer.html>`_ and `PCATransformer <https://tubular.readthedocs.io/en/latest/api/tubular.numeric.PCATransformer.html>`_ which create interaction terms and pca components.

Strings
^^^^^^^

The strings module contains useful transformers for working with strings.  `SeriesStrMethodTransformer <https://tubular.readthedocs.io/en/latest/api/tubular.strings.SeriesStrMethodTransformer.html>`_, allows the user to access `pandas.Series.str <https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html>`_ methods within ``tubular``.  `StringConcatenator <https://tubular.readthedocs.io/en/latest/api/tubular.strings.StringConcatenator.html>`_ allows a user to concatenate multiple columns together of varied dtype into a string output.



Reporting an issue
---------------------------------

If you find an issue or bug in the package please create an `issue <https://github.com/lvgig/tubular/issues>`_ on github.

We really appreciate the time anyone takes to file an issue as this helps us improve the packge.
