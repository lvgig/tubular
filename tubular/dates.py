"""This module contains a transformer that applies capping to numeric columns."""

from __future__ import annotations

import datetime
import itertools
import warnings

import numpy as np
import pandas as pd

from tubular.base import BaseTransformer


class BaseDateTransformer(BaseTransformer):
    """
    Transformer with date data checks needed by all transformers taking dates as input data.
    """

    def _generate_is_datetime64_dict(self, X: pd.DataFrame) -> None:
        """
        Function to generate a dictionary attribute storing the result of pd.api.types.is_datetime64_any_dtype()
        for each column in self.columns to avoid repeated calls to this function.
        """

        self._is_datetime64_dict = {}

        for col in self.columns:
            self._is_datetime64_dict[col] = pd.api.types.is_datetime64_any_dtype(X[col])

    def check_columns_are_date_or_datetime(self, X: pd.DataFrame) -> None:
        "Raise a type error if a column to be operated on is not a datetime.datetime or datetime.date object"

        self._generate_is_datetime64_dict(X)

        for col in self.columns:
            if (
                not self._is_datetime64_dict[col]
                and pd.api.types.infer_dtype(X[col]) != "date"
            ):
                msg = f"{self.classname()}: {col} should be datetime64 or date type but got {X[col].dtype}"
                raise TypeError(msg)

    def cast_columns_to_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Check whether columns are datetime64 or date type. If not, an error will be raised. If one is datetime.date
        it will be cast to datetime64.
        """

        self.check_columns_are_date_or_datetime(X)

        temp = X[self.columns].copy()

        for column in self.columns:
            if not self._is_datetime64_dict[column]:
                temp[column] = pd.to_datetime(X[column])

                warnings.warn(
                    f"""
                    {self.classname()}: temporarily cast {column} from datetime64 to date before transforming in order to apply the datetime method.

                    This will artificially increase the precision of each data point in the column. Original column not changed.
                    """,
                    stacklevel=2,
                )

        return temp

    def _cast_non_matching_columns(
        self,
        temp: pd.DataFrame,
        column_A_name: str,
        column_B_name: str,
    ) -> pd.DataFrame:
        """
        Helper function that asymetrically compares column A to column B and casts column B to match column A. This will need calling twice.

        If any casting is done a user warning is raised.

        Operation is:
        - check column A is not datetime 64 (assumed that only other option is 'date' type based on prior checks)
        - check column B is datetime 64
        - if both of the above are true, cast column B to 'date'
        """

        if (
            not self._is_datetime64_dict[column_A_name]
            and self._is_datetime64_dict[column_B_name]
        ):
            temp[column_B_name] = temp[column_B_name].apply(lambda x: x.date())

            warnings.warn(
                f"""
                {self.classname()}: temporarily cast {column_B_name} from datetime64 to date before transforming in order to match {column_A_name}.

                Some precision may be lost from {column_B_name}. Original column not changed.
                """,
                stacklevel=2,
            )

        return temp

    def match_column_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Check the dtype of the two columns to be compared by the transformer. If one is datetime.date and one is
        datetime.datetime, the datetime.datetime column will be cast to datetime.date.

        The casting is done this way to avoid artificially creating precision we don't have by going from a date to a datetime object
        """
        if len(self.columns) != 2:
            msg = f"{self.classname}: match_column_dtypes() called expecting 2 columns but got {len(self.columns)}"
            raise ValueError(msg)

        self.check_columns_are_date_or_datetime(X)

        temp = X[self.columns].copy()

        for column_one_name, column_two_name in itertools.permutations(self.columns):

            temp = self._cast_non_matching_columns(
                temp,
                column_one_name,
                column_two_name,
            )

        return temp


class DateDiffLeapYearTransformer(BaseDateTransformer):
    """Transformer to calculate the number of years between two dates.

    Parameters
    ----------
    column_lower : str
        Name of date column to subtract.

    column_upper : str
        Name of date column to subtract from.

    new_column_name : str
        Name for the new year column.

    drop_cols : bool
        Flag for whether to drop the original columns.

    missing_replacement : int/float/str
        Value to output if either the lower date value or the upper date value are
        missing. Default value is None.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    column_lower : str
        Name of date column to subtract. This attribute is not for use in any method,
        use 'columns' instead. Here only as a fix to allow string representation of transformer.

    column_upper : str
        Name of date column to subtract from. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : list
        List containing column names for transformation in format [column_lower, column_upper]

    new_column_name : str, default = None
        Name given to calculated datediff column. If None then {column_upper}_{column_lower}_datediff
        will be used.

    drop_cols : bool
        Indicator whether to drop old columns during transform method.

    """

    def __init__(
        self,
        column_lower: str,
        column_upper: str,
        drop_cols: bool,
        new_column_name: str | None = None,
        missing_replacement: int | float | str | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if not isinstance(column_lower, str):
            msg = f"{self.classname()}: column_lower should be a str"
            raise TypeError(msg)

        if not isinstance(column_upper, str):
            msg = f"{self.classname()}: column_upper should be a str"
            raise TypeError(msg)

        if new_column_name is not None:
            if not isinstance(new_column_name, str):
                msg = f"{self.classname()}: new_column_name should be a str"
                raise TypeError(msg)

            self.new_column_name = new_column_name

        else:
            self.new_column_name = f"{column_upper}_{column_lower}_datediff"

        if not isinstance(drop_cols, bool):
            msg = f"{self.classname()}: drop_cols should be a bool"
            raise TypeError(msg)

        if (missing_replacement) and (
            type(missing_replacement) not in [int, float, str]
        ):
            msg = f"{self.classname()}: if not None, missing_replacement should be an int, float or string"
            raise TypeError(msg)

        super().__init__(columns=[column_lower, column_upper], **kwargs)

        self.drop_cols = drop_cols
        self.missing_replacement = missing_replacement

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = column_lower
        self.column_upper = column_upper

    def calculate_age(self, row: pd.Series) -> int:
        """Function to calculate age from two date columns in a pd.DataFrame.

        This function, although slower than the np.timedelta64 solution (or something
        similar), accounts for leap years to accurately calculate age for all values.

        Parameters
        ----------
        row : pd.Series
            Named pandas series, with lower_date_name and upper_date_name as index values.

        Returns
        -------
        age : int
            Year gap between the upper and lower date values passes.

        """
        if not isinstance(row, pd.Series):
            msg = f"{self.classname()}: row should be a pd.Series"
            raise TypeError(msg)

        if (pd.isna(row[self.columns[0]])) or (pd.isna(row[self.columns[1]])):
            return self.missing_replacement

        age = row[self.columns[1]].year - row[self.columns[0]].year

        if age > 0:
            if (row[self.columns[1]].month, row[self.columns[1]].day) < (
                row[self.columns[0]].month,
                row[self.columns[0]].day,
            ):
                age += -1
        elif age < 0 and (row[self.columns[1]].month, row[self.columns[1]].day) > (
            row[self.columns[0]].month,
            row[self.columns[0]].day,
        ):
            age += 1

        return age

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate year gap between the two provided columns.

        New column is created under the 'new_column_name', and optionally removes the
        old date columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing column_upper and column_lower.

        Returns
        -------
        X : pd.DataFrame
            Transformed data with new_column_name column.

        """

        self.check_columns_are_date_or_datetime(X)

        X = super().transform(X)

        X[self.new_column_name] = X.apply(lambda x: self.calculate_age(x), axis=1)

        if self.drop_cols:
            X = X.drop(self.columns, axis=1)

        return X


class DateDifferenceTransformer(BaseDateTransformer):
    """Class to transform calculate the difference between 2 date fields in specified units.

    Parameters
    ----------
    column_lower : str
        Name of first date column, difference will be calculated as column_upper - column_lower
    column_upper : str
        Name of second date column, difference will be calculated as column_upper - column_lower
    new_column_name : str, default = None
        Name given to calculated datediff column. If None then {column_upper}_{column_lower}_datediff_{units}
        will be used.
    units : str, default = 'D'
        Numpy datetime units, accepted values are 'D', 'h', 'm', 's'
    copy : bool, default = True
        Should X be copied prior to transform?
    verbose: bool, default = False
    """

    def __init__(
        self,
        column_lower: str,
        column_upper: str,
        new_column_name: str | None = None,
        units: str = "D",
        copy: bool = True,
        verbose: bool = False,
    ) -> None:
        if type(column_lower) is not str:
            msg = f"{self.classname()}: column_lower must be a str"
            raise TypeError(msg)

        if type(column_upper) is not str:
            msg = f"{self.classname()}: column_upper must be a str"
            raise TypeError(msg)

        columns = [column_lower, column_upper]

        accepted_values_units = [
            "D",
            "h",
            "m",
            "s",
        ]

        if type(units) is not str:
            msg = f"{self.classname()}: units must be a str"
            raise TypeError(msg)

        if units not in accepted_values_units:
            msg = f"{self.classname()}: units must be one of {accepted_values_units}, got {units}"
            raise ValueError(msg)

        self.units = units

        if new_column_name is not None:
            if type(new_column_name) is not str:
                msg = f"{self.classname()}: new_column_name must be a str"
                raise TypeError(msg)

            self.new_column_name = new_column_name

        else:
            self.new_column_name = f"{column_upper}_{column_lower}_datediff_{units}"

        super().__init__(columns=columns, copy=copy, verbose=verbose)

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = column_lower
        self.column_upper = column_upper

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate the difference between the given fields in the specified units.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        """

        X = super().transform(X)

        temp = self.match_column_dtypes(X)

        X[self.new_column_name] = (
            temp[self.columns[1]] - temp[self.columns[0]]
        ) / np.timedelta64(1, self.units)

        return X


class ToDatetimeTransformer(BaseTransformer):
    """Class to transform convert specified column to datetime.

    Class simply uses the pd.to_datetime method on the specified column.

    Parameters
    ----------
    column : str
        Name of the column to convert to datetime.

    new_column_name : str
        Name given to the new datetime column.

    to_datetime_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.to_datetime method when it is called in transform.

    **kwargs
        Arbitrary keyword arguments passed onto pd.to_datetime().

    """

    def __init__(
        self,
        column: str,
        new_column_name: str,
        to_datetime_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if type(column) is not str:
            msg = f"{self.classname()}: column should be a single str giving the column to transform to datetime"
            raise TypeError(msg)

        if type(new_column_name) is not str:
            msg = f"{self.classname()}: new_column_name must be a str"
            raise TypeError(msg)

        if to_datetime_kwargs is None:
            to_datetime_kwargs = {}
        else:
            if type(to_datetime_kwargs) is not dict:
                msg = f"{self.classname()}: to_datetime_kwargs should be a dict but got type {type(to_datetime_kwargs)}"
                raise TypeError(msg)

            for i, k in enumerate(to_datetime_kwargs.keys()):
                if type(k) is not str:
                    msg = f"{self.classname()}: unexpected type ({type(k)}) for to_datetime_kwargs key in position {i}, must be str"
                    raise TypeError(msg)

        self.to_datetime_kwargs = to_datetime_kwargs
        self.new_column_name = new_column_name

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

        super().__init__(columns=[column], **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert specified column to datetime using pd.to_datetime.

        Parameters
        ----------
        X : pd.DataFrame
            Data with column to transform.

        """
        X = super().transform(X)

        X[self.new_column_name] = pd.to_datetime(
            X[self.columns[0]],
            **self.to_datetime_kwargs,
        )

        return X


class SeriesDtMethodTransformer(BaseDateTransformer):
    """Tranformer that applies a pandas.Series.dt method.

    Transformer assigns the output of the method to a new column. It is possible to
    supply other key word arguments to the transform method, which will be passed to the
    pandas.Series.dt method being called.

    Be aware it is possible to supply incompatible arguments to init that will only be
    identified when transform is run. This is because there are many combinations of method, input
    and output sizes. Additionally some methods may only work as expected when called in
    transform with specific key word arguments.

    Parameters
    ----------
    new_column_name : str
        The name of the column to be assigned to the output of running the pandas method in transform.

    pd_method_name : str
        The name of the pandas.Series.dt method to call.

    column : str
        Column to apply the transformer to. If a str is passed this is put into a list. Value passed
        in columns is saved in the columns attribute on the object. Note this has no default value so
        the user has to specify the columns when initialising the transformer. This is avoid likely
        when the user forget to set columns, in this case all columns would be picked up when super
        transform runs.

    pd_method_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.Series.dt method when it is called.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    column : str
        Name of column to apply transformer to. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : str
        Column name for transformation.

    new_column_name : str
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    pd_method_name : str
        The name of the pandas.DataFrame method to call.

    pd_method_kwargs : dict
        Dictionary of keyword arguments to call the pd.Series.dt method with.

    """

    def __init__(
        self,
        new_column_name: str,
        pd_method_name: str,
        column: str,
        pd_method_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if type(column) is not str:
            msg = f"{self.classname()}: column should be a str but got {type(column)}"
            raise TypeError(msg)

        super().__init__(columns=column, **kwargs)

        if type(new_column_name) is not str:
            msg = f"{self.classname()}: unexpected type ({type(new_column_name)}) for new_column_name, must be str"
            raise TypeError(msg)

        if type(pd_method_name) is not str:
            msg = f"{self.classname()}: unexpected type ({type(pd_method_name)}) for pd_method_name, expecting str"
            raise TypeError(msg)

        if pd_method_kwargs is None:
            pd_method_kwargs = {}
        else:
            if type(pd_method_kwargs) is not dict:
                msg = f"{self.classname()}: pd_method_kwargs should be a dict but got type {type(pd_method_kwargs)}"
                raise TypeError(msg)

            for i, k in enumerate(pd_method_kwargs.keys()):
                if type(k) is not str:
                    msg = f"{self.classname()}: unexpected type ({type(k)}) for pd_method_kwargs key in position {i}, must be str"
                    raise TypeError(msg)

        self.new_column_name = new_column_name
        self.pd_method_name = pd_method_name
        self.pd_method_kwargs = pd_method_kwargs

        try:
            ser = pd.Series(
                [datetime.datetime(2020, 12, 21, tzinfo=datetime.timezone.utc)],
            )
            getattr(ser.dt, pd_method_name)

        except Exception as err:
            msg = f'{self.classname()}: error accessing "dt.{pd_method_name}" method on pd.Series object - pd_method_name should be a pd.Series.dt method'
            raise AttributeError(msg) from err

        if callable(getattr(ser.dt, pd_method_name)):
            self._callable = True

        else:
            self._callable = False

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform specific column on input pandas.DataFrame (X) using the given pandas.Series.dt method and
        assign the output back to column in X.

        Any keyword arguments set in the pd_method_kwargs attribute are passed onto the pd.Series.dt method
        when calling it.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. These contain the output of
            running the pd.Series.dt method.

        """
        X = super().transform(X)

        temp = self.cast_columns_to_datetime(X)

        if self._callable:
            X[self.new_column_name] = getattr(
                temp[self.columns[0]].dt,
                self.pd_method_name,
            )(**self.pd_method_kwargs)

        else:
            X[self.new_column_name] = getattr(
                temp[self.columns[0]].dt,
                self.pd_method_name,
            )

        return X


class BetweenDatesTransformer(BaseDateTransformer):
    """Transformer to generate a boolean column indicating if one date is between two others.

    If not all column_lower values are less than or equal to column_upper when transform is run
    then a warning will be raised.

    Parameters
    ----------
    column_lower : str
        Name of column containing the lower date range values.

    column_between : str
        Name of column to check if it's values fall between column_lower and column_upper.

    column_upper : str
        Name of column containing the upper date range values.

    new_column_name : str
        Name for new column to be added to X.

    lower_inclusive : bool, defualt = True
        If lower_inclusive is True the comparison to column_lower will be column_lower <=
        column_between, otherwise the comparison will be column_lower < column_between.

    upper_inclusive : bool, defualt = True
        If upper_inclusive is True the comparison to column_upper will be column_between <=
        column_upper, otherwise the comparison will be column_between < column_upper.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    column_lower : str
        Name of date column to subtract. This attribute is not for use in any method,
        use 'columns' instead. Here only as a fix to allow string representation of transformer.

    column_upper : str
        Name of date column to subtract from. This attribute is not for use in any method,
        use 'columns instead. Here only as a fix to allow string representation of transformer.

    column_between : str
        Name of column to check if it's values fall between column_lower and column_upper. This attribute
        is not for use in any method, use 'columns instead. Here only as a fix to allow string representation of transformer.

    columns : list
        Contains the names of the columns to compare in the order [column_lower, column_between
        column_upper].

    new_column_name : str
        new_column_name argument passed when initialising the transformer.

    lower_inclusive : bool
        lower_inclusive argument passed when initialising the transformer.

    upper_inclusive : bool
        upper_inclusive argument passed when initialising the transformer.

    """

    def __init__(
        self,
        column_lower: str,
        column_between: str,
        column_upper: str,
        new_column_name: str,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
        **kwargs: dict[str, bool],
    ) -> None:
        if type(column_lower) is not str:
            msg = f"{self.classname()}: column_lower should be str"
            raise TypeError(msg)

        if type(column_between) is not str:
            msg = f"{self.classname()}: column_between should be str"
            raise TypeError(msg)

        if type(column_upper) is not str:
            msg = f"{self.classname()}: column_upper should be str"
            raise TypeError(msg)

        if type(new_column_name) is not str:
            msg = f"{self.classname()}: new_column_name should be str"
            raise TypeError(msg)

        if type(lower_inclusive) is not bool:
            msg = f"{self.classname()}: lower_inclusive should be a bool"
            raise TypeError(msg)

        if type(upper_inclusive) is not bool:
            msg = f"{self.classname()}: upper_inclusive should be a bool"
            raise TypeError(msg)

        self.new_column_name = new_column_name
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive

        super().__init__(columns=[column_lower, column_between, column_upper], **kwargs)

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column_lower = column_lower
        self.column_upper = column_upper
        self.column_between = column_between

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform - creates column indicating if middle date is between the other two.

        If not all column_lower values are less than or equal to column_upper when transform is run
        then a warning will be raised.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column (self.new_column_name) added. This column is
            boolean and indicates if the middle column is between the other 2.

        """
        X = super().transform(X)

        self.check_columns_are_date_or_datetime(X)

        if not (X[self.columns[0]] <= X[self.columns[2]]).all():
            warnings.warn(
                f"{self.classname()}: not all {self.columns[2]} are greater than or equal to {self.columns[0]}",
                stacklevel=2,
            )

        if self.lower_inclusive:
            lower_comparison = X[self.columns[0]] <= X[self.columns[1]]

        else:
            lower_comparison = X[self.columns[0]] < X[self.columns[1]]

        if self.upper_inclusive:
            upper_comparison = X[self.columns[1]] <= X[self.columns[2]]

        else:
            upper_comparison = X[self.columns[1]] < X[self.columns[2]]

        X[self.new_column_name] = lower_comparison & upper_comparison

        return X


class DatetimeInfoExtractor(BaseDateTransformer):
    """Transformer to extract various features from datetime var.

    Parameters
    ----------
    columns : str or list
        datetime columns to extract information from

    include : list of str, default = ["timeofday", "timeofmonth", "timeofyear", "dayofweek"]
        Which datetime categorical information to extract

    datetime_mappings : dict, default = {}
        Optional argument to define custom mappings for datetime values.
        Keys of the dictionary must be contained in `include`
        All possible values of each feature must be included in the mappings,
        ie, a mapping for `dayofweek` must include all values 0-6;
        datetime_mappings = {"dayofweek": {"week": [0, 1, 2, 3, 4],
                                           "weekend": [5, 6]}}
        The values for the mapping array must be iterable;
        datetime_mappings = {"timeofday": {"am": range(0, 12),
                                           "pm": range(12, 24)}}
        The required ranges for each mapping are:
            timeofday: 0-23
            timeofmonth: 1-31
            timeofyear: 1-12
            dayofweek: 0-6

        If in include but no mappings provided default values will be used as follows:
           timeofday_mapping = {
                "night": range(0, 6),  # Midnight - 6am
                "morning": range(6, 12),  # 6am - Noon
                "afternoon": range(12, 18),  # Noon - 6pm
                "evening": range(18, 24),  # 6pm - Midnight
            }
            timeofmonth_mapping = {
                "start": range(0, 11),
                "middle": range(11, 21),
                "end": range(21, 32),
            }
            timeofyear_mapping = {
                "spring": range(3, 6),  # Mar, Apr, May
                "summer": range(6, 9),  # Jun, Jul, Aug
                "autumn": range(9, 12),  # Sep, Oct, Nov
                "winter": [12, 1, 2],  # Dec, Jan, Feb
            }
            dayofweek_mapping = {
                "monday": [0],
                "tuesday": [1],
                "wednesday": [2],
                "thursday": [3],
                "friday": [4],
                "saturday": [5],
                "sunday": [6],
            }


    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    include : list of str, default = ["timeofday", "timeofmonth", "timeofyear", "dayofweek"]
        Which datetime categorical information to extract

    datetime_mappings : dict, default = None
        Optional argument to define custom mappings for datetime values.

    """

    def __init__(
        self,
        columns: str | list[str],
        include: str | list[str] | None = None,
        datetime_mappings: dict[
            str,
        ]
        | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if include is None:
            include = ["timeofday", "timeofmonth", "timeofyear", "dayofweek"]
        else:
            if type(include) is not list:
                msg = f"{self.classname()}: include should be List"
                raise TypeError(msg)

        if datetime_mappings is None:
            datetime_mappings = {}
        else:
            if type(datetime_mappings) is not dict:
                msg = f"{self.classname()}: datetime_mappings should be Dict"
                raise TypeError(msg)

        super().__init__(columns=columns, **kwargs)

        for var in include:
            if var not in [
                "timeofday",
                "timeofmonth",
                "timeofyear",
                "dayofweek",
            ]:
                msg = f'{self.classname()}: elements in include should be in ["timeofday", "timeofmonth", "timeofyear", "dayofweek"]'
                raise ValueError(msg)

        if datetime_mappings != {}:
            for key, mapping in datetime_mappings.items():
                if type(mapping) is not dict:
                    msg = f"{self.classname()}: values in datetime_mappings should be dict"
                    raise TypeError(msg)
                if key not in include:
                    msg = f"{self.classname()}: keys in datetime_mappings should be in include"
                    raise ValueError(msg)

        self.include = include
        self.datetime_mappings = datetime_mappings
        self.mappings_provided = self.datetime_mappings.keys()

        # Select correct mapping either from default or user input

        if ("timeofday" in include) and ("timeofday" in self.mappings_provided):
            timeofday_mapping = self.datetime_mappings["timeofday"]
        elif "timeofday" in include:  # Choose default mapping
            timeofday_mapping = {
                "night": range(0, 6),  # Midnight - 6am
                "morning": range(6, 12),  # 6am - Noon
                "afternoon": range(12, 18),  # Noon - 6pm
                "evening": range(18, 24),  # 6pm - Midnight
            }

        if ("timeofmonth" in include) and ("timeofmonth" in self.mappings_provided):
            timeofmonth_mapping = self.datetime_mappings["timeofmonth"]
        elif "timeofmonth" in include:  # Choose default mapping
            timeofmonth_mapping = {
                "start": range(0, 11),
                "middle": range(11, 21),
                "end": range(21, 32),
            }

        if ("timeofyear" in include) and ("timeofyear" in self.mappings_provided):
            timeofyear_mapping = self.datetime_mappings["timeofyear"]
        elif "timeofyear" in include:  # Choose default mapping
            timeofyear_mapping = {
                "spring": range(3, 6),  # Mar, Apr, May
                "summer": range(6, 9),  # Jun, Jul, Aug
                "autumn": range(9, 12),  # Sep, Oct, Nov
                "winter": [12, 1, 2],  # Dec, Jan, Feb
            }

        if ("dayofweek" in include) and ("dayofweek" in self.mappings_provided):
            dayofweek_mapping = self.datetime_mappings["dayofweek"]
        elif "dayofweek" in include:  # Choose default mapping
            dayofweek_mapping = {
                "monday": [0],
                "tuesday": [1],
                "wednesday": [2],
                "thursday": [3],
                "friday": [4],
                "saturday": [5],
                "sunday": [6],
            }

        # Invert dictionaries for quicker lookup

        if "timeofday" in include:
            self.timeofday_mapping = {
                vi: k for k, v in timeofday_mapping.items() for vi in v
            }
            if set(self.timeofday_mapping.keys()) != set(range(24)):
                msg = f"{self.classname()}: timeofday mapping dictionary should contain mapping for all hours between 0-23. {set(range(24)) - set(self.timeofday_mapping.keys())} are missing"
                raise ValueError(msg)

            # Check if all hours in dictionary
        else:
            self.timeofday_mapping = {}

        if "timeofmonth" in include:
            self.timeofmonth_mapping = {
                vi: k for k, v in timeofmonth_mapping.items() for vi in v
            }
            if set(self.timeofmonth_mapping.keys()) != set(range(32)):
                msg = f"{self.classname()}: timeofmonth mapping dictionary should contain mapping for all days between 1-31. {set(range(1, 32)) - set(self.timeofmonth_mapping.keys())} are missing"
                raise ValueError(msg)
        else:
            self.timeofmonth_mapping = {}

        if "timeofyear" in include:
            self.timeofyear_mapping = {
                vi: k for k, v in timeofyear_mapping.items() for vi in v
            }
            if set(self.timeofyear_mapping.keys()) != set(range(1, 13)):
                msg = f"{self.classname()}: timeofyear mapping dictionary should contain mapping for all months between 1-12. {set(range(1, 13)) - set(self.timeofyear_mapping.keys())} are missing"
                raise ValueError(msg)

        else:
            self.timeofyear_mapping = {}

        if "dayofweek" in include:
            self.dayofweek_mapping = {
                vi: k for k, v in dayofweek_mapping.items() for vi in v
            }
            if set(self.dayofweek_mapping.keys()) != set(range(7)):
                msg = f"{self.classname()}: dayofweek mapping dictionary should contain mapping for all days between 0-6. {set(range(7)) - set(self.dayofweek_mapping.keys())} are missing"
                raise ValueError(msg)

        else:
            self.dayofweek_mapping = {}

    def _map_values(self, value: int | float, interval: str) -> str:
        """Method to apply mappings for a specified interval ("timeofday", "timeofmonth", "timeofyear" or "dayofweek")
        from corresponding mapping attribute to a single value.

        Parameters
        ----------
        interval : str
            the time period to map "timeofday", "timeofmonth", "timeofyear" or "dayofweek"

        value : float or int
            the value to be mapped


        Returns
        -------
        str : str
            Mapped value
        """
        if type(value) is not float and type(value) is not int:
            msg = f"{self.classname()}: value should be float or int"
            raise TypeError(msg)

        errors = {
            "timeofday": "0-23",
            "dayofweek": "0-6",
            "timeofmonth": "1-31",
            "timeofyear": "1-12",
        }
        ranges = {
            "timeofday": (0, 24, 1),
            "dayofweek": (0, 7, 1),
            "timeofmonth": (1, 32, 1),
            "timeofyear": (1, 13, 1),
        }
        mappings = {
            "timeofday": self.timeofday_mapping,
            "dayofweek": self.dayofweek_mapping,
            "timeofmonth": self.timeofmonth_mapping,
            "timeofyear": self.timeofyear_mapping,
        }

        if (not np.isnan(value)) and (value not in np.arange(*ranges[interval])):
            msg = f"{self.classname()}: value for {interval} mapping  in self._map_values should be an integer value in {errors[interval]}"
            raise ValueError(msg)

        if np.isnan(value):
            return np.nan

        return mappings[interval][value]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform - Extracts new features from datetime variables.

        Parameters
        ----------
        X : pd.DataFrame
            Data with columns to extract info from.

        Returns
        -------
        X : pd.DataFrame
            Transformed input X with added columns of extracted information.
        """
        X = super().transform(X)

        temp = self.cast_columns_to_datetime(X)

        for col in self.columns:
            if "timeofday" in self.include:
                X[col + "_timeofday"] = temp[col].dt.hour.apply(
                    self._map_values,
                    interval="timeofday",
                )

            if "timeofmonth" in self.include:
                X[col + "_timeofmonth"] = temp[col].dt.day.apply(
                    self._map_values,
                    interval="timeofmonth",
                )

            if "timeofyear" in self.include:
                X[col + "_timeofyear"] = temp[col].dt.month.apply(
                    self._map_values,
                    interval="timeofyear",
                )

            if "dayofweek" in self.include:
                X[col + "_dayofweek"] = temp[col].dt.weekday.apply(
                    self._map_values,
                    interval="dayofweek",
                )

        return X


class DatetimeSinusoidCalculator(BaseDateTransformer):
    """Transformer to derive a feature in a dataframe by calculating the
    sine or cosine of a datetime column in a given unit (e.g hour), with the option to scale
    period of the sine or cosine to match the natural period of the unit (e.g. 24).

    Parameters
    ----------
    columns : str or list
        Columns to take the sine or cosine of. Must be a datetime[64] column.

    method : str or list
        Argument to specify which function is to be calculated. Accepted values are 'sin', 'cos' or a list containing both.

    units : str or dict
        Which time unit the calculation is to be carried out on. Accepted values are 'year', 'month',
        'day', 'hour', 'minute', 'second', 'microsecond'.  Can be a string or a dict containing key-value pairs of column
        name and units to be used for that column.

    period : int, float or dict, default = 2*np.pi
        The period of the output in the units specified above. To leave the period of the sinusoid output as 2 pi, specify 2*np.pi (or leave as default).
        Can be a string or a dict containing key-value pairs of column name and period to be used for that column.

    Attributes
    ----------
    columns : str or list
        Columns to take the sine or cosine of.

    method : str or list
        The function to be calculated; either sin, cos or a list containing both.

    units : str or dict
        Which time unit the calculation is to be carried out on. Will take any of 'year', 'month',
        'day', 'hour', 'minute', 'second', 'microsecond'. Can be a string or a dict containing key-value pairs of column
        name and units to be used for that column.

    period : str, float or dict, default = 2*np.pi
        The period of the output in the units specified above. Can be a string or a dict containing key-value pairs of column
        name and units to be used for that column.
    """

    def __init__(
        self,
        columns: str | list[str],
        method: str | list[str],
        units: str | dict,
        period: int | float | dict = 2 * np.pi,
    ) -> None:
        super().__init__(columns, copy=True)

        if not isinstance(method, str) and not isinstance(method, list):
            msg = "{}: method must be a string or list but got {}".format(
                self.classname(),
                type(method),
            )
            raise TypeError(msg)

        if not isinstance(units, str) and not isinstance(units, dict):
            msg = "{}: units must be a string or dict but got {}".format(
                self.classname(),
                type(units),
            )
            raise TypeError(msg)

        if (
            (not isinstance(period, int))
            and (not isinstance(period, float))
            and (not isinstance(period, dict))
            or (isinstance(period, bool))
        ):
            msg = "{}: period must be an int, float or dict but got {}".format(
                self.classname(),
                type(period),
            )
            raise TypeError(msg)

        if isinstance(units, dict) and (
            not all(isinstance(item, str) for item in list(units.keys()))
            or not all(isinstance(item, str) for item in list(units.values()))
        ):
            msg = "{}: units dictionary key value pair must be strings but got keys: {} and values: {}".format(
                self.classname(),
                {type(k) for k in units},
                {type(v) for v in units.values()},
            )
            raise TypeError(msg)

        if isinstance(period, dict) and (
            not all(isinstance(item, str) for item in list(period.keys()))
            or (
                not all(isinstance(item, int) for item in list(period.values()))
                and not all(isinstance(item, float) for item in list(period.values()))
            )
            or any(isinstance(item, bool) for item in list(period.values()))
        ):
            msg = "{}: period dictionary key value pair must be str:int or str:float but got keys: {} and values: {}".format(
                self.classname(),
                {type(k) for k in period},
                {type(v) for v in period.values()},
            )
            raise TypeError(msg)

        valid_method_list = ["sin", "cos"]

        method_list = [method] if isinstance(method, str) else method

        for method in method_list:
            if method not in valid_method_list:
                msg = '{}: Invalid method {} supplied, should be "sin", "cos" or a list containing both'.format(
                    self.classname(),
                    method,
                )
                raise ValueError(msg)

        valid_unit_list = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
        ]

        if isinstance(units, dict):
            if not set(units.values()).issubset(valid_unit_list):
                msg = "{}: units dictionary values must be one of 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond' but got {}".format(
                    self.classname(),
                    set(units.values()),
                )
                raise ValueError(msg)
        elif units not in valid_unit_list:
            msg = "{}: Invalid units {} supplied, should be in {}".format(
                self.classname(),
                units,
                valid_unit_list,
            )
            raise ValueError(msg)

        self.method = method_list
        self.units = units
        self.period = period

        if isinstance(units, dict) and sorted(units.keys()) != sorted(self.columns):
            msg = "{}: unit dictionary keys must be the same as columns but got {}".format(
                self.classname(),
                set(units.keys()),
            )
            raise ValueError(msg)

        if isinstance(period, dict) and sorted(period.keys()) != sorted(self.columns):
            msg = "{}: period dictionary keys must be the same as columns but got {}".format(
                self.classname(),
                set(period.keys()),
            )
            raise ValueError(msg)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform - creates column containing sine or cosine of another datetime column.

        Which function is used is stored in the self.method attribute.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional columns added, these are named "<method>_<original_column>"
        """
        X = super().transform(X)

        temp = self.cast_columns_to_datetime(X)

        for column in self.columns:
            if not isinstance(self.units, dict):
                column_in_desired_unit = getattr(temp[column].dt, self.units)
                desired_units = self.units
            elif isinstance(self.units, dict):
                column_in_desired_unit = getattr(temp[column].dt, self.units[column])
                desired_units = self.units[column]
            if not isinstance(self.period, dict):
                desired_period = self.period
            elif isinstance(self.period, dict):
                desired_period = self.period[column]

            for method in self.method:
                new_column_name = f"{method}_{desired_period}_{desired_units}_{column}"

                X[new_column_name] = getattr(np, method)(
                    column_in_desired_unit * (2.0 * np.pi / desired_period),
                )

        return X
