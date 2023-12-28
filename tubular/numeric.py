"""This module contains transformers that apply numeric functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)

from tubular.base import BaseTransformer, DataFrameMethodTransformer


class LogTransformer(BaseTransformer):
    """Transformer to apply log transformation.

    Transformer has the option to add 1 to the columns to log and drop the
    original columns.

    Parameters
    ----------
    columns : None or str or list
        Columns to log transform.

    base : None or float/int
        Base for log transform. If None uses natural log.

    add_1 : bool
        Should a constant of 1 be added to the columns to be transformed prior to
        applying the log transform?

    drop : bool
        Should the original columns to be transformed be dropped after applying the
        log transform?

    suffix : str, default = '_log'
        The suffix to add onto the end of column names for new columns.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init method.

    Attributes
    ----------
    add_1 : bool
        The name of the column or columns to be assigned to the output of running the
        pandas method in transform.

    drop : bool
        The name of the pandas.DataFrame method to call.

    suffix : str
        The suffix to add onto the end of column names for new columns.

    """

    def __init__(
        self,
        columns: str | list[str] | None,
        base: float | int | None = None,
        add_1: bool = False,
        drop: bool = True,
        suffix: str = "log",
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if base is not None:
            if not isinstance(base, (int, float)):
                msg = f"{self.classname()}: base should be numeric or None"
                raise ValueError(msg)
            if not base > 0:
                msg = f"{self.classname()}: base should be strictly positive"
                raise ValueError(msg)

        self.base = base
        self.add_1 = add_1
        self.drop = drop
        self.suffix = suffix

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log transform to the specified columns.

        If the drop attribute is True then the original columns are dropped. If
        the add_1 attribute is True then the original columns + 1 are logged.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe to be transformed.

        Returns
        -------
        X : pd.DataFrame
            The dataframe with the specified columns logged, optionally dropping the original
            columns if self.drop is True.

        """
        X = super().transform(X)

        numeric_column_types = X[self.columns].apply(
            pd.api.types.is_numeric_dtype,
            axis=0,
        )

        if not numeric_column_types.all():
            non_numeric_columns = list(
                numeric_column_types.loc[~numeric_column_types].index,
            )

            msg = f"{self.classname()}: The following columns are not numeric in X; {non_numeric_columns}"
            raise TypeError(msg)

        new_column_names = [f"{column}_{self.suffix}" for column in self.columns]

        if self.add_1:
            if (X[self.columns] <= -1).sum().sum() > 0:
                msg = f"{self.classname()}: values less than or equal to 0 in columns (after adding 1), make greater than 0 before using transform"
                raise ValueError(msg)

            if self.base is None:
                X[new_column_names] = np.log(X[self.columns] + 1)

            else:
                X[new_column_names] = np.log(X[self.columns] + 1) / np.log(self.base)

        else:
            if (X[self.columns] <= 0).sum().sum() > 0:
                msg = f"{self.classname()}: values less than or equal to 0 in columns, make greater than 0 before using transform"
                raise ValueError(msg)

            if self.base is None:
                X[new_column_names] = np.log(X[self.columns])

            else:
                X[new_column_names] = np.log(X[self.columns]) / np.log(self.base)

        if self.drop:
            X = X.drop(self.columns, axis=1)

        return X


class CutTransformer(BaseTransformer):
    """Class to bin a column into discrete intervals.

    Class simply uses the [pd.cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html)
    method on the specified column.

    Parameters
    ----------
    column : str
        Name of the column to discretise.

    new_column_name : str
        Name given to the new discrete column.

    cut_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the pd.cut method when it is called in transform.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    """

    def __init__(
        self,
        column: str,
        new_column_name: str,
        cut_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if type(column) is not str:
            msg = f"{self.classname()}: column arg (name of column) should be a single str giving the column to discretise"
            raise TypeError(msg)

        if type(new_column_name) is not str:
            msg = f"{self.classname()}: new_column_name must be a str"
            raise TypeError(msg)

        if cut_kwargs is None:
            cut_kwargs = {}
        else:
            if type(cut_kwargs) is not dict:
                msg = f"{self.classname()}: cut_kwargs should be a dict but got type {type(cut_kwargs)}"
                raise TypeError(msg)

        for i, k in enumerate(cut_kwargs.keys()):
            if type(k) is not str:
                msg = f"{self.classname()}: unexpected type ({type(k)}) for cut_kwargs key in position {i}, must be str"
                raise TypeError(msg)

        self.cut_kwargs = cut_kwargs
        self.new_column_name = new_column_name

        # This attribute is not for use in any method, use 'columns' instead.
        # Here only as a fix to allow string representation of transformer.
        self.column = column

        super().__init__(columns=[column], **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Discretise specified column using pd.cut.

        Parameters
        ----------
        X : pd.DataFrame
            Data with column to transform.

        """
        X = super().transform(X)

        if not pd.api.types.is_numeric_dtype(X[self.columns[0]]):
            msg = f"{self.classname()}: {self.columns[0]} should be a numeric dtype but got {X[self.columns[0]].dtype}"
            raise TypeError(msg)

        X[self.new_column_name] = pd.cut(
            X[self.columns[0]].to_numpy(),
            **self.cut_kwargs,
        )

        return X


class TwoColumnOperatorTransformer(DataFrameMethodTransformer):
    """This transformer applies a pandas.DataFrame method to two columns (add, sub, mul, div, mod, pow).

    Transformer assigns the output of the method to a new column. The method will be applied
    in the form (column 1)operator(column 2), so order matters (if the method does not commute). It is possible to
    supply other key word arguments to the transform method, which will be passed to the pandas.DataFrame method being called.

    Parameters
    ----------
    pd_method_name : str
        The name of the pandas.DataFrame method to be called.

    column1_name : str
        The name of the 1st column in the operation.

    column2_name : str
        The name of the 2nd column in the operation.

    new_column_name : str
        The name of the new column that the output is assigned to.

    pd_method_kwargs : dict, default =  {'axis':0}
        Dictionary of method kwargs to be passed to pandas.DataFrame method. Must contain an entry for axis, set to either 1 or 0.

    **kwargs :
        Arbitrary keyword arguments passed onto BaseTransformer.__init__().

    Attributes
    ----------
    pd_method_name : str
        The name of the pandas.DataFrame method to be called.

    columns : list
        list containing two string items: [column1_name, column2_name] The first will be operated upon by the
        chosen pandas method using the second.

    column2_name : str
        The name of the 2nd column in the operation.

    new_column_name : str
        The name of the new column that the output is assigned to.

    pd_method_kwargs : dict
        Dictionary of method kwargs to be passed to pandas.DataFrame method.

    """

    def __init__(
        self,
        pd_method_name: str,
        columns: list[str],
        new_column_name: str,
        pd_method_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        """Performs input checks not done in either DataFrameMethodTransformer.__init__ or BaseTransformer.__init__."""
        if pd_method_kwargs is None:
            pd_method_kwargs = {"axis": 0}
        else:
            if "axis" not in pd_method_kwargs.keys():
                msg = f'{self.classname()}: pd_method_kwargs must contain an entry "axis" set to 0 or 1'
                raise ValueError(msg)
            if pd_method_kwargs["axis"] not in [0, 1]:
                msg = f"{self.classname()}: pd_method_kwargs 'axis' must be 0 or 1"
                raise ValueError(msg)

        if type(columns) is not list and len(columns) != 2:
            msg = f"{self.classname()}: columns must be a list containing two column names but got {columns}"
            raise ValueError(msg)

        self.column1_name = columns[0]
        self.column2_name = columns[1]

        # call DataFrameMethodTransformer.__init__
        # This class will inherit all the below attributes from DataFrameMethodTransformer
        super().__init__(
            new_column_name=new_column_name,
            pd_method_name=pd_method_name,
            columns=columns,
            pd_method_kwargs=pd_method_kwargs,
            **kwargs,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data by applying the chosen method to the two specified columns.

        Args:
        ----
            X (pd.DataFrame): Data to transform.

        Returns:
        -------
            pd.DataFrame: Input X with an additional column.
        """
        # call BaseTransformer.transform
        X = super(DataFrameMethodTransformer, self).transform(X)

        is_numeric = X[self.columns].apply(pd.api.types.is_numeric_dtype, axis=0)

        if not is_numeric.all():
            msg = f"{self.classname()}: input columns in X must contain only numeric values"
            raise TypeError(msg)

        X[self.new_column_name] = getattr(X[[self.column1_name]], self.pd_method_name)(
            X[self.column2_name],
            **self.pd_method_kwargs,
        )

        return X


class ScalingTransformer(BaseTransformer):
    """Transformer to perform scaling of numeric columns.

    Transformer can apply min max scaling, max absolute scaling or standardisation (subtract mean and divide by std).
    The transformer uses the appropriate sklearn.preprocessing scaler.

    Parameters
    ----------
    columns : str, list or None
        Name of the columns to apply scaling to.

    scaler_type : str
        Type of scaler to use, must be one of 'min_max', 'max_abs' or 'standard'. The corresponding
        sklearn.preprocessing scaler used in each case is MinMaxScaler, MaxAbsScaler or StandardScaler.

    scaler_kwargs : dict, default = {}
        A dictionary of keyword arguments to be passed to the scaler object when it is initialised.

    **kwargs
        Arbitrary keyword arguments passed onto BaseTransformer.init().

    """

    def __init__(
        self,
        columns: str | list[str] | None,
        scaler_type: str,
        scaler_kwargs: dict[str, object] | None = None,
        **kwargs: dict[str, bool],
    ) -> None:
        if scaler_kwargs is None:
            scaler_kwargs = {}
        else:
            if type(scaler_kwargs) is not dict:
                msg = f"{self.classname()}: scaler_kwargs should be a dict but got type {type(scaler_kwargs)}"
                raise TypeError(msg)

        for i, k in enumerate(scaler_kwargs.keys()):
            if type(k) is not str:
                msg = f"{self.classname()}: unexpected type ({type(k)}) for scaler_kwargs key in position {i}, must be str"
                raise TypeError(msg)

        allowed_scaler_values = ["min_max", "max_abs", "standard"]

        if scaler_type not in allowed_scaler_values:
            msg = f"{self.classname()}: scaler_type should be one of; {allowed_scaler_values}"
            raise ValueError(msg)

        if scaler_type == "min_max":
            self.scaler = MinMaxScaler(**scaler_kwargs)

        elif scaler_type == "max_abs":
            self.scaler = MaxAbsScaler(**scaler_kwargs)

        elif scaler_type == "standard":
            self.scaler = StandardScaler(**scaler_kwargs)

        # This attribute is not for use in any method
        # Here only as a fix to allow string representation of transformer.
        self.scaler_kwargs = scaler_kwargs
        self.scaler_type = scaler_type

        super().__init__(columns=columns, **kwargs)

    def check_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Method to check all columns (specicifed in self.columns) in X are all numeric.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing columns to check.

        """
        numeric_column_types = X[self.columns].apply(
            pd.api.types.is_numeric_dtype,
            axis=0,
        )

        if not numeric_column_types.all():
            non_numeric_columns = list(
                numeric_column_types.loc[~numeric_column_types].index,
            )

            msg = f"{self.classname()}: The following columns are not numeric in X; {non_numeric_columns}"
            raise TypeError(msg)

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit scaler to input data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        """
        super().fit(X, y)

        X = self.check_numeric_columns(X)

        self.scaler.fit(X[self.columns])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data X with fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe containing columns to be scaled.

        Returns
        -------
        X : pd.DataFrame
            Input X with columns scaled.

        """
        X = super().transform(X)

        X = self.check_numeric_columns(X)

        X[self.columns] = self.scaler.transform(X[self.columns])

        return X


class InteractionTransformer(BaseTransformer):
    """Transformer that generates interaction features.
    Transformer generates a new column  for all combinations from the selected columns up to the maximum degree
    provided. (For sklearn version higher than 1.0.0>, only interaction of a degree higher or equal to the minimum
    degree would be computed).
    Each interaction column consists of the product of the specific combination of columns.
    Ex: with 3 columns provided ["a","b","c"], if max degree is 3, the total possible combinations are :
    - of degree 1 : ["a","b","c"]
    - of degree 2 : ["a b","b c","a c"]
    - of degree 3 : ["a b c"].

    Parameters
    ----------
        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object. Note this has no default value so
            the user has to specify the columns when initialising the transformer. This is avoid likely
            when the user forget to set columns, in this case all columns would be picked up when super
            transform runs.
        min_degree : int
            minimum degree of interaction features to be considered. For example if min_degree=3, only interaction
            columns from at least 3 columns would be generated. NB- only applies if sklearn version is 1.0.0>=
        max_degree : int
            maximum degree of interaction features to be considered. For example if max_degree=3, only interaction
            columns from up to 3 columns would be generated.


    Attributes
    ----------
        min_degree : int
            minimum degree of interaction features to be considered
        max_degree : int
            maximum degree of interaction features to be considered
        nb_features_to_interact : int
            number of selected columns from which interactions should be computed. (=len(columns))
        nb_combinations : int
            number of new interaction features
        interaction_colname : list
            names of each new interaction feature. The name of an interaction feature is the combinations of previous
            column names joined with a whitespace. Interaction feature of ["col1","col2","col3] would be "col1 col2 col3".
        nb_feature_out : int
            number of total columns of transformed dataset, including new interaction features

    """

    def __init__(
        self,
        columns: str | list[str] | None,
        min_degree: int = 2,
        max_degree: int = 2,
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if len(columns) < 2:
            msg = f"{self.classname()}: number of columns must be equal or greater than 2, got {str(len(columns))} column."
            raise ValueError(msg)

        if type(min_degree) is int:
            if min_degree < 2:
                msg = f"{self.classname()}: min_degree must be equal or greater than 2, got {str(min_degree)}"
                raise ValueError(msg)
            self.min_degree = min_degree
        else:
            msg = f"{self.classname()}: unexpected type ({type(min_degree)}) for min_degree, must be int"
            raise TypeError(msg)
        if type(max_degree) is int:
            if min_degree > max_degree:
                msg = f"{self.classname()}: max_degree must be equal or greater than min_degree"
                raise ValueError(msg)
            self.max_degree = max_degree
            if max_degree > len(columns):
                msg = f"{self.classname()}: max_degree must be equal or lower than number of columns"
                raise ValueError(msg)
            self.max_degree = max_degree
            if max_degree > len(columns):
                msg = f"{self.classname()}: max_degree must be equal or lower than number of columns"
                raise ValueError(msg)
            self.max_degree = max_degree
        else:
            msg = f"{self.classname()}: unexpected type ({type(max_degree)}) for max_degree, must be int"
            raise TypeError(msg)

        self.nb_features_to_interact = len(self.columns)
        self.nb_combinations = -1
        self.interaction_colname = []
        self.nb_feature_out = -1

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate from input pandas DataFrame (X) new interaction features using the "product" pandas.DataFrame method
         and add this column or columns in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column or columns (self.interaction_colname) added. These contain the output of
            running the  product pandas DataFrame method on identified combinations.

        """
        X = super().transform(X)

        try:
            interaction_combination_index = PolynomialFeatures._combinations(
                n_features=self.nb_features_to_interact,
                min_degree=self.min_degree,
                max_degree=self.max_degree,
                interaction_only=True,
                include_bias=False,
            )
        except TypeError as err:
            if (
                str(err)
                == "_combinations() got an unexpected keyword argument 'min_degree'"
            ):
                interaction_combination_index = PolynomialFeatures._combinations(
                    n_features=self.nb_features_to_interact,
                    degree=self.max_degree,
                    interaction_only=True,
                    include_bias=False,
                )
            else:
                raise err

        interaction_combination_colname = [
            [self.columns[col_idx] for col_idx in interaction_combination]
            for interaction_combination in interaction_combination_index
        ]
        self.nb_combinations = len(interaction_combination_colname)
        self.nb_feature_out = self.nb_combinations + len(X)

        self.interaction_colname = [
            " ".join(interaction_combination)
            for interaction_combination in interaction_combination_colname
        ]

        for inter_idx in range(len(interaction_combination_colname)):
            X[self.interaction_colname[inter_idx]] = X[
                interaction_combination_colname[inter_idx]
            ].product(axis=1, skipna=False)

        return X


class PCATransformer(BaseTransformer):
    """Transformer that generates variables using Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    It is based on sklearn class sklearn.decomposition.PCA

    Parameters
    ----------
        columns : None or list or str
            Columns to apply the transformer to. If a str is passed this is put into a list. Value passed
            in columns is saved in the columns attribute on the object. Note this has no default value so
            the user has to specify the columns when initialising the transformer. When the user forget to set columns,
            all columns would be picked up when super transform runs.
        n_components : int, float or 'mle', default=None
            Number of components to keep.
            if n_components is not set all components are kept::
                n_components == min(n_samples, n_features)
            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.
            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components.
            If ``svd_solver == 'arpack'``, the number of components must be
             strictly less than the minimum of n_features and n_samples.
            Hence, the None case results in::
                n_components == min(n_samples, n_features) - 1   svd_solver='auto', tol=0.0,  n_oversamples=10, random_state=None
        svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
            If auto :
                The solver is selected by a default policy based on `X.shape` and
                `n_components`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient 'randomized'
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full :
                run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing
            If arpack :
                run SVD truncated to n_components calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                0 < n_components < min(X.shape)
            If randomized :
                run randomized SVD by the method of Halko et al.
            .. sklearn versionadded:: 0.18.0

        random_state : int, RandomState instance or None, default=None
            Used when the 'arpack' or 'randomized' solvers are used. Pass an int
            for reproducible results across multiple function calls.
            .. sklearn versionadded:: 0.18.0
        pca_column_prefix : str, prefix added to each the n components features generated. Default is "pca_"
            example: if n_components = 3, new columns would be 'pca_0','pca_1','pca_2'.

    Attributes
    ----------
    pca : PCA class from sklearn.decomposition
    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.
    feature_names_out: list or None
        list of feature name representing the new dimensions.


    """

    def __init__(
        self,
        columns: str | list[str] | None,
        n_components: int = 2,
        svd_solver: str = "auto",
        random_state: int | np.random.RandomState = None,
        pca_column_prefix: str = "pca_",
        **kwargs: dict[str, bool],
    ) -> None:
        super().__init__(columns=columns, **kwargs)

        if type(n_components) is int:
            if n_components < 1:
                msg = f"{self.classname()}:n_components must be strictly positive got {str(n_components)}"
                raise ValueError(msg)
            self.n_components = n_components
        elif type(n_components) is float:
            if 0 < n_components < 1:
                self.n_components = n_components
            else:
                msg = f"{self.classname()}:n_components must be strictly positive and must be of type int when greater than or equal to 1. Got {str(n_components)}"
                raise ValueError(msg)

        else:
            if n_components == "mle":
                self.n_components = n_components
            else:
                msg = f"{self.classname()}:unexpected type {type(n_components)} for n_components, must be int, float (0-1) or equal to 'mle'."
                raise TypeError(msg)

        if type(svd_solver) is str:
            if svd_solver not in ["auto", "full", "arpack", "randomized"]:
                msg = f"{self.classname()}:svd_solver {svd_solver} is unknown. Please select among 'auto', 'full', 'arpack', 'randomized'."
                raise ValueError(msg)
            self.svd_solver = svd_solver
        else:
            msg = f"{self.classname()}:unexpected type {type(svd_solver)} for svd_solver, must be str"
            raise TypeError(msg)

        if type(random_state) is int or random_state is None:
            self.random_state = random_state
        else:
            msg = f"{self.classname()}:unexpected type {type(random_state)} for random_state, must be int or None."
            raise TypeError(msg)

        if (svd_solver == "arpack") and (n_components == "mle"):
            msg = f"{self.classname()}: n_components='mle' cannot be a string with svd_solver='arpack'"
            raise ValueError(msg)
        if (svd_solver in ["randomized", "arpack"]) and (type(n_components) is float):
            msg = f"{self.classname()}: n_components {n_components} cannot be a float with svd_solver='{svd_solver}'"
            raise TypeError(msg)

        if type(pca_column_prefix) is str:
            self.pca_column_prefix = pca_column_prefix
        else:
            msg = f"{self.classname()}:unexpected type {type(pca_column_prefix)} for pca_column_prefix, must be str"
            raise TypeError(msg)

        self.pca = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )

        self.pca_column_prefix = pca_column_prefix
        self.feature_names_out = None
        self.n_components_ = None

    def check_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Method to check all columns (specicifed in self.columns) in X are all numeric.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing columns to check.

        """
        numeric_column_types = X[self.columns].apply(
            pd.api.types.is_numeric_dtype,
            axis=0,
        )

        if not numeric_column_types.all():
            non_numeric_columns = list(
                numeric_column_types.loc[~numeric_column_types].index,
            )

            msg = f"{self.classname()}: The following columns are not numeric in X; {non_numeric_columns}"
            raise TypeError(msg)

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit PCA to input data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        """
        super().fit(X, y)

        X = self.check_numeric_columns(X)

        if self.n_components != "mle":
            if 0 < self.n_components <= min(X[self.columns].shape):
                pass
            else:
                msg = f"{self.classname()}: n_components {self.n_components} must be between 1 and min(n_samples {X[self.columns].shape[0]}, n_features {X[self.columns].shape[1]}) is {min(X[self.columns].shape)} with svd_solver '{self.svd_solver}'"
                raise ValueError(msg)

        self.pca.fit(X[self.columns])
        self.n_components_ = self.pca.n_components_
        self.feature_names_out = [
            self.pca_column_prefix + str(i) for i in range(self.n_components_)
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate from input pandas DataFrame (X) PCA features and add this column or columns in X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pd.DataFrame
            Input X with additional column or columns (self.interaction_colname) added. These contain the output of
            running the  product pandas DataFrame method on identified combinations.
        """
        X = super().transform(X)
        X = self.check_numeric_columns(X)
        X[self.feature_names_out] = self.pca.transform(X[self.columns])

        return X
