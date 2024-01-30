import re

import pytest

from tests.base_tests import GenericInitTests


class ColumnStrListInitTests(GenericInitTests):
    """
    More tests for BaseTransformer.init() behaviour when a transformer takes columns as string or list.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_columns_empty_list_error(
        self,
        minimal_attribute_dict,
        uninstantiated_transformers,
    ):
        """Test an error is raised if columns is specified as an empty list."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = []

        with pytest.raises(ValueError):
            uninstantiated_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("non_string", [1, True, {"a": 1}, [1, 2], None])
    def test_columns_list_element_error(
        self,
        non_string,
        minimal_attribute_dict,
        uninstantiated_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = [non_string, non_string]

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: each element of columns should be a single (string) column name",
            ),
        ):
            uninstantiated_transformers[self.transformer_name](**args)

    @pytest.mark.parametrize("non_string_or_list", [1, True, {"a": 1}, None])
    def test_columns_non_string_or_list_error(
        self,
        non_string_or_list,
        minimal_attribute_dict,
        uninstantiated_transformers,
    ):
        """Test an error is raised if columns is not passed as a string or list."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["columns"] = non_string_or_list

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: columns must be a string or list with the columns to be pre-processed (if specified)",
            ),
        ):
            uninstantiated_transformers[self.transformer_name](**args)


class ColumnsFromDictInitTests(GenericInitTests):
    """
    More tests for BaseTransformer.init() behaviour when a transformer reads columns from a dict.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("non_string", [1, True, None])
    def test_columns_list_element_error(
        self,
        non_string,
        minimal_attribute_dict,
        uninstantiated_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"][non_string] = {1: 2, 3: 4}

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: each element of columns should be a single (string) column name",
            ),
        ):
            uninstantiated_transformers[self.transformer_name](**args)
