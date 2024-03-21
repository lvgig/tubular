# tests to apply to all columns str or list transformers
import re

import pytest

from tests.base_tests import GenericInitTests


class GenericCappingInitTests(GenericInitTests):
    """
    Tests for init() behaviour specific to when a transformer reads columns from a dict of capping values.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("non_string", [1, True, None])
    def test_columns_list_element_error(
        self,
        non_string,
        minimal_attribute_dict,
        uninitialized_transformers,
    ):
        """Test an error is raised if columns list contains non-string elements."""

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["capping"][non_string] = {1: 2, 3: 4}

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: each element of columns should be a single (string) column name",
            ),
        ):
            uninitialized_transformers[self.transformer_name](**args)
