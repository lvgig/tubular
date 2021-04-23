import tubular.testing.helpers as h

from tubular.base import ReturnKeyDict


def test_inheritance():
    """Test ReturnKeyDict inherits from dict."""

    x = ReturnKeyDict()

    h.assert_inheritance(x, dict)


def test_has___missing___method():
    """Test that ReturnKeyDict has a __missing__ method."""

    x = ReturnKeyDict()

    h.test_object_method(
        x, "__missing__", "ReturnKeyDict does not have __missing__ method"
    )


def test___missing___returns_key():
    """Test that __missing__ returns the passed key."""

    x = ReturnKeyDict()

    k = "a"

    result = x.__missing__(k)

    assert k == result, "passed key not returned from __missing__"


def test_keyerror_not_raised():
    """Test that a key error is not raised, instead key is returned, in attempt to access key not present in dict."""

    d = ReturnKeyDict({"a": 1})

    result = d["b"]

    assert (
        result == "b"
    ), "passed key not returned from when attempting to lookup but not present in dict"
