"""Module for defining custom web fields to use on the API interface.
"""
from webargs import fields, validate

from api import parsers


class Accept(fields.String):  # TODO: Adjust or delete
    """Return format for method response.
    TODO: XXXXXX Complete description XXXXXXXXXXX.
    """

    def __init__(self, *, metadata=None, **kwds):
        metadata = metadata or {}
        metadata["description"] = self.__doc__
        metadata["location"] = "headers"
        kwds["validate"] = validate.OneOf(parsers.content_types)
        super().__init__(metadata=metadata, **kwds)


class CustomFileArg(fields.Field):  # TODO: Rename class
    """Custom file to train model or generate predict.
    TODO: XXXXXX Complete description XXXXXXXXXXX.
    """

    def __init__(self, *, metadata=None, **kwds):
        metadata = metadata or {}
        metadata["description"] = self.__doc__
        metadata["type"] = "file"
        metadata["location"] = "form"
        super().__init__(metadata=metadata, **kwds)


class CustomPredArg1(fields.Integer):  # TODO: Rename class
    """Custom prediction argument to generate predictions.
    TODO: XXXXXX Complete description XXXXXXXXXXX.
    """

    def __init__(self, *, metadata=None, **kwds):
        metadata = metadata or {}
        metadata["description"] = self.__doc__
        super().__init__(metadata=metadata, **kwds)


class CustomPredArg2(fields.Integer):  # TODO: Rename class
    """Custom prediction argument to generate predictions.
    TODO: XXXXXX Complete description XXXXXXXXXXX.
    """

    def __init__(self, *, metadata=None, **kwds):
        metadata = metadata or {}
        metadata["description"] = self.__doc__
        super().__init__(metadata=metadata, **kwds)


class CustomTrainArg1(fields.Integer):  # TODO: Rename class
    """Custom train argument to perform model training.
    TODO: XXXXXX Complete description XXXXXXXXXXX.
    """

    def __init__(self, *, metadata=None, **kwds):
        metadata = metadata or {}
        metadata["description"] = self.__doc__
        super().__init__(metadata=metadata, **kwds)


class CustomTrainArg2(fields.Integer):  # TODO: Rename class
    """Custom train argument to perform model training.
    TODO: XXXXXX Complete description XXXXXXXXXXX.
    """

    def __init__(self, *, metadata=None, **kwds):
        metadata = metadata or {}
        metadata["description"] = self.__doc__
        super().__init__(metadata=metadata, **kwds)
