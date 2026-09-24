"""Exceptions and warnings raised by otiio."""


class OtiioError(Exception):
    """Base class for errors raised while reading an Otii project."""


class ProjectNotFoundError(OtiioError, FileNotFoundError):
    """The path does not point at an Otii 3 project."""


class MissingDataError(OtiioError, FileNotFoundError):
    """A data blob referenced by project.json is missing on disk."""


class AmbiguousNameError(OtiioError, KeyError):
    """A name matches more than one object; look it up by id (or device) instead."""

    def __str__(self) -> str:
        # KeyError quotes its argument; show the message as-is.
        return str(self.args[0]) if self.args else ""


class AlignmentError(OtiioError, ValueError):
    """Two channels cannot be combined sample-by-sample."""


class UnsupportedFormatWarning(UserWarning):
    """The project was saved with a project_format otiio has not been verified against."""
