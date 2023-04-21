from dataclasses import dataclass


@dataclass
class ConfigurationError(BaseException):
    """Base class for configuration errors"""

    message: str


@dataclass
class DataSourceError(BaseException):
    """Base class for data source errors"""

    message: str
