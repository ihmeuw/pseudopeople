from dataclasses import dataclass


@dataclass
class ConfigurationError(Exception):
    """Base class for configuration errors"""

    message: str


@dataclass
class DataSourceError(Exception):
    """Base class for data source errors"""

    message: str
