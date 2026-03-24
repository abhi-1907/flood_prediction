"""
Data Ingestion Agent package — public API.
"""

from agents.data_ingestion.ingestion_agent import DataIngestionAgent
from agents.data_ingestion.ingestion_schemas import (
    DataCategory,
    DataSourceType,
    FetchResult,
    IngestionResult,
    SchemaReport,
)
from agents.data_ingestion.schema_validator import SchemaValidator
from agents.data_ingestion.source_identifier import SourceIdentifier, IngestionPlan, FetchTask
from agents.data_ingestion.data_merger import DataMerger

__all__ = [
    "DataIngestionAgent",
    # Schemas
    "DataCategory",
    "DataSourceType",
    "FetchResult",
    "IngestionResult",
    "SchemaReport",
    # Core classes
    "SchemaValidator",
    "SourceIdentifier",
    "IngestionPlan",
    "FetchTask",
    "DataMerger",
]
