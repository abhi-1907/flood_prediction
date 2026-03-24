"""
Fetchers sub-package exports.
"""
from agents.data_ingestion.fetchers.openmeteo_fetcher import OpenMeteoFetcher
from agents.data_ingestion.fetchers.terrain_fetcher import TerrainFetcher
from agents.data_ingestion.fetchers.hydro_fetcher import HydroFetcher
from agents.data_ingestion.fetchers.gov_dataset_fetcher import GovDatasetFetcher

__all__ = [
    "OpenMeteoFetcher",
    "TerrainFetcher",
    "HydroFetcher",
    "GovDatasetFetcher",
]
