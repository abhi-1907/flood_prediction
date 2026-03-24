import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from agents.data_ingestion.data_merger import CORE_COLUMNS
from agents.data_ingestion.schema_validator import TIMESTAMP_KEYWORDS, HYDRO_KEYWORDS, TERRAIN_KEYWORDS, RAINFALL_KEYWORDS
from agents.data_ingestion.fetchers.openmeteo_fetcher import OpenMeteoFetcher
from agents.data_ingestion.fetchers.terrain_fetcher import TerrainFetcher
from agents.data_ingestion.fetchers.hydro_fetcher import HydroFetcher
from agents.data_ingestion.fetchers.gov_dataset_fetcher import GovDatasetFetcher
from agents.data_ingestion.source_identifier import REQUIRED_FIELDS
from agents.data_ingestion.ingestion_schemas import DataCategory

checks = {
    "rain_mm_weekly in CORE_COLUMNS":      "rain_mm_weekly" in CORE_COLUMNS,
    "dist_major_river_km in CORE_COLUMNS": "dist_major_river_km" in CORE_COLUMNS,
    "slope_degree in CORE_COLUMNS":        "slope_degree" in CORE_COLUMNS,
    "lat in CORE_COLUMNS":                 "lat" in CORE_COLUMNS,
    "rainfall_mm GONE from CORE_COLUMNS":  "rainfall_mm" not in CORE_COLUMNS,
    "latitude GONE from CORE_COLUMNS":     "latitude" not in CORE_COLUMNS,
    "week in TIMESTAMP_KEYWORDS":          "week" in TIMESTAMP_KEYWORDS,
    "waterbody in HYDRO_KEYWORDS":         "waterbody" in HYDRO_KEYWORDS,
    "slope_degree in TERRAIN_KEYWORDS":    "slope_degree" in TERRAIN_KEYWORDS,
    "weekly in RAINFALL_KEYWORDS":         "weekly" in RAINFALL_KEYWORDS,
    "REQUIRED_FIELDS RAINFALL ok":         "rain_mm_weekly" in REQUIRED_FIELDS[DataCategory.RAINFALL],
    "REQUIRED_FIELDS HYDRO ok":            "dist_major_river_km" in REQUIRED_FIELDS[DataCategory.HYDRO],
    "REQUIRED_FIELDS TERRAIN ok":          "lat" in REQUIRED_FIELDS[DataCategory.TERRAIN],
}

all_ok = True
for k, v in checks.items():
    status = "OK" if v else "FAIL"
    if not v:
        all_ok = False
    print(f"[{status}] {k}")

print()
print("CORE_COLUMNS:", CORE_COLUMNS)
print()
print("DONE -", "ALL PASSED" if all_ok else "SOME FAILED")
