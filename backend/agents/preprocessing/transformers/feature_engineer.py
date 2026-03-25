"""
Feature Engineer – Creates derived features from the real flood dataset columns.

Dataset columns (india_pakistan_flood_balancednew.csv):
  Weather:      rain_mm_weekly, temp_c_mean, rh_percent_mean,
                wind_ms_mean, rain_mm_monthly
  Hydrological: dam_count_50km, dist_major_river_km, waterbody_nearby
  Terrain:      lat, lon, elevation_m, slope_degree, terrain_type_encoded
  Temporal:     week  (format: YYYY-MM)

Engineered features added (all conditionally – only if source col exists):
  Rainfall:
    - rain_intensity_ratio   : weekly / (monthly/4.3) — sub-monthly anomaly
    - rain_monthly_anomaly   : (weekly − monthly/4.3).clip  — how much this week deviates
    - heavy_rain_flag        : 1 if rain_mm_weekly > 50mm threshold (IMD heavy rain)
    - very_heavy_rain_flag   : 1 if rain_mm_weekly > 115mm (IMD very heavy)

  Temporal (from week column YYYY-MM):
    - month                  : 1–12 integer
    - is_monsoon             : 1 if month in {6,7,8,9}
    - month_sin, month_cos   : cyclical encoding

  Hydrological:
    - river_proximity_risk   : inverse dist_major_river_km, clipped → [0,1]
    - hydro_risk_score       : combined waterbody + dam density index

  Terrain:
    - terrain_wetness_idx    : ln(1 / tan(slope_degree)) – simplified TWI
    - low_elevation_flag     : 1 if elevation_m < 50m (flood plain / coastal)

  Interaction:
    - rain_x_slope           : rain_mm_weekly × slope_degree
    - rain_x_humidity        : rain_mm_weekly × rh_percent_mean / 100

  Flood risk proxy (for EDA only – NOT a training feature):
    - flood_risk_proxy       : rule-based 0/1/2 label

All raw input columns are preserved – only new columns are added.
Non-existent source columns are gracefully skipped.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import PreprocessingStrategy
from utils.logger import logger


# ── Thresholds (IMD classification) ──────────────────────────────────────────
HEAVY_RAIN_MM      = 50.0    # mm/week — heavy rain
VERY_HEAVY_RAIN_MM = 115.0   # mm/week — very heavy rain


class FeatureEngineer:
    """
    Adds derived features to the cleaned DataFrame using real dataset columns.

    Usage:
        fe = FeatureEngineer()
        df_with_features = fe.engineer(df, strategy, audit_logger)
    """

    def engineer(
        self,
        df:           pd.DataFrame,
        strategy:     PreprocessingStrategy,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """
        Adds all applicable engineered features.
        Gracefully skips any feature whose source column is missing.
        """
        if not strategy.engineer_features:
            logger.info("[FeatureEngineer] Skipped — engineer_features=False")
            return df

        df = df.copy()
        before_cols = set(df.columns)

        df = self._temporal_features(df, audit_logger)
        df = self._rainfall_features(df, audit_logger)
        df = self._hydrological_features(df, audit_logger)
        df = self._terrain_features(df, audit_logger)
        df = self._interaction_features(df, audit_logger)
        df = self._flood_risk_proxy(df, audit_logger)

        new_cols = set(df.columns) - before_cols
        audit_logger.log(
            step="feature_engineer",
            action=f"Added {len(new_cols)} engineered features: "
                   + ", ".join(sorted(new_cols)),
        )
        logger.info(f"[FeatureEngineer] Added {len(new_cols)} features. New shape: {df.shape}")
        return df

    # ── Temporal features (from 'week' column YYYY-MM format) ────────────────

    def _temporal_features(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """Extracts month and cyclical features from the 'week' column (YYYY-MM)."""
        if "week" not in df.columns:
            return df
        try:
            # week format: "1999-12" or "2001-06" – parse as year-month
            parsed = pd.to_datetime(df["week"].astype(str), format="%Y-%m", errors="coerce")

            df["month"]     = parsed.dt.month.astype(float)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            # Indian monsoon season flag (June–September)
            df["is_monsoon"] = parsed.dt.month.isin([6, 7, 8, 9]).astype(int)

            audit_logger.log(
                step="feature_engineer",
                action="Added temporal features from 'week': month, month_sin/cos, is_monsoon",
            )
        except Exception as exc:
            logger.warning(f"[FeatureEngineer] Temporal features failed: {exc}")
        return df

    # ── Rainfall features ─────────────────────────────────────────────────────

    def _rainfall_features(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """Derives rainfall intensity and anomaly features from real dataset cols."""
        w_col = "rain_mm_weekly"
        m_col = "rain_mm_monthly"
        added = []
        try:
            if w_col in df.columns:
                rain_w = df[w_col].fillna(0)

                # Binary flags
                df["heavy_rain_flag"]      = (rain_w >= HEAVY_RAIN_MM).astype(int)
                df["very_heavy_rain_flag"] = (rain_w >= VERY_HEAVY_RAIN_MM).astype(int)
                added += ["heavy_rain_flag", "very_heavy_rain_flag"]

                if m_col in df.columns:
                    rain_m = df[m_col].fillna(0)
                    # Expected weekly = monthly / 4.3
                    expected_weekly = (rain_m / 4.3).clip(lower=0)
                    # Intensity ratio: how much higher is this week vs monthly avg
                    df["rain_intensity_ratio"] = (
                        rain_w / (expected_weekly + 1e-6)
                    ).clip(0, 10)
                    # Weekly anomaly (positive = wetter than monthly average)
                    df["rain_monthly_anomaly"] = (rain_w - expected_weekly).clip(-200, 500)
                    added += ["rain_intensity_ratio", "rain_monthly_anomaly"]

            if added:
                audit_logger.log(
                    step="feature_engineer",
                    action=f"Added rainfall features: {', '.join(added)}",
                )
        except Exception as exc:
            logger.warning(f"[FeatureEngineer] Rainfall features failed: {exc}")
        return df

    # ── Hydrological features ─────────────────────────────────────────────────

    def _hydrological_features(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """Derives river proximity risk and hydro composite from real dataset cols."""
        dist_col     = "dist_major_river_km"
        dam_col      = "dam_count_50km"
        waterbody_col = "waterbody_nearby"
        added = []
        try:
            if dist_col in df.columns:
                # Proximity risk = 1 / (1 + distance) → ranges 0 to 1
                # Closer to river → higher risk
                df["river_proximity_risk"] = (
                    1.0 / (1.0 + df[dist_col].fillna(10.0).clip(lower=0))
                ).round(4)
                added.append("river_proximity_risk")

            if dam_col in df.columns and waterbody_col in df.columns:
                # Combined hydro risk: dams raise risk (unmanaged flow),
                # waterbody_nearby also raises it
                dam   = df[dam_col].fillna(0).clip(0, 20)
                wb    = df[waterbody_col].fillna(0)
                df["hydro_risk_score"] = (
                    (dam / 20.0) * 0.6 + wb * 0.4
                ).clip(0, 1).round(4)
                added.append("hydro_risk_score")

            if added:
                audit_logger.log(
                    step="feature_engineer",
                    action=f"Added hydrological features: {', '.join(added)}",
                )
        except Exception as exc:
            logger.warning(f"[FeatureEngineer] Hydrological features failed: {exc}")
        return df

    # ── Terrain features ──────────────────────────────────────────────────────

    def _terrain_features(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """Derives terrain wetness and low-elevation flags from real dataset cols."""
        slope_col   = "slope_degree"    # real column name
        elev_col    = "elevation_m"      # real column name
        terrain_raw = "terrain_type"      # raw string column
        added = []
        try:
            # 1. Slope-based features
            if slope_col in df.columns:
                slope_rad = np.radians(df[slope_col].fillna(1.0).clip(lower=0.01))
                df["terrain_wetness_idx"] = np.log(
                    1.0 / (np.tan(slope_rad) + 1e-6)
                ).clip(-5, 20)
                added.append("terrain_wetness_idx")

            # 2. Elevation-based features
            if elev_col in df.columns:
                df["low_elevation_flag"] = (df[elev_col].fillna(50) < 50).astype(int)
                added.append("low_elevation_flag")
            
            # 3. Handle terrain_type mapping (from string to encoded)
            if terrain_raw in df.columns and "terrain_type_encoded" not in df.columns:
                # Import map from PredictionAgent to stay in sync
                from agents.prediction.prediction_agent import TERRAIN_TYPE_ENCODING
                
                df["terrain_type_encoded"] = df[terrain_raw].astype(str).str.lower().map(
                    TERRAIN_TYPE_ENCODING
                ).fillna(1.0).astype(float)
                added.append("terrain_type_encoded")

            if added:
                audit_logger.log(
                    step="feature_engineer",
                    action=f"Added terrain features: {', '.join(added)}",
                )
        except Exception as exc:
            logger.warning(f"[FeatureEngineer] Terrain features failed: {exc}")
        return df

    # ── Interaction features ──────────────────────────────────────────────────

    def _interaction_features(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """Creates multiplicative interaction terms using real dataset columns."""
        rain_col  = "rain_mm_weekly"
        slope_col = "slope_degree"
        rh_col    = "rh_percent_mean"
        added = []
        try:
            if rain_col in df.columns and slope_col in df.columns:
                df["rain_x_slope"] = (
                    df[rain_col].fillna(0) * df[slope_col].fillna(0)
                )
                added.append("rain_x_slope")

            if rain_col in df.columns and rh_col in df.columns:
                df["rain_x_humidity"] = (
                    df[rain_col].fillna(0) * (df[rh_col].fillna(60) / 100.0)
                )
                added.append("rain_x_humidity")

            if added:
                audit_logger.log(
                    step="feature_engineer",
                    action=f"Added interaction features: {', '.join(added)}",
                )
        except Exception as exc:
            logger.warning(f"[FeatureEngineer] Interaction features failed: {exc}")
        return df

    # ── Flood risk proxy (EDA only — NOT a training feature) ─────────────────

    def _flood_risk_proxy(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """
        Rule-based flood risk proxy for EDA/debugging.
        0=LOW, 1=MEDIUM, 2=HIGH
        NOT used as a training label — flood_occurred is the target.
        """
        try:
            risk = pd.Series(0, index=df.index)

            if "rain_mm_weekly" in df.columns:
                rain = df["rain_mm_weekly"].fillna(0)
                risk = risk.where(rain < 15.6,  1)   # moderate+
                risk = risk.where(rain < VERY_HEAVY_RAIN_MM, 2)  # very heavy

            # Elevate risk if close to river
            if "dist_major_river_km" in df.columns:
                close = df["dist_major_river_km"].fillna(100) < 5
                risk  = risk.where(~close, risk.clip(lower=1))

            df["flood_risk_proxy"] = risk
            audit_logger.log(
                step="feature_engineer",
                action="Added flood_risk_proxy (0=LOW,1=MED,2=HIGH) — EDA only, NOT a training label",
            )
        except Exception as exc:
            logger.warning(f"[FeatureEngineer] flood_risk_proxy failed: {exc}")
        return df
