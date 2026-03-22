"""Feature engineering for train delay prediction."""
import pandas as pd
import numpy as np
from typing import Tuple


def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create advanced features for delay prediction."""

    # Combine for feature consistency
    train['is_train'] = 1
    test['is_train'] = 0

    combined = pd.concat([train, test], ignore_index=True)

    # Time-based features
    if 'month' in combined.columns:
        # Monsoon season (July-August)
        combined['is_monsoon'] = combined['month'].isin([7, 8]).astype(int)
        # Fog season (Dec-Feb)
        combined['is_fog_season'] = combined['month'].isin([12, 1, 2]).astype(int)
        # Peak summer
        combined['is_peak_summer'] = combined['month'].isin([4, 5]).astype(int)

    if 'departure_hour' in combined.columns:
        # Time of day buckets
        combined['is_morning_peak'] = ((combined['departure_hour'] >= 7) &
                                       (combined['departure_hour'] <= 10)).astype(int)
        combined['is_evening_peak'] = ((combined['departure_hour'] >= 17) &
                                       (combined['departure_hour'] <= 20)).astype(int)
        combined['is_night'] = ((combined['departure_hour'] >= 23) |
                                (combined['departure_hour'] <= 4)).astype(int)

    # Zone congestion features
    if 'zone' in combined.columns:
        # High congestion zones
        high_congestion = ['ER', 'NR', 'NCR', 'NER']
        combined['is_high_congestion_zone'] = combined['zone'].isin(high_congestion).astype(int)

        # Worst performing zones
        worst_zones = ['ER']
        combined['is_worst_zone'] = combined['zone'].isin(worst_zones).astype(int)

    # Route complexity
    if 'num_scheduled_stops' in combined.columns and 'distance_km' in combined.columns:
        # Stops per 100km
        combined['stops_density'] = combined['num_scheduled_stops'] / (combined['distance_km'] / 100 + 1)

    if 'scheduled_travel_hours' in combined.columns and 'distance_km' in combined.columns:
        # Average speed
        combined['avg_speed_kmh'] = combined['distance_km'] / (combined['scheduled_travel_hours'] + 0.1)

    # Weather interaction
    if 'fog_risk_score' in combined.columns and 'is_fog_season' in combined.columns:
        combined['fog_season_risk'] = combined['fog_risk_score'] * combined['is_fog_season']

    if 'season_severity_score' in combined.columns:
        combined['high_severity'] = (combined['season_severity_score'] >
                                   combined['season_severity_score'].quantile(0.75)).astype(int)

    # Rolling stock age
    if 'loco_age_years' in combined.columns and 'coach_age_years' in combined.columns:
        combined['avg_stock_age'] = (combined['loco_age_years'] + combined['coach_age_years']) / 2
        combined['old_stock'] = ((combined['loco_age_years'] > 20) |
                                (combined['coach_age_years'] > 25)).astype(int)

    # LHB advantage
    if 'has_lhb_coaches' in combined.columns:
        combined['modern_stock'] = combined['has_lhb_coaches']

    # Maintenance interaction
    if 'maintenance_score' in combined.columns and 'old_stock' in combined.columns:
        combined['maintenance_urgency'] = combined['old_stock'] * (100 - combined['maintenance_score'])

    # Operational risk
    if 'late_incoming_rake' in combined.columns:
        combined['operational_cascade_risk'] = combined['late_incoming_rake']

    if 'is_overloaded' in combined.columns:
        combined['capacity_stress'] = combined['is_overloaded']

    # Historical performance
    if 'route_historical_ontime_pct' in combined.columns:
        combined['poor_route_history'] = (combined['route_historical_ontime_pct'] < 50).astype(int)

    # Track conditions
    if 'track_doubled' in combined.columns and 'is_electrified' in combined.columns:
        combined['poor_infrastructure'] = ((combined['track_doubled'] == 0) |
                                          (combined['is_electrified'] == 0)).astype(int)

    # PSR risk
    if 'psr_count' in combined.columns:
        combined['high_psr'] = (combined['psr_count'] > 5).astype(int)

    # Seat utilization stress
    if 'seat_utilisation_pct' in combined.columns:
        combined['high_utilization'] = (combined['seat_utilisation_pct'] > 90).astype(int)

    # Composite risk score
    risk_features = ['is_high_congestion_zone', 'is_monsoon', 'late_incoming_rake',
                    'old_stock', 'poor_infrastructure', 'high_psr', 'high_utilization']
    available_risk = [f for f in risk_features if f in combined.columns]
    if available_risk:
        combined['composite_risk_score'] = combined[available_risk].sum(axis=1)

    # Split back
    train_fe = combined[combined['is_train'] == 1].drop('is_train', axis=1)
    test_fe = combined[combined['is_train'] == 0].drop(['is_train', 'is_delayed'], axis=1, errors='ignore')

    return train_fe, test_fe
