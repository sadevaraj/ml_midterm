from __future__ import annotations

import math
from typing import Any


CATEGORICAL_FEATURES = (
    "gender",
    "city",
    "state",
    "marital_status",
    "education_level",
    "company_type",
    "company_size",
    "job_role",
    "employment_type",
    "promotion_last_2_years",
    "weekend_work_frequency",
    "night_shift_frequency",
    "layoffs_seen_in_company",
    "primary_language",
    "tech_stack",
    "therapy_or_counseling",
    "linkedin_job_search_activity",
    "burnout_risk_category",
    "likely_to_switch_job",
    "high_layoff_risk",
)
NUMERICAL_FEATURES = (
    "age",
    "experience_years",
    "salary_lpa",
    "years_in_current_company",
    "remote_work_ratio",
    "weekly_work_hours",
    "number_of_switches",
    "ai_tools_usage_hours_per_week",
    "upskilling_hours_per_week",
    "leetcode_problems_solved",
    "certifications_count",
    "side_projects_count",
    "github_activity_score",
    "interview_preparation_hours_weekly",
    "stress_level",
    "burnout_score",
    "anxiety_score",
    "sleep_hours",
    "caffeine_intake_per_day",
    "work_life_balance_rating",
    "imposter_syndrome_score",
    "social_media_usage_hours",
    "physical_activity_days_per_week",
    "layoff_anxiety_score",
    "ai_replacement_fear_score",
    "job_security_confidence",
    "emergency_savings_months",
    "applied_jobs_last_30_days",
)
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
FEATURE_DEFAULTS = {
    **{feature: "NA" for feature in CATEGORICAL_FEATURES},
    **{feature: 0 for feature in NUMERICAL_FEATURES},
}


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply the notebook's feature selection and missing-value policy."""

    record: dict[str, Any] = {}
    for feature in FEATURES:
        value = raw.get(feature)
        record[feature] = (
            FEATURE_DEFAULTS[feature]
            if value is None
            or value == ""
            or (isinstance(value, float) and math.isnan(value))
            else value
        )

    return record
