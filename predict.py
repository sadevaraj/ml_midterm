from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, get_args

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from model import CATEGORICAL_FEATURES, FEATURE_DEFAULTS, FEATURES, normalize_record


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("burn_out_model.bin")))
with MODEL_PATH.open("rb") as model_file:
    model, dv = pickle.load(model_file)
MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
FEATURE_DEFAULTS_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_DEFAULTS, sort_keys=True).encode()
).hexdigest()
FEATURE_OPTIONS = {
    name: sorted(
        feature_name[len(name) + 1 :]
        for feature_name in dv.feature_names_
        if feature_name.startswith(f"{name}=")
    )
    for name in CATEGORICAL_FEATURES
}

app = FastAPI(title="ML Zoomcamp 2026 mid term")
PAGE_PATH = Path(__file__).with_name("static") / "index.html"


class Lead(BaseModel):
    developer_id: str | None = None
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    state: str | None = None
    marital_status: str | None = None
    education_level: str | None = None
    company_type: str | None = None
    company_size: str | None = None
    job_role: str | None = None
    employment_type: str | None = None
    experience_years: float | None = None
    salary_lpa: float | None = None
    years_in_current_company: float | None = None
    promotion_last_2_years: str | None = None
    remote_work_ratio: int | None = None
    weekly_work_hours: float | None = None
    weekend_work_frequency: str | None = None
    night_shift_frequency: str | None = None
    layoffs_seen_in_company: str | None = None
    number_of_switches: int | None = None
    primary_language: str | None = None
    tech_stack: str | None = None
    ai_tools_usage_hours_per_week: float | None = None
    upskilling_hours_per_week: float | None = None
    leetcode_problems_solved: int | None = None
    certifications_count: int | None = None
    side_projects_count: int | None = None
    github_activity_score: float | None = None
    interview_preparation_hours_weekly: float | None = None
    stress_level: float | None = None
    burnout_score: float | None = None
    anxiety_score: float | None = None
    sleep_hours: float | None = None
    caffeine_intake_per_day: int | None = None
    work_life_balance_rating: float | None = None
    imposter_syndrome_score: float | None = None
    social_media_usage_hours: float | None = None
    physical_activity_days_per_week: int | None = None
    therapy_or_counseling: str | None = None
    layoff_anxiety_score: float | None = None
    ai_replacement_fear_score: float | None = None
    job_security_confidence: float | None = None
    emergency_savings_months: int | None = None
    linkedin_job_search_activity: str | None = None
    applied_jobs_last_30_days: int | None = None
    burnout_risk_category: str | None = None
    likely_to_switch_job: str | None = None
    mentally_exhausted: str | None = None
    high_layoff_risk: str | None = None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PAGE_PATH)


@app.get("/schema")
def schema() -> dict[str, Any]:
    fields = []
    for name in FEATURES:
        annotation = Lead.model_fields[name].annotation
        field_types = get_args(annotation)
        if int in field_types:
            field_type = "integer"
        elif float in field_types:
            field_type = "number"
        else:
            field_type = "text"
        fields.append(
            {
                "name": name,
                "label": name.replace("_", " ").title(),
                "type": field_type,
                "default": FEATURE_DEFAULTS[name],
                "options": FEATURE_OPTIONS.get(name, []),
                "group": "Text and category fields"
                if name in CATEGORICAL_FEATURES
                else "Numeric details",
            }
        )
    return {"fields": fields}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_sha256": MODEL_SHA256,
        "feature_defaults_sha256": FEATURE_DEFAULTS_SHA256,
    }


@app.post("/predict")
def predict(lead: Lead) -> dict[str, Any]:
    record = normalize_record(lead.model_dump())
    probability = float(model.predict_proba(dv.transform([record]))[0, 1])
    return {
        "conversion_probability": round(probability, 6),
        "conversion": probability >= 0.23,  # best threshold based on ROC AUC analysis
    }