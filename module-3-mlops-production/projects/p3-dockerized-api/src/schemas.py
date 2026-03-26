"""Pydantic schemas for the churn prediction API."""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """Input features for a single customer churn prediction."""

    tenure_months: int = Field(ge=0, le=120, description="Months as customer")
    monthly_charges: float = Field(ge=0, le=500, description="Monthly bill amount")
    total_charges: float = Field(ge=0, description="Total charges to date")
    contract: str = Field(
        description="Contract type",
        pattern="^(Month-to-month|One year|Two year)$",
    )
    internet_service: str = Field(
        description="Internet type",
        pattern="^(DSL|Fiber optic|No)$",
    )
    online_security: int = Field(ge=0, le=1, description="Has online security")
    tech_support: int = Field(ge=0, le=1, description="Has tech support")
    senior_citizen: int = Field(ge=0, le=1, default=0)
    partner: int = Field(ge=0, le=1, default=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure_months": 12,
                    "monthly_charges": 79.99,
                    "total_charges": 959.88,
                    "contract": "Month-to-month",
                    "internet_service": "Fiber optic",
                    "online_security": 0,
                    "tech_support": 0,
                    "senior_citizen": 0,
                    "partner": 1,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response for a single churn prediction."""

    churn_probability: float
    churn_prediction: bool
    model_version: str = "1.0.0"


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""

    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
