from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated, Optional
import os
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE_DIR, "model_lgbm", "model.pkl"))
scaler   = joblib.load(os.path.join(BASE_DIR, "model_lgbm", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "model_lgbm", "features.pkl"))

logger.info(f"✅ Modèle, scaler et features chargés ({len(features)} features)")

app = FastAPI(
    title="LightGBM Energy Consumption Predictor",
    description="API de prédiction de consommation énergétique",
    version="1.0.0"
)

# Types de propriété valides
VALID_PROPERTY_TYPES = [
    "Warehouse",
    "Distribution Center",
    "Self-Storage Facility",
    "Supermarket / Grocery Store",
    "Other",
    "Worship Facility",
    "Senior Care Community",
    "Mixed Use Property",
    "Medical Office",
    "Standard"  # aucune des catégories encodées → toutes à 0
]

VALID_BUILDING_TYPES = [
    "Nonresidential COS",
    "Standard"  # toutes les autres → colonne à 0
]

class HouseFeatures(BaseModel):
    anciennete: Annotated[float, Field(
        ge=0, le=200,
        description="Ancienneté du bâtiment en années"
    )]
    square_foot: Annotated[float, Field(
        gt=0,
        description="Surface en pieds carrés"
    )]
    has_gas: Annotated[int, Field(
        ge=0, le=1,
        description="Présence de gaz : 0 ou 1"
    )]
    property_gfa_parking: Annotated[float, Field(
        ge=0,
        description="Surface de parking en pieds carrés"
    )]
    primary_property_type: Annotated[str, Field(
        description=f"Type de propriété : {VALID_PROPERTY_TYPES}"
    )]
    building_type: Annotated[str, Field(
        description=f"Type de bâtiment : {VALID_BUILDING_TYPES}"
    )]
    has_steam: Annotated[int, Field(
        ge=0, le=1,
        description="Présence de vapeur : 0 ou 1"
    )]

    # Validation du type de propriété
    @field_validator("primary_property_type")
    @classmethod
    def validate_property_type(cls, v):
        if v not in VALID_PROPERTY_TYPES:
            raise ValueError(
                f"Type de propriété invalide : '{v}'. "
                f"Valeurs acceptées : {VALID_PROPERTY_TYPES}"
            )
        return v

    # Validation du type de bâtiment
    @field_validator("building_type")
    @classmethod
    def validate_building_type(cls, v):
        if v not in VALID_BUILDING_TYPES:
            raise ValueError(
                f"Type de bâtiment invalide : '{v}'. "
                f"Valeurs acceptées : {VALID_BUILDING_TYPES}"
            )
        return v

    # Validation croisée
    @model_validator(mode="after")
    def coherence_globale(self):
        if self.square_foot < self.property_gfa_parking:
            raise ValueError(
                f"La surface de parking ({self.property_gfa_parking} ft²) "
                f"ne peut pas dépasser la surface totale ({self.square_foot} ft²)"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "anciennete": 30,
                "square_foot": 50000,
                "has_gas": 1,
                "property_gfa_parking": 5000,
                "primary_property_type": "Warehouse",
                "building_type": "Nonresidential COS",
                "has_steam": 0
            }
        }
    }


def build_input_dataframe(feat: HouseFeatures) -> pd.DataFrame:
    """Reconstruit le DataFrame avec les 15 features attendues par le scaler"""

    # Initialise toutes les colonnes à 0 avec les noms exacts du scaler
    row = {col: 0 for col in features}  # 'features' = liste chargée depuis features.pkl

    # Colonnes numériques directes (noms exacts comme dans le scaler)
    row["Anciennetée"]        = feat.anciennete
    row["SquareFoot"]         = feat.square_foot
    row["hasGas"]             = feat.has_gas
    row["PropertyGFAParking"] = feat.property_gfa_parking
    row["hasSteam"]           = feat.has_steam

    # One-hot encoding PrimaryPropertyType
    col_type = f"PrimaryPropertyType_{feat.primary_property_type}"
    if col_type in row:
        row[col_type] = 1

    # One-hot encoding BuildingType
    col_building = f"BuildingType_{feat.building_type}"
    if col_building in row:
        row[col_building] = 1

    # Création du DataFrame avec colonnes en string
    df = pd.DataFrame([row])
    df.columns = df.columns.astype(str)  # ← correction du bug

    logger.info(f"Colonnes : {list(df.columns)}")
    logger.info(f"Valeurs  : {df.values}")

    return df


@app.get("/")
def root():
    return {"message": "API LightGBM opérationnelle ✅"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "LightGBM",
        "nb_features": len(features),
        "features": features
    }

@app.post("/predict")
def predict(feat: HouseFeatures):
    try:
        # Construction du DataFrame avec les 15 features
        input_df = build_input_dataframe(feat)
        logger.info(f"Input DataFrame :\n{input_df}")

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Prédiction
        log_pred = model.predict(input_scaled)
        EUIWN = float(np.exp(log_pred[0]))

        return {
            "SiteEUIWN(kBtu/sf)": round(EUIWN, 2),
            "input": feat.model_dump()
        }

    except Exception as e:
        logger.error(f"Erreur predict : {e}")
        raise HTTPException(status_code=500, detail=str(e))
