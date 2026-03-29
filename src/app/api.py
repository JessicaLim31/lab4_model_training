# src/app/api.py
import joblib
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from sklearn.datasets import load_breast_cancer


"""build a Pydantic model from the breast cancer feature names"""

feature_names = [
    name.replace(" ", "_")
    for name in load_breast_cancer().feature_names
    ]
CancerRequest = create_model(
    "CancerRequest",
    **{name: (float, ...) for name in feature_names}
    )
    
def create_app(model_path: str = "models/cancer_model.pkl"):
    
    # Helpful guard so students get a clear error if they forgot to train first
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Run the pipeline first."
        )

    model = joblib.load(model_path)
    app = FastAPI(title="Breast Cancer Model API")

    # Map numeric predictions to class names
    target_names = {0: "malignant", 1: "benign"}
    

    @app.get("/")
    def root():
        return {
            "message": "BreasT Cancer model is ready for inference!",
            "classes": target_names,
            "expected_features": feature_names
        }

    @app.post("/predict")
    def predict(request: CancerRequest):
        # Convert request into the correct shape (1 x 4)
        X = np.array([
            getattr(request, name) for name in feature_names]).reshape(1,-1)
        
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        
        return {"prediction": target_names[idx], "class_index": idx}

    @app.get("/model/info")
    def model_info():
        """Return metadata for the currently promoted model."""
        metadata_path = Path("models/metadata.json")
        
        if not metadata_path.exists():
            raise  HTTPException(status_code=400, detail="No metadata found. Run the pipeline First")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    
    # return the FastAPI app
    return app
