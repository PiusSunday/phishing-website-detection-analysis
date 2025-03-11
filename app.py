import hydra
import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as uvicorn_run

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.pipelines.training_pipeline import TrainingPipeline

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:

        # Load Hydra configuration directly
        with hydra.initialize(config_path="conf", version_base=None):
            cfg = hydra.compose(config_name="base")

        # Initialize TrainingPipeline with the loaded configuration
        training_pipeline = TrainingPipeline(cfg)

        training_pipeline.run_pipeline()

        return Response("Training completed successfully.")
    except Exception as e:
        return Response(f"Error during training: {e}")


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        prediction_pipeline = PredictionPipeline()
        result_df = prediction_pipeline.predict(df)

        prediction_pipeline.save_prediction_output(result_df)

        # Convert DataFrame to a list of dictionaries
        predictions = result_df.to_dict(orient="records")

        return templates.TemplateResponse(
            "table.html", {"request": request, "predictions": predictions}
        )

    except Exception as e:
        return Response(f"Error during prediction: {e}")


if __name__ == "__main__":

    uvicorn_run(app, host="0.0.0.0", port=8000)
