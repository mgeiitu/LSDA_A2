import pandas as pd
import mlflow
import json
from influxdb import InfluxDBClient

from sklearn.pipeline import Pipeline
from transformers import (Column_selection, Scaling, Drop_na, Radians, VectorizeDir)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import shutil
from datetime import datetime


def main():
    
    def make_pipeline(regressor):
        pipe = Pipeline(
        [('Column_selection', Column_selection()),
        ('Drop_na', Drop_na()),
        ('Scaling', Scaling()),
        ('Radians', Radians()),
        ('VectorizeDir', VectorizeDir()),
        ('PolynomialFeatures', PolynomialFeatures(degree=2)),
        ('Regressor', regressor)]
        )

        return pipe

    def save_best_model(pipeline):
        mlflow.sklearn.save_model(
            pipeline,
            path="best_model",
            conda_env="conda.yaml")

    def save_model(pipeline):
        mlflow.sklearn.save_model(
            pipeline,
            path="best_model",
            conda_env="conda.yaml")


    def read_best_model():
        with open("best_model.json", "r") as f:
            best_model = json.loads(f.read())
            return best_model

    def create_new_model_dict(model_name, final_score):
        model_dict = {
            "timestamp": datetime.now().timestamp(),
            "date": datetime.today().strftime("%m/%d/%Y"),
            "model_name": model_name,
            "score": final_score,
        }

        return model_dict

    def save_model_metadata_to_disk(best_model):
        with open("best_model.json", "w") as f:
            json.dump(best_model, f)

            

    df = pd.read_json("dataset.json", orient="split")
    X = df
    y = df['Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    regression_models = {'LinearRegression': LinearRegression(),
                        'KNeighborsRegressor': KNeighborsRegressor(),
                        'GradientBoostingRegressor': GradientBoostingRegressor(),
                        'RandomForestRegressor': RandomForestRegressor()}

    for name, model in regression_models.items():

        with mlflow.start_run(run_name=f"{name}"):

            pipeline = make_pipeline(model)

            pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(pipeline, artifact_path=f"{name}")

            pipe_score = pipeline.score(X_test, y_test)

            mlflow.log_metric('R2 score', pipe_score)

            try:
                best_model = read_best_model()
                if pipe_score > best_model["score"]:
                    shutil.rmtree('best_model')  # deleting current best_model directory
                    print(f'Saving new best model {model}')
                    save_model(pipeline)
                    best_model = create_new_model_dict(name, pipe_score)
                    save_model_metadata_to_disk(best_model)

                else:
                    print(f"Model {model} not better. Not saving")

            except FileNotFoundError:
                print(f"No model saved to disk yet. Saving {model}")
                save_model(pipeline)
                best_model = create_new_model_dict(name, pipe_score)
                save_model_metadata_to_disk(best_model)

if __name__ == '__main__':
    main()
