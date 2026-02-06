# =========================================================
# Imports
# =========================================================
import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


# =========================================================
# MLflow Remote Tracking Setup (SERVER BASED)
# =========================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI environment variable not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("predictive-maintenance-engine-condition")

print(f"MLflow logging to remote server: {MLFLOW_TRACKING_URI}")


# =========================================================
# Hugging Face API
# =========================================================
api = HfApi(token=os.getenv("HF_TOKEN"))


# =========================================================
# Load Dataset
# =========================================================
DATASET_REPO = "raj2261992/predictive_maintenance"

Xtrain = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO, filename="X_train.csv", repo_type="dataset"))
Xtest  = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO, filename="X_test.csv", repo_type="dataset"))
ytrain = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO, filename="y_train.csv", repo_type="dataset"))
ytest  = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO, filename="y_test.csv", repo_type="dataset"))


# =========================================================
# Feature Definitions
# =========================================================
required_columns = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp"
]

target_column = "Engine Condition"

Xtrain = Xtrain[required_columns]
Xtest = Xtest[required_columns]


# =========================================================
# Class Weight
# =========================================================
class_weight = ytrain[target_column].value_counts()[0] / ytrain[target_column].value_counts()[1]


# =========================================================
# Pipeline
# =========================================================
preprocessor = make_column_transformer((StandardScaler(), required_columns))

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8],
    "xgbclassifier__colsample_bytree": [0.7],
    "xgbclassifier__reg_lambda": [1.0]
}

model_pipeline = make_pipeline(preprocessor, xgb_model)


# =========================================================
# Training + MLflow
# =========================================================
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain[target_column])

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    threshold = 0.45

    y_test_pred = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    report = classification_report(ytest[target_column], y_test_pred, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy": report["accuracy"],
        "test_precision": report["1"]["precision"],
        "test_recall": report["1"]["recall"],
        "test_f1": report["1"]["f1-score"]
    })

    model_path = "engine_condition_xgboost_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print("Model logged to remote MLflow")


# =========================================================
# Upload Model to HuggingFace
# =========================================================
MODEL_REPO = "raj2261992/predictive_maintenance_model"

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Model uploaded to Hugging Face")
