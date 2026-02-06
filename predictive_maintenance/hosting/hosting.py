
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ------------------------------------------------
# Config
# ------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

USERNAME = "raj2261992"
SPACE_NAME = "predictive_maintenance"

repo_id = f"{USERNAME}/{SPACE_NAME}"
repo_type = "space"

DEPLOYMENT_FOLDER = "predictive_maintenance/deployment"

# ------------------------------------------------
# Check if Space Exists
# ------------------------------------------------

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space already exists: {repo_id}")

except RepositoryNotFoundError:
    print("Space not found. Creating new Space...")

    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="docker",   # since you are using Docker
        exist_ok=True
    )

    print(f"Space created: {repo_id}")

# ------------------------------------------------
# Upload Deployment Files
# ------------------------------------------------

print("Uploading deployment folder...")

api.upload_folder(
    folder_path=DEPLOYMENT_FOLDER,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)

print("Deployment completed successfully!")
