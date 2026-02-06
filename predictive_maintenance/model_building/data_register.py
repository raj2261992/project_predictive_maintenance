
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Get token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN')

# Ensure token is available
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it.")

api = HfApi(token=HF_TOKEN)

repo_id = "raj2261992/predictive_maintenance"
repo_type = "dataset"

# Check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Verify folder exists
folder_path = "predictive_maintenance/data"
print("Folder exists:", os.path.exists(folder_path))
print("Files in folder:", os.listdir(folder_path))

# Upload folder
api.upload_folder(
    folder_path=folder_path,
    path_in_repo="",  # root
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload CSV files"
)

print("Upload finished. Check your HF dataset page!")
