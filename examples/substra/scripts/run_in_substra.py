import os
import time
import zipfile
from distutils.util import strtobool
from pathlib import Path

import substra

N_EPOCHS = 5
N_TRAIN_SAMPLES = 40  # Max 40
DEBUG = bool(strtobool(os.environ.get("DEBUG", "False")))

# Path to the assets folder
ASSETS_DIR = Path(__file__).parents[1].resolve() / "assets"

# List of train data sample folders
TRAIN_DATA_SAMPLE_FOLDERS = [f for f in (ASSETS_DIR / "train_data_samples").iterdir() if f.is_dir()][
    :N_TRAIN_SAMPLES
]  # Keep only n train samples

# List of test data sample folders
TEST_DATA_SAMPLE_FOLDERS = [f for f in (ASSETS_DIR / "test_data_samples").iterdir() if f.is_dir()]


def build_archive(name, files):
    """Create a zip archive from files"""
    archive = ASSETS_DIR / name
    with zipfile.ZipFile(archive, "w") as z:
        for filepath in files:
            z.write(filepath, arcname=filepath.name)
    return archive


# Connect to Substra
client = substra.Client.from_config_file(profile_name="node-1", debug=DEBUG)

# Create a dataset
print("Adding train dataset")
train_dataset_key = client.add_dataset(
    {
        "name": "Segmentation 3D - Train",
        "description": ASSETS_DIR / "train_dataset" / "description.md",
        "type": "3D images",
        "data_opener": ASSETS_DIR / "train_dataset" / "opener.py",
        "objective_key": None,
        "permissions": {
            "public": True,
            "authorized_ids": list(),
        },
    },
    exist_ok=True,
)

# Create the train data samples
train_data_sample_keys = []
for i, train_data_sample_folder in enumerate(TRAIN_DATA_SAMPLE_FOLDERS):
    print(f"Adding train data sample {i+1}/{len(TRAIN_DATA_SAMPLE_FOLDERS)}")
    data_sample_key = client.add_data_sample(
        {
            "path": train_data_sample_folder,
            "data_manager_keys": [train_dataset_key],
            "test_only": False,
        },
        exist_ok=True,
    )
    train_data_sample_keys.append(data_sample_key)

train_dataset = client.get_dataset(train_dataset_key)

# Create the test dataset
print("Adding test dataset")
test_dataset_key = client.add_dataset(
    {
        "name": "Segmentation 3D - Test",
        "description": ASSETS_DIR / "test_dataset" / "description.md",
        "type": "3D images",
        "data_opener": ASSETS_DIR / "test_dataset" / "opener.py",
        "objective_key": None,
        "permissions": {
            "public": True,
            "authorized_ids": list(),
        },
    },
    exist_ok=True,
)

# Create the test data samples
test_data_sample_keys = []
for i, test_data_sample_folder in enumerate(TEST_DATA_SAMPLE_FOLDERS):
    print(f"Adding test data sample {i+1}/{len(TEST_DATA_SAMPLE_FOLDERS)}")
    data_sample_key = client.add_data_sample(
        {
            "path": test_data_sample_folder,
            "data_manager_keys": [test_dataset_key],
            "test_only": True,
        },
        exist_ok=True,
    )
    test_data_sample_keys.append(data_sample_key)

test_dataset = client.get_dataset(test_dataset_key)

# Create the objective (metrics)
print("Adding objective")
metrics_archive = build_archive(
    "metrics.zip",
    [
        ASSETS_DIR / "objective" / "metrics.py",
        ASSETS_DIR / "objective" / "Dockerfile",
    ],
)
objective_key = client.add_objective(
    {
        "name": "Segmentation 3D",
        "description": ASSETS_DIR / "objective" / "description.md",
        "metrics_name": "mean dice",
        "metrics": metrics_archive,
        "test_data_manager_key": test_dataset_key,
        "test_data_sample_keys": test_data_sample_keys,
        "permissions": {
            "public": True,
            "authorized_ids": list(),
        },
    },
    exist_ok=True,
)

# Create the algo
print("Adding algo")
algo_archive = build_archive(
    "algo.zip",
    [
        ASSETS_DIR / "algo" / "algo.py",
        ASSETS_DIR / "algo" / "Dockerfile",
    ],
)
algo_key = client.add_algo(
    {
        "name": "UNet",
        "description": ASSETS_DIR / "algo" / "description.md",
        "file": algo_archive,
        "permissions": {
            "public": True,
            "authorized_ids": list(),
        },
    },
    exist_ok=True,
)

# Create the compute plan: one traintuple and testtuple per epoch
print("Adding compute plan")
traintuples = []
testtuples = []
traintuple = None
for epoch_number in range(N_EPOCHS):
    traintuple = {
        "traintuple_id": epoch_number,
        "algo_key": algo_key,
        "data_manager_key": train_dataset_key,
        "train_data_sample_keys": train_dataset.train_data_sample_keys,
        "in_models_ids": [traintuple["traintuple_id"]] if traintuple else [],
    }
    testtuple = {
        "objective_key": objective_key,
        "traintuple_id": traintuple["traintuple_id"],
    }
    traintuples.append(traintuple)
    testtuples.append(testtuple)

print("Waiting for compute plan to finish")
start = time.time()

compute_plan = client.add_compute_plan(
    {
        "traintuples": traintuples,
        "testtuples": testtuples,
    }
)

# In normal (not debug) mode, the execution is asynchronous so we poll the platform
# every 10s to check the progress of the execution.
while True:
    compute_plan = client.get_compute_plan(compute_plan.compute_plan_id)
    testtuples = [client.get_testtuple(testtuple_key) for testtuple_key in compute_plan.testtuple_keys]
    testtuples = sorted(testtuples, key=lambda x: x.rank)
    for testtuple in testtuples:
        if testtuple.status == "done":
            print(f"Performance on epoch {testtuple.rank}: {testtuple.dataset.perf:.2f}")
    print(
        f"  - {compute_plan.done_count}/{compute_plan.tuple_count} traintuple and testtuples done in {time.time() - start:.2f}s"
    )
    if compute_plan.status in ["done", "failed"]:
        break
    time.sleep(10)

end = time.time()
print(f"Execution took {(end - start)/60:.2f} minutes with the status {compute_plan.status}")
