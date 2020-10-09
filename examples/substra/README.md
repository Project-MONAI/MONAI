# MONAI on the Substra platform

This examples demonstrates how to run the [3D segmentation tutorial](https://github.com/Project-MONAI/tutorials/tree/master/3d_segmentation) on the [Substra](https://github.com/substrafoundation/substra) platform.

In order to run the example, you'll need to have an instance of the substra platform running. Please refer to the [Substra documentation](https://doc.substra.ai/) for more information.

This is based on the version [0.7.1](https://github.com/SubstraFoundation/substra/releases/tag/0.7.1) of Substra.

## Running the example

### Preliminary

```sh
pip install -r requirements.txt
python scripts/generate_data_samples.py
```

### Testing in debug mode

```sh
substra config --profile node-1 http://substra-backend.node-1.com
DEBUG=True python scripts/run_in_substra.py
```

In debug mode, the scripts run locally and spawns Docker containers to execute the traintuples
and testtuples. Increase the resources allocated to Docker to avoid memory errors and make the execution faster.

### Running on a deployed Substra platform

If you do not have access to a Substra platform, you can [deploy one locally](https://doc.substra.ai/setup/local_install_skaffold.html).

Connect to Substra using the CLI, for example:
```sh
substra config --profile node-1 http://substra-backend.node-1.com
substra login --profile node-1 --username node-1 --password 'p@$swr0d44'
```

And launch the script:
```sh
python scripts/run_in_substra.py
```

### Performance

With 40 train data samples, 5 test data samples and 5 epochs, this example runs on GPU in 3 minutes and has a final score of 0.93.

Example output:
```sh
...
Execution took 2.576630981763204 minutes.
Performance on epoch 0: 0.591707444190979
Performance on epoch 1: 0.8561596512794495
Performance on epoch 2: 0.8999291062355042
Performance on epoch 3: 0.9219937562942505
Performance on epoch 4: 0.9314350485801697
```

## Monai to Substra adaptation notes

### Randomization and openers

`substratools.Opener` requests that user implement both get_X and get_y. It gets really tricky when using a randomized batch iterator like DataLoader because we cannot instantiate 2 DataLoader (one in each method) on the same data samples because each loader would be randomized differently and X_i would not match y_i.

Also we cannot use a randomized loader for testtuple, otherwise the order of items in y_pred (generated from one instance of the loader) would not match the one in y_true (generated from another instance of the loader).

A good solution would be to be able to set the order of data samples in the testtuple but this is currently not supported by Substra. If it were, the randomization could be done outside of the opener.

> Note: the randomization is especially important when trying to run multiple epochs (with randomization happing in-between epochs). In substra however each epoch should be a separate traintuple. So it would be perfectly possible to randomize differently the order of data samples at each traintuple.

In the meantime, here are a few solutions:

* Have a separate opener for training and one for testing. This is necessary if we want to keep the randomization.
* Only load the data once during training, either by:
  * having get_y return None and get_X return the loader (implemented here)
  * having get_y and get_X return iterators that iterate over separate parts of the loader (can be tricky)

### Debug mode memory issue

Attempting to run the example using the debug mode will raise a memory error:

```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)
```

By patching the code of the spawner in the substra repo as follows you should be able to run it without trouble.

```diff
diff --git a/substra/sdk/backends/local/compute/spawner.py b/substra/sdk/backends/local/compute/spawner.py
index f65b2ae..92af9d8 100644
--- a/substra/sdk/backends/local/compute/spawner.py
+++ b/substra/sdk/backends/local/compute/spawner.py
@@ -68,8 +68,6 @@ class DockerSpawner:
             detach=True,
             tty=True,
             stdin_open=True,
-            ipc_mode="host",
-            shm_size='8G',
         )

         execution_logs = []
```

See https://github.com/pytorch/pytorch#docker-image for reference.

### Misc

* Each prediction is saved as a different file, but substra expects only one file. The strategy was to save all files
  to a temporary directory and then zip the directory.

* Data sample files have their names stored in metadata. And since these metadata are pushed to the save_prediction,
  if all files have the same name (just in different folders) the predictions will also all have the same name,
  overwriting each other.

* NiftiSaver uses the channel last format, which means we have to revert to channel first when loading saved predictions

  Code from NiftiSaver:

  ```python
  # change data to "channel last" format and write to nifti format file
  data = np.moveaxis(data, 0, -1)
  ```

* Sorting data sample path with `sorted(paths)` works locally because each path looks like
  `~/MONAI/examples/segmentation_3d/assets/train_data_samples/train_data_sample_0/0_im.nii.gz` and therefore share a
  common path before the `*/*_im.nii.gz`.

  When running in substra, the same file is available at
  `/tmp/substra/medias/datasamples/<hash>/train_data_sample_0/0_im.nii.gz`. Applying `sorted` to such path will sort
  not by the number in `train_data_sample_X` but rather by the alphabetical order of hashes.

  However, since predictions are stored in a zip and then unzipped in a common folder, their relative order will be
  the same as for the local run. Which means the order of predictions and get_y results is the same in local but
  different in substra.

  The solution was therefore to use the file's basename to sort and not the full path.
