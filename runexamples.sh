#!/bin/bash
set -e
# script for running the examples


# Download data to specific directory
wget  https://www.dropbox.com/s/y890gb6axzzqff5/testing_ixi_t1.tar.gz?dl=1
mv testing_ixi_t1.tar.gz?dl=1 testing_ixi_t1.tar.gz
mkdir -p ./workspace/data/medical/ixi/IXI-T1/
tar -C ./workspace/data/medical/ixi/IXI-T1/ -xf testing_ixi_t1.tar.gz

# home directory
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"

# run files in examples/classification_3d
for file in "examples/classification_3d"/*train*
do
    python "$file"
done

for file in "examples/classification_3d"/*eval*
do
    python "$file"
done

# run files in examples/classification_3d_ignite
for file in "examples/classification_3d_ignite"/*train*
do
    python "$file"
done

for file in "examples/classification_3d_ignite"/*eval*
do
    python "$file"
done

# run files in examples/segmentation_3d
for file in "examples/segmentation_3d"/*train*
do
    python "$file"
done

for file in "examples/segmentation_3d"/*eval*
do
    python "$file"
done

# run files in examples/segmentation_3d_ignite
for file in "examples/segmentation_3d_ignite"/*train*
do
    python "$file"
done

for file in "examples/segmentation_3d_ignite"/*eval*
do
    python "$file"
done

# run files in examples/workflows
for file in "examples/workflows"/*train*
do
    python "$file"
done

for file in "examples/workflows"/*eval*
do
    python "$file"
done

# run files in examples/synthesis
for file in "examples/synthesis"/*train*
do
    python "$file"
done

for file in "examples/synthesis"/*eval*
do
    python "$file"
done
