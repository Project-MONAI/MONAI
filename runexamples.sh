#!/bin/bash
set -e
# script for running the examples


# download data to specific directory
wget  https://www.dropbox.com/s/y890gb6axzzqff5/testing_ixi_t1.tar.gz?dl=1
mv testing_ixi_t1.tar.gz?dl=1 testing_ixi_t1.tar.gz
mkdir -p ./workspace/data/medical/ixi/IXI-T1/
tar -C ./workspace/data/medical/ixi/IXI-T1/ -xf testing_ixi_t1.tar.gz

# check data downloaded or not
[ -e "./testing_ixi_t1.tar.gz" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0


# home directory
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"


# run training files in examples/classification_3d
for file in "examples/classification_3d"/*train*
do
    python "$file"
done

# check training files generated from examples/classification_3d
[ -e "./best_metric_model_classification3d_array.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0
[ -e "./best_metric_model_classification3d_dict.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0

# run eval files in examples/classification_3d
for file in "examples/classification_3d"/*eval*
do
    python "$file"
done


# run training files in examples/classification_3d_ignite
for file in "examples/classification_3d_ignite"/*train*
do
    python "$file"
done

# check training files generated from examples/classification_3d_ignite
[ -e "./runs_array/net_checkpoint_20.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0
[ -e "./runs_dict/net_checkpoint_20.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0

# run eval files in examples/classification_3d_ignite
for file in "examples/classification_3d_ignite"/*eval*
do
    python "$file"
done


# run training files in examples/segmentation_3d
for file in "examples/segmentation_3d"/*train*
do
    python "$file"
done

# check training files generated from examples/segmentation_3d
[ -e "./best_metric_model_segmentation3d_array.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0
[ -e "./best_metric_model_segmentation3d_dict.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0

# run eval files in examples/segmentation_3d
for file in "examples/segmentation_3d"/*eval*
do
    python "$file"
done


# run training files in examples/segmentation_3d_ignite
for file in "examples/segmentation_3d_ignite"/*train*
do
    python "$file"
done

# check training files generated from examples/segmentation_3d_ignite
[ -e "./runs_array/net_checkpoint_100.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0
[ -e "./runs_dict/net_checkpoint_50.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0

# run eval files in examples/segmentation_3d_ignite
for file in "examples/segmentation_3d_ignite"/*eval*
do
    python "$file"
done


# run training file in examples/workflows
for file in "examples/workflows"/*train*
do
    python "$file"
done

# check training file generated from examples/workflows
[ -e "./runs/net_key_metric*.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0

# run eval file in examples/workflows
for file in "examples/workflows"/*eval*
do
    python "$file"
done


# run training file in examples/synthesis
for file in "examples/synthesis"/*train*
do
    python "$file"
done

# check training file generated from examples/synthesis
[ -e "./model_out/*.pth" ] && echo "1" >> "temp.txt" || echo "0" >> "temp.txt" && exit 0

# run eval file in examples/synthesis
for file in "examples/synthesis"/*eval*
do
    python "$file"
done
