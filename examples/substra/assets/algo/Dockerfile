# this base image works in both CPU and GPU enabled environments
FROM substrafoundation/substra-tools:0.6.1

# install dependencies
RUN pip3 install numpy==1.18.2
RUN pip3 install nibabel
RUN apt-get update && apt-get install -y git
RUN pip3 install git+https://github.com/Project-MONAI/MONAI@0.3.0

# add your algorithm script to docker image
ADD algo.py .

# define how script is run
ENTRYPOINT ["python3", "algo.py"]
