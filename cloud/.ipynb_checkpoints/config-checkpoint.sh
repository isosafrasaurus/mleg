conda create -n gcc13env
conda activate gcc13env
conda install -c conda-forge gcc=13.2 ipykernel
/home/ubuntu/miniforge3/envs/gcc13env/bin/python -m ipykernel install --name gcc13env
# open the json file at ~/miniforge3/envs/gcc13env/share/jupyter/kernels/python3/kernel.json, add an attribute
# "env" : {
# "PATH": "/home/ubuntu/miniforge3/envs/gcc13env/bin/:${PATH}"
# }