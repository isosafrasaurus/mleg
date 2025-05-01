cd /home/ubuntu

conda create -n gcc13env
conda activate gcc13env
conda install -c conda-forge gcc=13.2 ipykernel
/home/ubuntu/miniforge3/envs/gcc13env/bin/python -m ipykernel install --name gcc13env
# open the json file at ~/miniforge3/envs/gcc13env/share/jupyter/kernels/python3/kernel.json, add an attribute
# "env" : {
# "PATH": "/home/ubuntu/miniforge3/envs/gcc13env/bin/:${PATH}"
# }

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

export PATH=/usr/local/cuda/bin/:$PATH
export PATH=/home/ubuntu/miniforge3/envs/cuda118env/lib/:$PATH
wget https://github.com/bitsandbytes-foundation/bitsandbytes/archive/refs/tags/0.45.5.tar.gz
tar -xf 0.45.5.tar.gz
cd bitsandbytes-0.45.5/
cmake -DCOMPUTE_BACKEND=/usr/local/cuda/bin/cuda/nvcc -S .
make
pip install .
cd ..

pip install transformers accelerate peft -q
pip install git+https://github.com/huggingface/diffusers.git -q
pip install sentencepiece protobuf datasets
pip install google-generativeai pillow