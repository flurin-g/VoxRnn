Bootstrap: docker
From: nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04


%files
    requirements.txt

%environments
    export LANG=C.UTF-8

%post
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-2 \
        cuda-cublas-9-2 \
        cuda-cufft-9-2 \
        cuda-curand-9-2 \
        cuda-cusolver-9-2 \
        cuda-cusparse-9-2 \
        python3 \
        python3-pip
    pip3 install --upgrade setuptools wheel
    pip3 install -r requirements.txt
