Bootstrap: docker
From: nvidia/cuda:9.0-cudnn7-runtime-centos7


%files
    requirements.txt

%environments
    export LANG=C.UTF-8

%post
    yum -y update
    yum -y install yum-utils groupinstall development libgomp
    yum -y install https://centos7.iuscommunity.org/ius-release.rpm
    yum -y install python36u python36u-pip 
    pip3.6 install --upgrade setuptools wheel
    pip3.6 install -r requirements.txt
