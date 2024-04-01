FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get clean \
&& apt-get update -y \
&& apt install -y \
sudo \
net-tools \
fcitx-hangul \
fonts-nanum* \
vim \
wget \
git

# cudnn update
# ref - https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb \
&& sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb \
&& sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/ \
&& sudo apt-get update -y 

# download python3.11
RUN apt-get install -y python3.11 python3-pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# pip3 upgrade
RUN python3 -m pip install --upgrade pip
# cudnn download
RUN sudo apt-get -y install cudnn-cuda-12

# torch 2.20 download
RUN pip3 install numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# download for amd
RUN pip3 install pandas yacs PyYAML termcolor Cython tensorboard gdown tabulate mat4py scikit-learn
