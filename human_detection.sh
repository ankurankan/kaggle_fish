# Script for setting up yolo on a machine and train the network for detecting humans.

# Install cuda
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda -y

# Download data
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar

# Generate labels
wget https://gist.githubusercontent.com/ankurankan/8ee742ece4b83cc9dc30f62cc855a9e2/raw/8057fea421804d84e2333d8f165e1a136b039483/voc_labels.py
python voc_labels.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt

# Change config files
cd darknet/
rm cfg/voc.data
wget https://gist.githubusercontent.com/ankurankan/df557a00c40d2949942188e41b828a50/raw/4c8e7624bf9150474292482406d4f3c227e52490/voc.data

rm data/voc.names
wget https://gist.githubusercontent.com/ankurankan/16ea0524bc12f8f623a19a6b828a8e7e/raw/c76d88eab5bf84951119c727300b7b44da26db22/voc.names

# Get the weights
wget http://pjreddie.com/media/files/darknet19_448.conv.23

# Setup darknet
git clone https://github.com/pjreddie/darknet
cd darknet/
rm Makefile
wget https://gist.githubusercontent.com/ankurankan/effb677dba7a9bc9f0039b75c9c4aa60/raw/a2bfd0c0dfc24ea09f86676f00554f7a7e31e081/Makefile
cd

# Install opencv
sudo apt-get update
sudo apt-get install htop libopencv-dev python-opencv -y

# Setup cnn
# Transfer the file
tar -xvf cudnn-8.0-linux-x64-v5.1.tgz
cd cuda
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*

# Add cuda bin to path
echo 'export PATH = "/usr/local/cuda/bin/:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Make darknet
cd darknet/
make

# Train the model
./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23

