cd /usr/local
# install TensorRT
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-14177-files/174710ef-e9a8-4406-a332-746713d04e6c/TensorRT-7.2.3.4.tar.gz
tar -xf TensorRT-7.2.3.4.tar.gz
echo "export LD_LIBRARY_PATH=/usr/local/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH" >> ~/.zshrc && source ~/.zshrc
#install cudann8.1
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-14177-files/93b8a46d-8d0a-445c-bd37-89dbfc6d8ea4/libcudnn8.1.1.33.deb && dpkg -i libcudnn8.1.1.33.deb