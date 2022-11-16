2022-11-15

**更新了针对最新的yolov5代码的ji_new.py**

2022-11-16

**更新了用于测试多边形的ji_polygon.py 感谢qian0733提供的代码**

## 1.报名打榜

![image-20221109092143080](./img/1.jpg)

## 2.创建镜像

**2.1 创建镜像**

![image-20221109092352060](./img/2.jpg)

**2.2 选择torch编码环境**

![image-20221109092720036](./img/3.jpg)

**进入编码环境**

![image-20221109092720036](./img/4.jpg)

## 3.编码环境

**新建Terminal**

![image-20221109093742004](./img/5.jpg)

### 3.1 数据介绍

```shell
#新建Terminal
cd /home/data
ls
```

![image-20221109093742004](./img/6.jpg)

**将数据拷贝到/project/train/src_repo**

```
cp -r /home/data/1225 /project/train/src_repo/trainval
cp -r /home/data/599 /project/train/src_repo/trainval
cp -r /home/data/820 /project/train/src_repo/trainval
```

### 3.2 数据处理

**将代码中的split.py 直接拷贝，修改路径**，**主要修改下面两张图的部分**

![image-20221109093742004](./img/7.jpg)

![image-20221109093742004](./img/8.jpg)

**运行split.py**

```
python split.py
```

### 3.3 yolov5代码

```shell
#下载yolov5代码
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-14177-files/6c888875-7de7-4f60-ba15-88de0304fe70/yolov5.zip
#解压
unzip yolov5.zip
cd yolov5
#安装python库
pip install -r requirements.txt 
#下载yolov5s模型
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-14177-files/d8ee0651-9d48-4480-a161-d22750ba7eba/yolov5s.pt
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-14177-files/d97a419e-2bb4-4822-b398-b37244473a5f/Arial.ttf
```

**在yolov5/data/目录下新建fire.yaml文件**

```yaml
#数据路径，不用更改
train: ../train.txt
val: ../val.txt

#类别数量，根据题目设定
nc: 3

#类别名称，根据题目设定
name: ['fire','big_fire','smoke']
```

```
python train.py --batch-size 32 --epochs 3 --data ./data/fire.yaml --hyp ./data/hyps/hyp.scratch-low.yaml --weight ./yolov5s.pt --img 640 --project /project/train/models/ --cfg ./models/yolov5s.yaml
```





## 4.训练

 **就是在编码做了哪些事情，在训练环境环境也需要去做**

**新建 run.sh 脚本**

```shell
cp -r /home/data/1225 /project/train/src_repo/trainval
cp -r /home/data/599 /project/train/src_repo/trainval
cp -r /home/data/820 /project/train/src_repo/trainval
python /project/train/src_repo/split.py
cd /project/train/src_repo/yolov5
python train.py --batch-size 64 --epochs 100 --data ./data/fire.yaml --hyp ./data/hyps/hyp.scratch-low.yaml --weight ./yolov5s.pt --img 640 --project /project/train/models/ --cfg ./models/yolov5s.yaml
```

**发起训练之前将编码环境里面生成的数据清理一下**

**删除 trainval 文件夹**

**删除 val.txt ,train.txt val.cache train.cache**

**删除 /project/train/models/ 文件夹下面的所有文件**



**发起训练**

在执行命令里面输入

```
bash /project/train/src_repo/run.sh
```



![image-20221109093742004](./img/9.jpg)



## 5.测试

**在 ev_sdk/src/ 目录下新建ji.py** 

**只需要更改这里的参数**

![image-20221109093742004](./img/10.jpg)

