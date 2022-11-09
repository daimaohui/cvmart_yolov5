cp -r /home/data/1225 /project/train/src_repo/trainval
cp -r /home/data/599 /project/train/src_repo/trainval
cp -r /home/data/820 /project/train/src_repo/trainval
python /project/train/src_repo/split.py
cd /project/train/src_repo/yolov5
python train.py --batch-size 64 --epochs 100 --data ./data/fire.yaml --hyp ./data/hyps/hyp.scratch-low.yaml --weight ./yolov5s.pt --img 640 --project /project/train/models/ --cfg ./models/yolov5s.yaml