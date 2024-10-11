1. Clone repo and install requirement on edge <br>

**Clone repo**
```shell
git clone https://github.com/datdt198213/CAF-YOLO.git
```
**Install requirement**
```shell
cd /CAF-YOLO/
pip install -r requirements.txt
pip install albumentations==1.4
```

2. Send dataset from local to edge <br>
```shell
scp -i TinhNT_test.pem ubuntu@54.177.117.25:/home/ubuntu/dat_folder/CAF-YOLO/runs/detect/train13/weights/best.pt .
scp -i TinhNT_test.pem datasets.zip ubuntu@54.177.117.25:/home/ubuntu/dat_folder/CAF-YOLO/
```

3. Training on edge <br>
**Run file train.py to train model**

```shell
cd /CAF-YOLO/
python train.py
```
