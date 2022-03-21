# CVPR2022_STNet
 
The code is based on [SiamFC++](https://github.com/MegviiDetection/video_analyst) and tested on Ubuntu 20.04 with PyTorch 1.8.0.

##  Test on FE240hz Dataset
1. Download our preprocessed [test dataset](https://drive.google.com/drive/folders/1pNY8kahrof9l9zCw7TtXY4RhvJ4GGx37?usp=sharing) of FE240hz. (The whole FE240hz dataset can be downloaded [here](https://zhangjiqing.com/publication/iccv21_fe108_tracking/)).

2. Download the [pretrained model](https://drive.google.com/file/d/1xD-d24TRoMHRAQKIxE7CxMhI2UffSiUG/view?usp=sharing) and put it into ./snapshots/stnet.

3. Change dataset path at line 32 in videoanalyst/engine/tester/tester_impl/eventdata.py. ```data_root="/your_data_path/img_120_split"```

4. run ``` python main/test.py --config experiments/test/fe240/fe240.yaml ``` the predicted bounding boxes are saved in logs/EVENT-Benchmark/. 
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one event frame.

##  Test on VisEvent Dataset
1. Download our preprocessing [test dataset](https://drive.google.com/drive/folders/1nrHaJysllPOq0VxA1p-Q-4WOr6IVrqO-?usp=sharing) of VisEvent. (The whole VisEvent dataset can be downloaded [here](https://sites.google.com/view/viseventtrack/)).

2. Download the [pretrained model](https://drive.google.com/file/d/17eA45v3XH14qkE5HrAfGc4fcVMUWIEYD/view?usp=sharing) and put it into ./snapshots/stnet.

3. Change dataset path at line 32 in videoanalyst/engine/tester/tester_impl/eventdata.py, ```data_root="/your_data_path/img_120_split"```

4. Change model path at line 25 in experiments/test/fe240/fe240.yaml, ```pretrain_model_path: "snapshots/stnet/fe240.pkl"```

5. run ``` python main/test.py --config experiments/test/fe240/fe240.yaml ``` the predicted bounding boxes are be saved in logs/EVENT-Benchmark/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one event frame.
****
