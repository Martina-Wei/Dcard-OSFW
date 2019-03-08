# Dcard-OSFW
Only Sex For Work : Sex Post Image Classification
####  西斯版圖片分類器

# Notes
* 在訓練資料裡面我們分成六類
  *  全部版都不可以(NSF_other, 2044張)
  *  全部版都可以(SF_other, 10021張)
  *  西斯版可以(SF_sex, 10503張)
  *  女生殖器(pussy, 1355張)
  *  男生殖器(cock, 4190張)
  *  乳頭(nipple, 2546張)
  
* 目前將密集恐懼症(trypo)以及昆蟲(bugs)等類別拿掉，由於資料過少的關係，在訓練之前請先從dataset內移出
* 使用Transfer Learning並且用Nasnet進行fine tune
* 準確率目前為Training 98%, Testing 83%. 有overfit以及data imbalance情形發生.
* 在越多資料加入之後可獲得改進
* <img src="https://github.com/Dcard/Dcard-OSFW/blob/master/confuse.png?raw=true" width="75%" align=center />
* Classification Report

precision    recall  f1-score   support

   NSF_other       0.72      0.59      0.65       228
    SF_other       0.85      0.96      0.90      1113
      SF_sex       0.82      0.84      0.83      1169
        cock       0.93      0.88      0.91       466
     nipples       0.77      0.64      0.70       283
       pussy       0.81      0.51      0.63       151

   micro avg       0.83      0.83      0.83      3410
   macro avg       0.81      0.74      0.77      3410
weighted avg       0.83      0.83      0.83      3410

# Usage
### Install
```
pip3 install -r requirements.txt
```

###  Preprocess
此步驟將資料分成訓練資料(Training dataset)以及驗證資料(Validation dataset)
```
python3 preprocess.py 
  --src_dir_path <還沒分開的資料夾位置> 
  --des_path <目標位置>
  [--action split_data] 
```
執行結束之後會在<目標位置>出現train以及valid兩個資料夾
###  Train
利用分好的資料訓練模型 \
訓練時會經過兩階段訓練，首先將pretrain model固定weights之後訓練 \
接著將pretrain model解鎖訓練整個模型得到最好的accuracy \
訓練過程中將會儲存最高accuracy的model防止overfit太嚴重
```
python3 train.py 
  --dir_path <目標位置> 
  [--batch_size_1 32]
  [--batch_size_2 6]
  [--lr_1 0.0005]
  [--lr_2 0.00002]
  [--epochs_1 5]
  [--epochs_2 20]
  [--workers 16]
  [--skip_step_1 False]
```
###  Evaluation
評估model的準確率
```
python3 evaluate.py
  --dir_path <目標位置> 
  --model_path <model weight file>
  [--batch_size 32]
  [--lr 0.0005]
  [--workers 16]
```
###  Predict
將尚未分類的圖片經過分類器之後自動擺到對應的資料夾中 \
“打開它，機器就會幫你分好，多棒”
```
python3 predict.py
  --ref_path <有train跟valid資料的資料夾位置>
  --src_path <還沒分類好的資料夾位置>
  --des_path <目標位置> 
  --model_path <model weight file>
```
