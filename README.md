# Content based Recommendation

## Model performance (accuracy)

* Class 1:
    - Train set: 0.99568
    - Validation set: 0.91117
    - Test set: -

* Class 2:
    - Train set: 0.99716
    - Validation set: 0.95808
    - Test set: -

* Average:
    - Train set: 0.99642
    - Validation set: 0.93463
    - Test set: -

---

## Developer Guide

### Install requirements

``` bash
pyenv virtualenv 3.6.8 p36
pyenv activate p36
pip install -r requirements.txt

``` 


### Preprocess dataset

Basic shell command, 

* If validation set is **not** provided:

``` bash
python make_dataset.py --data data/data.csv -d data/dev_data -v
```

Hyper-parameters used for best model are as follow, 

``` bash
python make_dataset.py --data data/data.csv -d data/dev_data -v --bert_config bert-base-multilingual-cased

``` 

### Train model

* Train model

``` bash
python train.py -md models/dev_model_normal_loss -d data/dev_data/ -e 50 --bert_config bert-base-multilingual-cased

``` 

* Train model with label smoothing, triplet loss and arcface loss

``` bash
python train.py -md models/dev_model_normal_loss -d data/dev_data/ -ls -lt -la -e 50 --bert_config bert-base-multilingual-cased

``` 


### Inference model (extract embedding)

``` bash
python extract_emb.py -d data/<data directory> --model models/<model directory>/model_best.pth.tar --test_data <data file to extract embedding> --embedding_path <file to save extracted embedding (must be .json)>
```

Example, 

``` bash
python extract_emb.py -d data/uni_data/ --model models/uni_model_default_loss/model_best.pth.tar --test_data data/EPG_uni.csv --embedding_path embedding.json
```

### Evaluate trained model on validation set

``` bash
python eval.py -d data/dev_data/ --model models/dev_model_normal_loss/model_best.pth.tar

```

---

### Arguments

Please refer to module/argparser.py

or

``` bash
python train.py -h
```

