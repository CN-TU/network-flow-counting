# network-flow-counting
Contact: Fares Meghdouri

This repository contains the code and data for our paper **Unveiling Flows in the Dark: Analysis of Encrypted Tunnels**

# Datasets
## Used in the Paper

## Extract Your Own Data
Start by downloading the following pcap files: 
* [Training and Validation Data](https://mawi.wide.ad.jp/mawi/samplepoint-G/2020/202006101400.html)
* [2nd Test Data](https://mawi.wide.ad.jp/mawi/samplepoint-F/2021/202103151400.html)
The rest is available soon.

# Usage Examples
The scaler objects are objects that store the scikitlearn StandardScaler models. We scale the data once during training and then we use the same model for testing since test data is considered as unknown and thus, can't be scaled with its knowledge. Hence, you notice when calling the script, we always load the scaler except for sequences of length 500 where it is created for the first time.

## Using our pre-trained models
* Get metrics presented in the paper for sequences of length 500
```sh
python learn.py --task count --window 500 --model best_models/best_of_std/500.hdf5 --evaluate --reset_scalers
```
* The same thing for sequences of length 20
```sh
python learn.py --task count --window 20 --model best_models/best_of_std/20.hdf5 --evaluate --reset_scalers --data_scaler "scaler_objects/data_scaler"
```

## Use the pre-trained model with your data
```sh
python learn.py --task count --window 20 --model best_models/best_of_std/20.hdf5 --evaluate --reset_scalers --data_scaler "scaler_objects/data_scaler" --external_data --dataroot your_data_path
```

## Train your own model and evaluate it
```sh
python learn.py --task count --dataroot your_data_path --window 50 --function train --evaluate --working_dir path_to_tmp
```
PS: many files will be generated such as scalers, models etc. and put into `path_to_tmp`

To generate Table2 of the paper:
```sh
python learn.py --task count --window 500 --model best_models/best_of_std/500.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate
python learn.py --task count --window 100 --model best_models/best_of_std/100.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate
python learn.py --task count --window 50 --model best_models/best_of_std/50.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate
python learn.py --task count --window 20 --model best_models/best_of_std/20.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate

 python learn.py --task count --window 500 --model best_models/best_of_std/500.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test1.csv
 python learn.py --task count --window 100 --model best_models/best_of_std/100.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test1.csv
python learn.py --task count --window 20 --model best_models/best_of_std/20.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test1.csv
 python learn.py --task count --window 50 --model best_models/best_of_std/50.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test1.csv

 python learn.py --task count --window 500 --model best_models/best_of_std/500.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test2.csv
 python learn.py --task count --window 100 --model best_models/best_of_std/100.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test2.csv
 python learn.py --task count --window 50 --model best_models/best_of_std/50.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test2.csv
python learn.py --task count --window 20 --model best_models/best_of_std/20.hdf5 --reset_scalers --data_scaler "scaler_objects/data_scaler" --evaluate --external_data --dataroot data/test2.csv
```

Further documentation will be available soon.
