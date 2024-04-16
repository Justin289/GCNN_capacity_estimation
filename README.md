# Environment requirement
See the requirements.txt file
Use `pip install -r requirements.txt` to install all packages that are used in our program

# Dataset preparation
## Download
Download from [OneDrive](https://1drv.ms/f/s!AnE8BfHe3IOlg13v2ltV0eP1-AgP?e=9o4zgL) and unzip them. More details are shown in our [paper](https://arxiv.org/abs/2201.12358). 

Please make sure the data structure is like the following. 


```
    |--data
        |--battery_dataset1
            |--label
            |--data
        |--battery_dataset2
            |--...
        |--battery_dataset3
            |--...
    
```


## File content

Each `pkl` file is a tuple including two parts. The first part is charging time series
data. `column.pkl` contains column names for the time sequence data. 
The second part is metadata which contains fault label, car number, charge segment number
and mileage. 

The `label` folder contains car numbers and their anomaly labels.

## Generate path information for five-fold validation

In our paper, we provide experiments of training with different brands.
To facilitate the organization of training and test data, we use 1) a python dict to save 
`car number-snippet paths` information, which is named as `all_car_dict.npz.npy`, and 2) a dict to save the
randomly shuffled normal and abnormal car number to perform five-fold training and testing, which is 
named as `ind_odd_dict*.npz.npy`. By default, the code is running on the first brand. So our code
is now running on `ind_odd_dict1.npz.npy`. 

The details of **running experiments on other brands** requires modification on some code, which is 
illustrated in this readme file later. 

To build the `all_car_dict.npz.npy` and `ind_odd_dict*.npz.npy`, run

`cd preprocess`

Run `five_fold_train_test_split.py` and then you get all the files saved in folder
`five_fold_utils\`.
(Running each cell of the `five_fold_train_test_split.py` may take 
a few minutes. If not, please check the data path carefully.)

The cell output of each cell contains randomly shuffled `ind_car_num_list` 
and `ood_car_num_list`. You may print it out to see the car numbers you are using. 

This python file will save two files, `/five_fold_utils/all_car_dict.npz` and 
`/five_fold_utils/ind_odd_dict.npz`

# Battery capacity estimation
## Run GCNN and FT-GCNN
### For file `trans_train.py`:
**Setting another brand:** By default, we are using one brand. To run experiments on other brands, 
you should manually change the variable
`ind_ood_car_dict_path` in `trans_train.py`. 
(An easy way to do so is to use Ctrl+F to search the name of the variables.) 
For example, if you want to use brand 2, 
then you should go back to the `five_fold_train_test_split.py`, inspect the car numbers
 of brand 2 and save the in-distribution and out-distribution numbers as a dict,
 and load it here. 
Various indicators during operation are also recorded in the csv of the `file_path`.


### train
### For file `model_configs.json`:
Please check the `model_configs.json` files carefully for hyperparameter settings. 
You can change `save_model_path` and `data_name` to the dataset you are using.
Use `train_size_ratio` to do the split of trainning dataset, and use `file_ratio` to do the split of time length. 
GCNN:
Set `optuna` to `true` to find optimal parameters, and set `trans_train` to `false` /
FT-GCNN:
Change `model_path` to the best saved model of GCNN on dataset1, and set `trans_train` to `true` to finetuning

To start training, run
```
cd finetuning

python ft-gcn_main.py --config_path model_configs.json --fold_num 0
```
If you want to fully run the five-fold experiments, you should remove the `#` from `line 73` to `line 76` in `ft-gcn_main.py`, and add a `#` at `line 71`
If you want to fully run the train size and time length split experiments, you should remove the `"""` at `line 78` and `line 87` in `ft-gcn_main.py`, and add a `#` from `line 71` to `line 76`
After training, the reconstruction errors of data are recorded  in `save_model_path` configured by the
`json` file, Various indicators during operation are also recorded in the csv of the target path.

## LSTM, MLP, XGB and RandomForest

**Setting another brand:** By default, we are using one brand. 
To run experiments on other brands, 
you should manually generate the `npz` files by `five_fold_train_test_split.py`
and change the variable
`ind_ood_car_dict` in `othermodel_main.py` similarly as above. 
You may change the parameter of LSTM, MLP, XGB and RF by your self

### train
To start training, run
```
cd finetuning
python othermodel_main.py --fold_num 0
```
If you want to fully run the five-fold experiments, you should run five times with different 
`--fold_num`. Or you can write a `.sh` file to loop through five folds.
You can change `--dataset` to the dataset you want to train, or write a `.sh` file to loop through all datasets.

## Calculated RMSE and MAPE score
For all the mentioned algorithms, we calculated the RMSE and MAPE values in their train file. You can check the score either in terminal or in the saving csv files.

## Evaluate and save predict result
You can save preidct result of all models in `evaluate.py` for FT-GCNN and `othermodel_main.py` for other models. Preduct result is saving to `file_path`, you can change to your path.

**Necessary modification:** Since the save path may be time dependent and machine dependent, one needs
to change the path information in each py files.
One should also modify the path of the saved reconstruction error
if one is using different brands. 


``` 


