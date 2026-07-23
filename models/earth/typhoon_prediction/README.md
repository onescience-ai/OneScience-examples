---
license: mit
---

Note: Benchmarks in their original form can be found [at the original github repo](https://github.com/kitamoto-lab/benchmarks).


## Instructions to run

#### Docker
All of the below commands should be run in a Docker container built using the Dockerfile in the repo, with the data and repo being exposed as volumes in the container. 

To build:

```docker build  -t benchmarks_img .```

To run an interactive shell:

```docker run -it --shm-size=2G --gpus all -v /path/to/neurips2023-benchmarks:/neurips2023-benchmarks -v /path/to/datasets/:/data benchmarks_img```


### Reanalysis Task
Every command should be run in the reanalysis folder. The path to this folder and to the data should be provided in the config.py file.

#### Create buckets
First, you have to split and save the dataset into 3 buckets according to the type of splitting refered in the config.py file ('standard' for standard splitting between before 2005 / between 2005 and 2015 / after 2015, 'same_size' for the same splitting but with a equal number of sequences per bucket).
```
python3 createdataset.py
```
This will create a folder (named 'save' or 'save_same') with 6 .txt file containing the id of the sequences used for training and testing in each bucket.

#### Train
You can now train for a number of runs (called version in the logs) and epochs specified in the config.py file.
```
python3 train_split.py
```
A tensorboard log while be created for each run with each bucket in the tb_logs.

#### Test
After specifing a list of versions in the config.py file, you'll be able to test the model.
```
python3 split_testing.py
```
The accuracy (RMSE in hPa) will be displayed on the terminal but also written in a log.txt file in the directory ```reanalysis```.