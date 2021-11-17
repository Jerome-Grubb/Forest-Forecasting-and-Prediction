# Forest-Forecasting-and-Prediction

## How to get started
This project has been tested and run on Python 2.7 (3.6 partially tested). This project was used with Tensorflow (v1.5.0) and Keras (v2.0.6)
### 1. Prepare a dataset

A dataset must be split into three parts, which are a Training, Validation and Testing.
Copy the desired dataset into the Data directory by running the following command:
```
cp -r path/to/data/* Data/
```
An alternative option is to go into the Settings.py file, and modifying the DATA_DIR varibale to the directory of the dataset.

Next, it may be necessary to review and adjust some of the settings in the Setting.py file. Many of these can be left at there default values, but some will likely need changed depending on the choosen dataset. Some examples could be the images dimensions (default is 256x256), number of images in each sequence(default 10), or whether there is environmental data (default is False).

The next step is to preprocess the data, which can be done by runnig the following command:
```
python process_data.py
```
This will produce some files that store the image locations, which will be saved under the Data folder. These are used to create batches of images that will be used while training the model.

If there is also environmental data, then the ENV_DATA variable in Settings.py will need to be set to true. Currently the program will presume that the data is stored in three separate CSV files, which will be for testing, training and validation.



### 2. Train the model 

To train the model with the prepared dataset, run the command:
```
python train.py
```
During this process the training and validation loss will display for every epoch.

A copy of the trained model will be saved under the Models folder.

The trained model will output t + 1 predictions. If a longer prediction is required, then the model will need to be finetuned, which is done by running:
```
python  extrap_finetune.py
```
A separate copy of this finetuned model will then be saved under the Models folder.

### 3. Evaluate the model

To test the trained model, run the command:
```
python evaluate.py -ft extrap_image
```
The results of the evaluation will be saved under the Exports folder.
The extrap_image argument is the number of the image in the sequence where the extrapolation will begin. This argument is optional, and if not used the model will be evaluated on t+1 predictions.
