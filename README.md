# Forest-Forecasting-and-Prediction

# How to get started

### 1. Prepare a dataset

Copy the dataset that the model is to be trained on into the Data directory. This can be done by ruunning the following command:
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
python evaluate.py
```
The frame that the model starts making predictions from can be modified in the stttings.py file, by changing the EXTRAP variable (default is 5).
The results of the evaluation will be saved under the Exports folder.

### 4. Longer Extrapolation of Predictions

To obtain predictions with a longer extrapolation, run the following command:

Note: A finetuned model is required for this step. Information about how to finetune a model can be found in step 2.
```
# python predict.py test_images_directory -pf number_of_predictions 
```
The resulting predictions will be saved under the Exports folder.
