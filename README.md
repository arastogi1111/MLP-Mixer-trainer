# MLP-MIXER-KERAS

We present a framework to train a variety of MLP Mixer models. Reference : [Original Paper](https://arxiv.org/abs/2105.01601) by Ilya Tolstikhin et al. [Google Research]

Our pipeline is implemented in Keras and has many functionalities to train, fine-tune, and test it on different datasets.

We provide the user with the option to choose:
-	Model architecture : b16/l16/s16/b32/l32/s32
-	Dataset : Cifar10, Pets, Tiny Imagenet
-	Pretrained Model weights : B16 or L16 with ImageNet 1k or 21k
-	Load a locally saved model
-	Whether to fine-tune, train from scratch, or just test.

The user can further specify hyperparameters like:
-	Activations and whether to use them (original paper did not for first layer)
-	Un-Freezing of layers (Only top or all)
-	Optimizers and learning rates, decays etc.
-	Batch Size
-	Dropout and Drop Connect 

# Environment setup

We have provided reqs.txt file, please use it to recreate an environment.
Otherwise it runs fine with the default ecbm4040 tf24 environment, with additional installation of only a few libraries like pandas, scikit-learn and seaborn needed.

# Dataset Preparation

### Cifar

We automatically handle the preparation for Cifar if it has not been downloaded yet. We upscale Cifar to 224,224 if a pretrained model is being fine-tuned on Cifar.

### Pets
Go to official [Oxford Link](https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz) and download it.
Please extract the 'images' directory from Pets dataset into ../Datasets/PETS/

We automatically handle the re-organization of directories after this.


### Tiny Imagenet

Go to [ImageNet website](https://www.image-net.org/download.php) and make an account. Download tiny-imagenet-200.zip and unzip it at ../Datasets/tiny-imagenet-200/

We handle the loading from here automatically.


# Training

User has 2 options:

### Notebook

Open Jupyter Notebook using

```jupyter notebook```

Open MLP_MIXER_MAIN.ipynb and change whatever you wish in the user_configs dictionary.
Even if user comments out the parameters, we will handle it with default params.
User will be shown the configs before training. Some default parameters may not be used if irrelevant and can be ignored safely.

Run all cells sequentially to get all training and testing results. Plots will be saved as well as displayed on notebook.


### python file

Open run.py and change whatever you wish in the user_configs dictionary.
User will be shown the configs before training. Some default parameters may not be used if irrelevant and can be ignored safely.

Run the python file with nohup for No hangups, useful when training for a high number of epochs.

```nohup python run.py```


# Results

User will have obtained the results of the training in saved_models/ directory under a relevant folder named according to your configs. 

The user can find in the same directory the following:
-	Best saved model (h5 file, can be loaded directly)
-	Tensorboard logs in TBlogs/
-	Csv of training history
-	Plots for both training and validation accuracies 
-	Plots for both training and validation curves
-	Plot for Confusion Matrix 
-	Plot for ROC Curve
-	Testing results
-	Time taken result


## Description of Files 

#### train.py

Has functions to train the model (and other assisting functions to support it) and save plots of accuracies and validation in directies named according to experiment. Handles 

#### data_utils.py

Contains code to automatically handle which Data Generators to load. Contains functions to re-organize directories in a way that it can be fed to the Data Generators. Throws exceptions if requested dataset is not present. 

#### model_utils.py

Has functions to create a new MLP Mixer model based on the architecture requested. Gets the number of classes, and optimizers etc

#### test_utils.py

Has functions to test the model, and plot Confusion matrices and ROC curves.

#### MLP_mixer.py

Has the whole model implementation for MLP mixer and other intermediate blocks. Handles whether we preatrain or finetune, and can reload from a local model. 



### Acknowledgements:
Our MLP Mixer model implementation borrows code from [keras_mlp](https://github.com/leondgarse/keras_mlp) repository by user [leondgarse](https://github.com/leondgarse) to implement the MLP Mixer architecture such that we can reload saved pretrained weights provided by the same repository. However it only provides prediction capability on singular images, and we provide extensive training and testing functionality on large datasets.



![image](https://github.com/ecbme4040/e4040-2023fall-project-gavq/assets/27878610/aeaeb323-e0ea-4c8a-8607-e37a98fac342)
