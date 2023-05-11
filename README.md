# Bone-supression-models
This repository contains the implementation and deployment of various Deep learning models for obtaining bone suppression images in chest x-ray images. Gradually new models will be added.


![Bonse supression imgae](bsi.PNG)

# Dataset
To train the models forced a public data set available on the kaggle platform [here](https://www.kaggle.com/datasets/hmchuong/xray-bone-shadow-supression). El dataset cuenta con dos versiones de los datos una versión contiene 4080 pares de imágenes de rayos x de tórax y su correspondiente  imagen de supresión osea,  la otra contiene  241 pares de imágenes originales sin ninguna técnica de aumentacion de datos.
Bone suppression images were obtained using the Dual energy subtraction technique. They have a resolution of 1024x1024.

# Training
The preprocessing and training routine for each specific model is distributed in different jupyter notebooks in the [training scripts](train scripts) folder.

# Models
The models folder contains the deployed and deployed models. So far two models are available.
unet_resnet50: Based on a unet architecture with a pretrained resnet50 as encoder and a single channel attention technique.
gan_mso2: Model based on a pix2pix GAN architecture.

# Requirements
Each implemented model has its own requirements. The [model requirement folder](model requirement) contains several .txt files where the requirements of each model are described. The [app_requirement.txt](app_requirements.txt.txt) file contains the requirements for the model deployment script.
#### Deployment script requirements
1. gradio
2. opencv-python-headless
3. fastai
4. numpy
5. tensorflow

# Try the models
The models can be used by running the [app.py](app.py) script. This script displays a gradio interface that makes it easy to test the deployed models. You can also try it through this hugging face space.
