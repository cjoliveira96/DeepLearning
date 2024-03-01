# Skin Cancer Classification with Deep Learning Techniques

## Project Description

This project aimed to develop a deep learning model for skin cancer classification. The dataset that was used in this project consisted of various skin lesion images, where each was associated with a specific type of skin cancer or lesion. Furthermore, the dataset also included metadata containing personal information such as location of the lesion, age and gender of the patients.

## Dataset Details
- The dataset comprised images from the HAM10000 competition.
- Each image corresponded to a specific skin cancer type, categorized as follows:
    - **Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)**
    - **Basal cell carcinoma (bcc)**
    - **Benign keratosis-like lesions (bkl)**
    - **Dermatofibroma (df)**
    - **Melanoma (mel)**
    - **Melanocytic nevi (nv)**
    - **Vascular lesions (vasc)**
 
Although the data the team used on this project was given by the professor; the following link leads to the same dataset available on [Kaggle](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification).

## Project Objective
Our task was to build a deep learning model using CNN (Convolutional Neural Networks) that was able to accurately classify unseen images of skin lesions (cancer).

Since the group had access to both the images and metadata files, it was also decided to try working with a functional API model, in order to see if it could help improve the accuracy of the predictions of skin cancer. All of these choices can be revised and their results are available in the [Report](Report_Group6.pdf) in this repository.

## Possible Improvements
While this project was able to successfully demonstrate a large efficiency in the classification of types of skin lesions (cancer) the following improvements can still be performed:
1. Application of more image pre-processing techniques 
    - Given that the time for the development of the project, as well as the computational power, was rather limited, the team had to try only a specific range of pre-processing techniques and experiment further with the ones which gave the best results originally. Hence, with more time, more options could be explored to further enhance the performance of this classification algorithm.

2. Experimentation with other types of over or undersampling
   - Since the team had an imbalanced dataset on their hands, it really affected the performance of the model. Due to the lack of computational power, neither total oversampling nor undersampling were options that were tried on this dataset. Hence, they could be interesting options to use in the future.

4. Research on more inclusive datasets
   - The HAM10000 dataset only contains images of lesions on white skin. This means, that if a picture of a lesion on darker skin was to be evaluated by our model it would most likely result in a lower performance of the classification process. Thus, as a way to improve the model and overall research on the efficiency of classification models on cancer prediction, models should be trained on darker skin conditions.

## Repository Description
This repository contains all the final files created during the development of our project. Hence, the following list contains a short description of how this repository is organized and what each file contains:
- [README](README.md): file which contains all the basic information on the project (objectives, motivations, features and improvements);
- [Project Report](Report_Group6.pdf): this is a pdf file of the project report where all the steps of development of the project; reasoning behind every decision and key findings/achievements are summarized;
- [Functional API Notebook](functional_api.ipynb): this is the jupyter notebook were the functional API was developed. Hence, it contains feature engineering steps of the metadata of this project and then the joining of both created models. 
- [Modelling Notebook](modelling.ipynb): this notebook contains all the steps taken in the final version of our project - modelling phase. It contains - EDA, Preprocessing techniques, Modelling and Model Evaluation stages.
- [Path file](path.py): this file is where the path to your local project folder (the place in your pc where you'll have the repository or the notebooks and data) is defined.
- [Utils](utils.py): this python file contains important functions that were used in other parts of the project, but for the sake of optimization and organization where placed on this separate file and imported into the notebooks.

## Project Developed by:
- Bruna Faria | [LinkedIn](https://www.linkedin.com/in/brunafdfaria/)
- Catarina Oliveira | [LinkedIn](https://www.linkedin.com/in/cjoliveira96/)
- Inês Vieira | [LinkedIn](https://www.linkedin.com/in/inesarvieira/)
- Joana Rosa | [LinkedIn](https://www.linkedin.com/in/joanarrosa/) 
- Rita Centeno | [LinkedIn](https://www.linkedin.com/in/rita-centeno/)
##
