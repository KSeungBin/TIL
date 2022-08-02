# Chest X-Ray Images
[kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


## Context
![X-ray image](https://user-images.githubusercontent.com/90584177/182315473-a422f5f2-3336-4be8-b802-64ea6f273299.png)


## Content
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Normal/Covid/Pneumonia).


## Process
1. Build image dataset
2. Convert the image to tensor using torchvision transforms library 
3. Fine tune a pretrained VGG19 model : custom model does not classify 1000 classes but 3 classes by modifying header(FC Layer)
4. Understanding and Application of cross-entropy loss function to optimize classification model
5. Understanding and Application of optimization technique : SGDM(SDG + Momentum)
6. Understanding of Deep Learning Results similar to the principles of Human Reasoning   
     -> There is no clear difference between normal and pneumonia, making it difficult for deep learning models to infer just like humans.
