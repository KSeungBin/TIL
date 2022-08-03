# CT Lung & Heart & Trachea segmentation
- Segmentation masks for CT reconstruction image  




### Context
---
kaggle link <https://www.kaggle.com/datasets/sandorkonya/ct-lung-heart-trachea-segmentation>  
![segmentation mask](https://user-images.githubusercontent.com/90584177/182589707-3a395a96-5c82-4e12-91df-b5a754de9546.jpg)  






### Content
---
(dataset)|**# of train data**|**# of validation data**
|:-:|:---:|:---:
**CT image**|14910|1798
**mask**|14910|1798  





### Process
---
1. Build tensor dataset
2. Implementation of U-Net architecture : Fully Convolutional Network "Encoder-Decoder structure"
3. Implementation of Cross-entropy, Dice Loss Function
4. Application of optimization technique(SGDM)
5. Implementation of Dice similarity coefficient
     * a spatial overlap index and a reproducibility validation metric
     * Metrics to evaluate your semantic segmnetation model
6. Write and train pixel level classification model
7. Test model and Application of Morphological filtering
