# Making and Autoencoder Using PyTorch and training on MNIST
### 1. Test Result  
---
![image](https://user-images.githubusercontent.com/90584177/183294244-61d7b820-24de-4cf1-8eff-de5a94fb7c6d.png)

<p class="callout info">How to troubleshoot this autoencoder's bad reconstruction</p>

> 1. Bottleneck layer is too narrow

    - The bottleneck size was set to 2 for latent space visualization
    - Therefore, the resulting algorithm missed important dimensions(features) for the problem
    - Optimal bottle neck size is 10 to 20

> 2. Increasing the number of epochs
    
    - Even though the validation loss steadily decreased, training was stopped at 50 epochs due to limits in computing power



### 2. Latent Space Visualization 
---
embedding of 784-dimensional inputs within a 2-dimensional manifold   
where input vectors which resemble each other more closely are positioned closer to one another in the latent space  
![image](https://user-images.githubusercontent.com/90584177/183295267-7ab177fd-6354-4613-bdc2-f4d40e79febe.png)


### 3. Latent Space Inference
---
Inference about (-2,2) on the x-axis and (-2,2) on the y-axis of the image above
![image](https://user-images.githubusercontent.com/90584177/183295550-5f7c6869-70e5-49aa-921d-2fcdfa500826.png)
