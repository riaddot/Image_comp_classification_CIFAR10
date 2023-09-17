# Image_comp_classification_CIFAR10
Training a model (Encoder and a Classifier) with a standard decoder capable of compressing image and using the latent representation for classification 

In this model we took the weight of the decoder in **High Fidelity Generative Image Compression** and we trained both the Encoder and the Classifier. with a loss function that combines both the restoration loss and the classificatino loss. 
We used CIFAR-10 as our dataset.

The global architecture can be shown in the following figure: 

![image](https://github.com/riaddot/Image_comp_classification_CIFAR10/assets/100699369/804a6ca8-5283-426c-ad8c-168424bd1514)
