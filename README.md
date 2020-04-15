The purpose of this project is to create a model which classifies images as good or bad depending on their appearance.
Image dataset is relatively small, so to increase chances of success the transfer learning approach is applied.
As pretrained model I'm using Xception model from Keras library with weights trained on ImageNet dataset.
At the end LIME is used to explain decisions made by the network while classifying images.
