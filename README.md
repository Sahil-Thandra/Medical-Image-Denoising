# Medical-Image-Denoising

In this project a deep learning based model is proposed for medical image denoising using convolutional denoising autoencoders. Two datasets, mini-MIAS database of mammograms(MMM) and a dental radiography database(DX) are used. These datasets are pre-processed and corrupted with gaussian noise. These corrupted images are used as inputs to the convolutional autoencoder model. The resulting images were compared with the original images using structural similarity index measure (SSIM) for the consistency and accuracy of the model. Experiments verify the advantages of the model compared with NL means and median filter method.

## Framework Used

The project was done under Torch framework using luarocks modules: nn, nnx, dpnn, image and optim. The images were pre-processed and compared in MATLAB

## Data

The two datasets used were:
Mini-MIAS database of mammograms(MMM)-http://www.mammoimage.org/databases/
Dental radiography database(DX)-http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge1/index.html
MMM has 322 images of 1024 × 1024 resolution and DX has 400 cephalometric X-ray images collected from 400 patients with a resolution of 1935 × 2400. 

## Files Description

data.lua: file to load training and testing images (both noisy and original) to torch.CudaTensors

main.lua: file in which the Convolutional De-noising Autoencoder architecture and the evaluating loss function is defined. The weights and biases of the model are initialised and the model is then trained on the training data. The weights and biases of the trained model are saved as well as the outputs obtained when testing images were forwarded.

train.lua: file which loads pre-trained models and saves new weights and biases after further training.

eval.lua: file to evaluate trained models on testing data.

ssim_score.m: file to compare the original and output images and obtain their ssim score.

data_corruption.m: file to introduce gaussian noise in original images to form the input dataset.

## Futher Description of the project

https://docs.google.com/document/d/15XZjyqMLpEEv-iWuyCCopyZIcK54inKtJkZgZSQfRNc/edit?usp=sharing
