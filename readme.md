# VAE code

This is an implementation of VAE and its variations, applied to image and text data

## How to use
* Python >= 3.6 is required
* Install requirements
```
$ pip install -r requirements.txt
```
* Run the model (VAE)
```
$ python3 main.py --config 1
```
* Configuration number (--config #)
```
1 : VAE MNIST
2: AAE MNIST
3: ARAE MNIST
4: VQ VAE MNIST
5: LSTM VAE SNLI
6: LSTM AAE SNLI
7: LSTM ARAE SNLI
8: LSTM VQ VAE SNLI
```


## Examples (MNIST data)
* VAE (Image)

  ![VAE_epoch1](./upload_image/vae_mnist_epoch_1.png)
 
  ##### epoch 1
 
  ![VAE_epoch20](./upload_image/vae_mnist_epoch_20.png)
  
  ##### epoch 20

* AAE (Image)

  ![AAE_epoch1](./upload_image/aae_mnist_epoch_1.png)
 
  ##### epoch 1
 
  ![AAE_epoch20](./upload_image/aae_mnist_epoch_20.png)
  
  ##### epoch 20
  
* ARAE (Image)

  ![ARAE_epoch1](./upload_image/arae_mnist_epoch_1.png)
 
  ##### epoch 1
 
  ![ARAE_epoch20](./upload_image/arae_mnist_epoch_20.png)
  
  ##### epoch 20
  
* VQ VAE (Image)

  ![VQVAE_epoch1](./upload_image/vqvae_mnist_epoch_1.png)
 
  ##### epoch 1
  
  ![VQVAE_epoch2](./upload_image/vqvae_mnist_epoch_2.png)
 
  ##### epoch 2
  
  ![VQVAE_epoch3](./upload_image/vqvae_mnist_epoch_3.png)
 
  ##### epoch 3
  
  ![VQVAE_epoch4](./upload_image/vqvae_mnist_epoch_4.png)
 
  ##### epoch 4
  
  ![VQVAE_epoch5](./upload_image/vqvae_mnist_epoch_5.png)
 
  ##### epoch 5
 
  ![VQVAE_epoch20](./upload_image/vqvae_mnist_epoch_20.png)
  
  ##### epoch 20

## Examples (SNLI data)
* LSTM-AE Reconstruction
  - A man in a black shirt is looking at a bike in a workshop .
  - A man in a blue shirt is looking at a car in a distance .

## References
* Papers
  1. [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114)  
  2. [Adversarial Autoencoders (AAE)](https://arxiv.org/abs/1511.05644)
  3. [LSTM VAE](https://arxiv.org/abs/1511.06349)
  4. [Adversarially Regularized Autoencoders (ARAE)](https://arxiv.org/abs/1706.04223)
  5. [Neural Discrete Representation Learning (VQ VAE)](https://arxiv.org/abs/1711.00937)

* Codes
  1. [Pytorch VAE](https://github.com/pytorch/examples/tree/master/vae)
  2. [Pytorch AAE](https://github.com/bfarzin/pytorch_aae)
  3. [Pytorch Sentence VAE](https://github.com/timbmg/Sentence-VAE)
  4. [Pytorch VQ VAE](https://github.com/zalandoresearch/pytorch-vq-vae)
  5. [Pytorch ARAE](https://github.com/jakezhaojb/ARAE)





