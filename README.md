# paperarXiv
Papers for Deep Learning
- with [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/)
> 논문을 읽고 다양한 생각해보기 

# deep-learning-papers
- [JONGSKY](https://github.com/JONGSKY/paper)
## Computer Vision
### CNN Architecture
* AlexNet: ImageNet Classification with Deep Convolutional Neural Networks 
* ZFNet (DeconvNet): Visualizing and Understanding Convolutional Networks ([pdf](https://arxiv.org/pdf/1311.2901.pdf), [note](https://drive.google.com/open?id=1bzkoKVxLALaD6ZWQh5-vP9qOodMdIwi0), code)
* NIN: Network in Network
* VggNet: Very Deep Convolutional Networks for Large-Scale Image Recognition
* GoogLeNet: Going Deeper with Convolutions
* ResNet:
  - ResNet-v1: Deep Residual Learning for Image Recognition ([note](https://drive.google.com/open?id=1Ahws2bBE_YSjvNcxsF9tnwRCv3HfaVhr))
  - ResNet-v2: Identity Mappings in Deep Residual Networks
  - Wide Residual Networks ([note](https://drive.google.com/open?id=14eQSeymwXgS7JvBbAkOnudAm6MFnJify), code)
* InceptionNet:
  - Inception-v1: GoogLeNet
  - Inception-v2, v3: Rethinking the Inception Architecture for Computer Vision ([note](https://drive.google.com/open?id=1SVOpf9aElrAGCZHlX7NvYL8pbehXpw8i), code)
  - Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning 
* DenseNet:
* NASNet: Learning Transferable Architectures for Scalable Image Recognition ([note](https://drive.google.com/open?id=1o1SfbVIgEhRWGQG_mPpxKoCDyWtNoifJ), code)
* EfficientNet:([note](https://drive.google.com/open?id=1LtdSId0HTpM8_O4k4WFrzHz4ldPf7dTu))


### [Visualizing CNNs](./doc/visualizing_cnn.md)
* DeconvNet
* BP: Deep inside convolutional networks: Visualising image classification models and saliency maps ([note](https://drive.google.com/open?id=1IBP1uMr08hBp3bKjvyNnwFMu0S8ORGcs))
* Guided-BP (DeconvNet+BP): Striving for simplicity: The all convolutional net ([note](https://drive.google.com/open?id=1KUq5-h_xVmjd4FudGDeBUfPV9vBMHV68), code)
* Understanding Neural Networks Through Deep Visualization


### [Weakly Supervised Localization](./doc/cam.md)
* From Image-level to Pixel-level Labeling with Convolutional Networks (2015)
* GMP-CAM: Is object localization for free? - Weakly-supervised learning with convolutional neural networks (2015) ([note](https://drive.google.com/open?id=1Xpnhq0snjkPMsxKLpmhLOpoZgfFlL9H3), code)
* GAP-CAM: Learning Deep Features for Discriminative Localization (2016) ([note](https://drive.google.com/open?id=1lrkE07E3bnLscAnScwq0OIO3AaRHrqnb), code)
* c-MWP: Top-down Neural Attention by Excitation Backprop
* Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (2017) ([note](https://drive.google.com/open?id=10obbO7F2igia6gCcc9IqxATzOxdoPl7L), code)


### [Object Detection](./doc/detection.md)
* OverFeat - Integrated Recognition, Localization and Detection using Convolutional Networks ([note](https://drive.google.com/open?id=1O3j-ag0pPRbRjG4ovWmxnZIwhUJ0twvK), code)


### [Semantic Segmentation](./doc/semantic_segmentation.md)
* FCN_V1 (2014)에서 직접적인 영향을 받은 모델들:
  * FCN + max-pooling indices를 사용: SegNet V2 (2015) ([note](https://drive.google.com/open?id=1CDNkW-3LKVDjGAyPCgj8fOz78pMY0Pd7))
  * FCN 개선: Fully Convolutional Networks for Semantic Segmentation (FCN_V2, 2016) ([note](https://drive.google.com/open?id=1Kr2-ZdiqKmsgXP2ofaUZm_PT5UbbTyDN), [code](https://github.com/bt22dr/CarND-Semantic-Segmentation/blob/master/main.py))
  * FCN + atrous convolution과 CRF를 사용: DeepLap V2 (2016)
  * FCN + Dilated convolutions 사용: Multi-Scale Context Aggregation by Dilated Convolutions (2015)
  * FCN + Multi-scale: Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture (2015)
  * FCN + global context 반영 위해 global pooling 사용: ParseNet (2015) 
  * FCN + 모든 레이어에서 skip 사용: U-Net (2015) ([note](https://drive.google.com/open?id=1Up8PiwA79J8R3ScjgYTmLzt8pYYa83EN))
* PSPNet (2016) ([note](https://drive.google.com/open?id=1xPu7Z-0jWepxb1av9fG2Py72Yz0enWym))
* DeepLabv3+ (2018) ([note](https://drive.google.com/open?id=1YFUdcwKzIrTzfmL6o94y01tDXsZ2n6vc))
* EncNet (2018)
* FastFCN (2019)
* Instance Segmentation
  * DeepMask
  * SharpMask
  * Mask R-CNN (2017) ([note](https://drive.google.com/open?id=1kFVOdctJTcWYkflfCM1Ys-J7Fo8COC6R))
* 3D / Point Cloud
  * PointNet (2017)
  * SGPN (2017)
* Weakly-supervised Segmentation

### [Style Transfer](./doc/style_transfer.md)
* ~~A Neural Algorithm of Artistic Style (2015)~~
* Image Style Transfer Using Convolutional Neural Networks (2016)
* Perceptual Losses for Real-Time Style Transfer and Super-Resolution (2016)
* Instance Normalization: 
  * Instance Normalization: The Missing Ingredient for Fast Stylization (2016)
  * Improved Texture Networks: Maximizing Quality and Diversity in Feed-forward Stylization and Texture Synthesis (2017)


### Siamese, Triplet Network
* Triplet Network
  * FaceNet: A Unified Embedding for Face Recognition and Clustering ([note](https://drive.google.com/open?id=1E9ZGncIvpJoPK5_mSq5J-Mn33r2_xAqj), code)
  * Learning Fine-grained Image Similarity with Deep Ranking ([note](https://drive.google.com/open?id=1BrjRlzB139v5nmCgdLruJGn1drmBb33m), code)
* Siamese Network


### Mobile
* Shufflenet: An extremely efficient convolutional neural network for mobile devices
* Mobilenets: Efficient convolutional neural networks for mobile vision applications


### [Etc.](./doc/etc/md)
* A guide to convolution arithmetic for deep learning ([note](https://drive.google.com/open?id=1zGGzI4qc49u5zV0jFSkzD8xDMY0OalN1))


## [Generative Models](./doc/gan.md)

### Models

#### Autoregressive Models
* NADE (2011)
* RNADE (2013)
* MADE (2015)
* PixelRNN 계열
  * PixelCNN (2016): ([note](https://drive.google.com/open?id=1G_iIjf9dIWqge21sxrpcqK2L76PY8elN), [code1(mnist)](./code/PixelCNN_mnist.ipynb), [code2(fashion_mnist)](./code/PixelCNN_fashionmnist.ipynb))
  * WaveNet (2016) ([note](https://drive.google.com/open?id=1qnNQS_aFuPly8MVO7kSPytPAgf-KifbC), [code](./code/WaveNet.ipynb))
  * VQ-VAE: Neural Discrete Representation Learning

#### Variational Autoencoders
* VAE ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/doc/gan.md#auto-encoding-variational-bayes), [code1(mnist)](./code/vae.ipynb), [code2(fashion_mnist)](./code/vae_fashion_mnist.ipynb))
* Conditional VAE ([note](https://drive.google.com/open?id=1f9fGvvtj-FdPJRwtw7PFLW0ysAu-U_2O), [code](./code/conditional_vae_fashion_mnist.ipynb))
* VAE-GAN: Autoencoding beyond pixels using a learned similarity metric

#### Normalizing Flow Models
* NICE (2014) ([note](https://drive.google.com/open?id=1Bz8i8lASNr8SS6vBraDOyOPTJY61q2aG))
* Variational Inference with Normalizing Flows (2015) ([code](https://drive.google.com/drive/folders/1-kqyXOvnuw7aeOwbAfKO2OWUkFdv3AmR))
* IAF (2016)
* MAF (2017)
* Glow (2018)

#### GANs
* GAN: Generative Adversarial Networks ([note](https://drive.google.com/open?id=1gymav6NryH-0AJqJ7hRt6SzFLrX8bCIn), [code1(mnist)](./code/gan.ipynb), [code2(fashion_mnist)](./code/gan_fashion_mnist.ipynb))
* DCGAN ([note](https://drive.google.com/open?id=1IWeM32QDq97mQ8BdA-rWa58AiRqWTepG), [code1](./code/dcgan_mnist.ipynb), [code2](./code/dcgan_celebA.ipynb))
* WGAN 계열: 
  * WGAN: Wasserstein GAN ([note(진행중)](https://drive.google.com/open?id=1CnfvynSKj9apRZBLjzWB--QJ69PKj2wy), [code](./code/wgan.ipynb))
  * WGAN_GP: Improved Training of Wasserstein GANs 
  * CT-GAN: Improving the Improved Training of Wasserstein GANs
* infoGAN
* Improved GAN: 
* SNGAN: Spectral Normalization for Generative Adversarial Networks ([note(진행중)](https://drive.google.com/open?id=1qJmWsSKPQ2yXQDh68KcZsdEHeAkRvcgZ), [code](./code/sngan_fashion_mnist.ipynb))
* SAGAN: 
* CoGAN: Coupled Generative Adversarial Networks (note, code)

### [Image generation](./doc/img2img_translation.md)
#### image-to-image
* cGAN: Conditional Generative Adversarial Nets (2014) ([note](https://drive.google.com/open?id=1z1flvsqORItbCTZEGuFBQmubvNnAGzs-), [code](./code/cgan.ipynb))
* (내맘대로)pix2pix 계열:
  * pix2pix: Image-to-Image Translation with Conditional Adversarial Networks (2016) ([note](https://drive.google.com/open?id=1GYphdvvfuyb-YKDd_ItvwCLkNDqtY_5F))
  * ~~pix2pixHD~~
  * CycleGAN: 
  * BicycleGAN
  * vid2vid: Video-to-Video Synthesis
  * SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization (2019) ([note](https://drive.google.com/open?id=1CtbIRJ7wi7-wH9h7zQiClcHeqcfdoHEU))
* StarGAN: 
* PGGAN: 
* ~~UNIT/MUNIT~~
* ~~iGAN~~
* StyleGAN: 

#### text-to-image
* Generative adversarial text to image synthesis
* StackGAN
* AttnGAN

### Sequence generation
* WaveGAN: ([note](https://drive.google.com/open?id=1zd4pw884TztzisixmJSJDYTXcdXZSbdo), code)
* SeqGAN:

### Evaluation
* A note on the evaluation of generative models
* A Note on the Inception Score

### CS236 (Deep Generative Models)
* Introduction and Background ([slide 1](https://drive.google.com/open?id=1y9-nkh9OhxAuRP009FsZUDB8hjb3b9iG), [slide 2](https://drive.google.com/open?id=1Kmd7lnZJTw-mgwcTR91nWRVdxQ9Ot5X7))
* Autoregressive Models ([slide 3](https://drive.google.com/open?id=18l4h4iQ_lAROCOlKf44VGk-Q7DEvtBp6), [slide 4](https://drive.google.com/open?id=1IQ5LdSyO9UXi3yjh9c_m7AhK0jmIqQNF))
* Variational Autoencoders 
* Normalizing Flow Models 
* Generative Adversarial Networks 
* Energy-based models 

## NLP
* Recent Trends in Deep Learning Based Natural Language Processing ([note](https://drive.google.com/open?id=12dosro89x1wy3wXUfJ26oa3_2hU_S6n6))

### RNN Architecture
* Seq2Seq
  - Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation (2014)
  - Sequence to Sequence Learning with Neural Networks (2014) ([note](https://drive.google.com/open?id=1crgnXU-3JClMsF1ZYLr8bxxWkSZjcTEZ), code)
  - A Neural Conversational Model
* Attention
  - (Luong) Effective Approaches to Attention-based Neural Machine Translation (2015)
  - (Bahdanau) Neural Machine Translation by Jointly Learning to Align and Translate (2014) ([note](https://drive.google.com/open?id=1YJLljd9YbOzW5mADOtAT7V3imfMBi3Fg), code)
  - Transformer: Attention Is All You Need (2017)
* Memory Network
  - Memory Networks (2014) 
  - End-To-End Memory Networks (2015)
* Residual Connection
  - Deep Recurrent Models with Fast-Forward Connections for NeuralMachine Translation (2016)
  - Google's Neural MachineTranslation: Bridging the Gap between Human and Machine Translation (2016)
* CNN
  - Convolutional Neural Networks for Sentence Classification (2014)
  - ByteNet: Neural Machine Translation in Linear Time (2017)
  - Depthwise Separable Convolutions for Neural Machine Translation (2017)
  - SliceNet: Convolutional Sequence to Sequence Learning (2017)
### Word Embedding

## Multimodal Learning
* DeVise - A Deep Visual-Semantic Embedding Model: ([note](https://drive.google.com/open?id=19gr2FsgvfUAHHA4E25UFJAFmFSUD_L4k))
* Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models: ([note](https://drive.google.com/open?id=1qytxJgLsZvlbazeXTfExqPA-2rUX1irB))
* Show and Tell: ([note](https://drive.google.com/open?id=1fZJ7jopShsepyderJ03ivzoiGqMUOGl3))
* Show, Attend and Tell: ([note](https://drive.google.com/open?id=1COSvkFUWxuotzicGFAlLbl5l_iRrsTtG))
* Multimodal Machine Learning: A Survey and Taxonomy: ([note](https://drive.google.com/open?id=1qibjIoD5z6HjC_G6ICixpA_G0-A5P8t5))


## [Etc.](./doc/etc.md) (Optimization, Normalization, Applications)
* An overview of gradient descent optimization algorithms ([note](https://drive.google.com/open?id=1eSNr4zQBKbQQRpxDl06AWEwAU5qysTo_))
* Dropout:
* Batch Normalization: ([pdf+memo](https://drive.google.com/open?id=1rSM2Q510EjEZ3J6YpWH_ZwPiUS2JHQTp), code)
* How Does Batch Normalization Help Optimization?
* Spectral Norm Regularization for Improving the Generalizability of Deep Learning ([note(진행중)](https://drive.google.com/open?id=1_Th_cpo5rgTyQqi3085To_YgyNCmwlzL), code)
* Wide & Deep Learning for Recommender Systems
* Xavier Initialization - Understanding the difficulty of training deep feedforward neural networks
* PReLU, He Initialization - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification



## [Drug Discovery](./doc/medical.md)

## Basic-Advanced Papers

- with : [처음 배우는 머신러닝](https://github.com/underthelights/paperarXiv/blob/main/references/%EC%B2%98%EC%9D%8C%EB%B0%B0%EC%9A%B0%EB%8A%94%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D.md) 목차 논문들
> references from..
  > terry um [awesome DL papers](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md)
    > 

## Contents

* [Understanding / Generalization / Transfer](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#understanding--generalization--transfer)
* [Optimization / Training Techniques](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#optimization--training-techniques)
* [Unsupervised / Generative Models](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#unsupervised--generative-models)
* [Convolutional Network Models](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#convolutional-neural-network-models)
* [Image Segmentation / Object Detection](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#image-segmentation--object-detection)
* [Image / Video / Etc](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#image--video--etc)
* [Natural Language Processing / RNNs](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#natural-language-processing--rnns)
* [Speech / Other Domain](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#speech--other-domain)
* [Reinforcement Learning / Robotics](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#reinforcement-learning--robotics)
* [More Papers from 2016](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#more-papers-from-2016)

*(More than Top 100)*

* [New Papers](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#new-papers) : Less than 6 months
* [Old Papers](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#old-papers) : Before 2012
* [HW / SW / Dataset](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#hw--sw--dataset) : Technical reports
* [Book / Survey / Review](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#book--survey--review)
* [Video Lectures / Tutorials / Blogs](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#video-lectures--tutorials--blogs)
* [Appendix: More than Top 100](https://github.com/terryum/awesome-deep-learning-papers/blob/master/README.md#appendix-more-than-top-100) : More papers not in the list

* * *
  > sungjoon choi lectures

---
- 인공지능 및 기계학습
- 개론 1, 2 (문일철 교수님, KAIST 산업및시스템공학과)


- **CHAPTER 1. Motivations and Basics**

**[강좌 수강을 환영합니다! 여기부터 꼭 보고 넘어가세요-!](https://www.edwith.org/machinelearning1_17/lecture/40603?isDesc=false)
[1.1. Motivations](https://www.edwith.org/machinelearning1_17/lecture/10574?isDesc=false)
[1.2. MLE](https://www.edwith.org/machinelearning1_17/lecture/10575?isDesc=false)
[1.3. MAP](https://www.edwith.org/machinelearning1_17/lecture/10576?isDesc=false)
[1.4. Probability and Distribution](https://www.edwith.org/machinelearning1_17/lecture/10577?isDesc=false) 
[Ch1. Quiz](https://www.edwith.org/machinelearning1_17/quiz/10578?isDesc=false)**

**CHAPTER 2. Fundamentals of Machine Learning**

**[2.1. Rule Based Machine Learning Overview](https://www.edwith.org/machinelearning1_17/lecture/10579?isDesc=false)
[2.2. Introduction to Rule Based Algorithm](https://www.edwith.org/machinelearning1_17/lecture/10580?isDesc=false)
[2.3. Introduction to Decision Tree](https://www.edwith.org/machinelearning1_17/lecture/10581?isDesc=false)
[2.4. Entropy and Information Gain](https://www.edwith.org/machinelearning1_17/lecture/10582?isDesc=false)**

**[2.5. How to create a decision tree given a training dataset](https://www.edwith.org/machinelearning1_17/lecture/10583?isDesc=false)
[Ch2. Quiz](https://www.edwith.org/machinelearning1_17/quiz/10584?isDesc=false)**

**CHAPTER 3. Naive Bayes Classifier
[3.1. Optimal Classification](https://www.edwith.org/machinelearning1_17/lecture/10585?isDesc=false)
[3.2. Conditional Independence](https://www.edwith.org/machinelearning1_17/lecture/10586?isDesc=false)[3.3. Naive Bayes Classifier](https://www.edwith.org/machinelearning1_17/lecture/10587?isDesc=false)
[3.4. Naive Bayes Classifier Application (Matlab Code)](https://www.edwith.org/machinelearning1_17/lecture/10588?isDesc=false)
[Ch3. Quiz](https://www.edwith.org/machinelearning1_17/quiz/10589?isDesc=false)**

**CHAPTER 4. Logistic Regression
[4.1. Decision Boundary](https://www.edwith.org/machinelearning1_17/lecture/10590?isDesc=false)
[4.2. Introduction to Logistic Regression](https://www.edwith.org/machinelearning1_17/lecture/10591?isDesc=false)
[4.3. Logistic Regression Parameter Approximation 1](https://www.edwith.org/machinelearning1_17/lecture/10592?isDesc=false)
[4.4. Gradient Method](https://www.edwith.org/machinelearning1_17/lecture/10593?isDesc=false)
[4.5. How Gradient method works](https://www.edwith.org/machinelearning1_17/lecture/10594?isDesc=false)
[4.6. Logistic Regression Parameter Approximation 2](https://www.edwith.org/machinelearning1_17/lecture/10595?isDesc=false)
[4.7. Naive Bayes to Logistic Regression](https://www.edwith.org/machinelearning1_17/lecture/10596?isDesc=false)
[4.8. Naive Bayes vs Logistic Regression](https://www.edwith.org/machinelearning1_17/lecture/10597?isDesc=false)
[Ch4. Quiz](https://www.edwith.org/machinelearning1_17/quiz/10598?isDesc=false)**

**CHAPTER 5. Support Vector Machine
[5.1. Decision Boundary with Margin](https://www.edwith.org/machinelearning1_17/lecture/10599?isDesc=false)
[5.2. Maximizing the Margin](https://www.edwith.org/machinelearning1_17/lecture/10600?isDesc=false)
[5.3. SVM with Matlab](https://www.edwith.org/machinelearning1_17/lecture/10601?isDesc=false)
[5.4. Error Handling in SVM](https://www.edwith.org/machinelearning1_17/lecture/10602?isDesc=false)
[5.5. Soft Margin with SVM](https://www.edwith.org/machinelearning1_17/lecture/10603?isDesc=false)
[5.6. Rethinking of SVM](https://www.edwith.org/machinelearning1_17/lecture/10604?isDesc=false)
[5.7. Primal and Dual with KKT Condition](https://www.edwith.org/machinelearning1_17/lecture/10605?isDesc=false)
[5.8. Kernel](https://www.edwith.org/machinelearning1_17/lecture/10606?isDesc=false)
[5.9. SVM with Kernel](https://www.edwith.org/machinelearning1_17/lecture/10607?isDesc=false)
[Ch5. Quiz](https://www.edwith.org/machinelearning1_17/quiz/10608?isDesc=false)**

**CHAPTER 6. Training Testing and Regularization
[6.1. Over-fitting and Under-fitting](https://www.edwith.org/machinelearning1_17/lecture/10609?isDesc=false)
[6.2. Bias and Variance](https://www.edwith.org/machinelearning1_17/lecture/10610?isDesc=false)
[6.3. Occam's Razor](https://www.edwith.org/machinelearning1_17/lecture/10611?isDesc=false)
[6.4. Cross Validation](https://www.edwith.org/machinelearning1_17/lecture/10856?isDesc=false)
[6.5. Performance Metrics](https://www.edwith.org/machinelearning1_17/lecture/10860?isDesc=false)
[6.6. Definition of Regularization](https://www.edwith.org/machinelearning1_17/lecture/10863?isDesc=false)
[6.7. Application of Regularization](https://www.edwith.org/machinelearning1_17/lecture/10866?isDesc=false)
[Ch6. Quiz](https://www.edwith.org/machinelearning1_17/quiz/10889?isDesc=false)**

**CHAPTER 7. Bayesian Network**
**[1 Probability Concepts](https://www.edwith.org/machinelearning2__17/lecture/10573?isDesc=false)
[7.2 Probability Theorems](https://www.edwith.org/machinelearning2__17/lecture/10844?isDesc=false)
[7.3 Interpretation of Bayesian Network](https://www.edwith.org/machinelearning2__17/lecture/10845?isDesc=false)
[7.4 Bayes Ball Algorithm](https://www.edwith.org/machinelearning2__17/lecture/10846?isDesc=false)
[7.5 Factorization of Bayesian networks](https://www.edwith.org/machinelearning2__17/lecture/10847?isDesc=false)
[7.6 Inference Question on Bayesian network](https://www.edwith.org/machinelearning2__17/lecture/10848?isDesc=false)
[7.7 Variable Elimination](https://www.edwith.org/machinelearning2__17/lecture/10849?isDesc=false)
[7.8 Potential Function and Clique Graph](https://www.edwith.org/machinelearning2__17/lecture/10850?isDesc=false)
[7.9 Simple Example of Belief Propagation](https://www.edwith.org/machinelearning2__17/lecture/10851?isDesc=false)
[Chapter 7. Quiz](https://www.edwith.org/machinelearning2__17/quiz/10852?isDesc=false)**

**CHAPTER 8. K-Means Clustering and Gaussian Mixture Model
[8.1 K-Means Algorithm 1](https://www.edwith.org/machinelearning2__17/lecture/10853?isDesc=false)
[8.2 K-Means Algorithm 2](https://www.edwith.org/machinelearning2__17/lecture/10854?isDesc=false)
[8.3 Multinomial Distribution](https://www.edwith.org/machinelearning2__17/lecture/10855?isDesc=false)
[8.4 Multivariate Gaussian Distribution](https://www.edwith.org/machinelearning2__17/lecture/10857?isDesc=false)
[8.5 Gaussian Mixture Model](https://www.edwith.org/machinelearning2__17/lecture/10858?isDesc=false)
[8.6 EM step for Gaussian Mixture Model](https://www.edwith.org/machinelearning2__17/lecture/10859?isDesc=false)
[8.7 Relation between K-means and GMM](https://www.edwith.org/machinelearning2__17/lecture/10861?isDesc=false)
[8.8 Fundamentals of the EM Algorithm](https://www.edwith.org/machinelearning2__17/lecture/10862?isDesc=false)
[8.9 Derivation of EM Algorithm](https://www.edwith.org/machinelearning2__17/lecture/10864?isDesc=false)
[Chapter 8. Quiz](https://www.edwith.org/machinelearning2__17/quiz/10865?isDesc=false)**

**CHAPTER 9. Hidden Markov Model
[9.1 Concept of Hidden Markov Model](https://www.edwith.org/machinelearning2__17/lecture/10868?isDesc=false)
[9.2 Joint and Marginal Probability of HMM](https://www.edwith.org/machinelearning2__17/lecture/10869?isDesc=false)
[9.3 Forward-Backward probability Calculation](https://www.edwith.org/machinelearning2__17/lecture/10870?isDesc=false)
[9.4 Viterbi Decoding Algorithm](https://www.edwith.org/machinelearning2__17/lecture/10871?isDesc=false)
[9.5 Baum-Welch Algorithm](https://www.edwith.org/machinelearning2__17/lecture/10872?isDesc=false)
[Chapter 9. Quiz](https://www.edwith.org/machinelearning2__17/quiz/10873?isDesc=false)**

**CHAPTER 10. Sampling Based Inference
[10.1 Forward Sampling](https://www.edwith.org/machinelearning2__17/lecture/10874?isDesc=false)
[10.2 Rejection Sampling](https://www.edwith.org/machinelearning2__17/lecture/10875?isDesc=false)
[10.3 Importance Sampling](https://www.edwith.org/machinelearning2__17/lecture/10876?isDesc=false)
[10.4 Markov Chain](https://www.edwith.org/machinelearning2__17/lecture/10877?isDesc=false)
[10.5 Markov Chain for Sampling](https://www.edwith.org/machinelearning2__17/lecture/10878?isDesc=false)
[10.6 Metropolis-Hastings Algorithm](https://www.edwith.org/machinelearning2__17/lecture/10879?isDesc=false)
[10.7 Gibbs Sampling](https://www.edwith.org/machinelearning2__17/lecture/10880?isDesc=false)
[10.8 Understand the LDA(Latent Dirichlet Allocation)](https://www.edwith.org/machinelearning2__17/lecture/10881?isDesc=false)
[10.9 Gibbs Sampling for LDA - 1](https://www.edwith.org/machinelearning2__17/lecture/10882?isDesc=false)
[10.10 Gibbs Sampling for LDA -2](https://www.edwith.org/machinelearning2__17/lecture/10883?isDesc=false)
[Chapter 10. Quiz](https://www.edwith.org/machinelearning2__17/quiz/10884?isDesc=false)**
