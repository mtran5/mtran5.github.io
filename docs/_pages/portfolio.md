---
title: "Portfolio"
permalink: /portfolio/
date: 2016-02-24T03:02:20+00:00
---

# Computer Vision 👁️‍🗨️

## Thyroid Cancer Identification with Deep Learning

In my research, I used deep learning to improve identification of cancer.
I started with using a modified VGG-19 network. 
Later on, I used 3-dimensional vision transformer for this task.
I adapted the system to train spectral images with great success and found improvement
over regular RGB images.

![Tissue](/assets/about/research1.png)


***Personal Contributions*** 👨🏻‍💻
* Developed software in C++ to control imaging microscope and capture 100K images of thyroid
* Used OpenCV to identify and remove 25% of images that are whitespace or out-of-focus
* Trained a vision transformer in PyTorch to detect thyroid cancer with 0.906 AUC
* Used Amazon SageMaker and wandb to monitor training and select hyperparameters
* Presented research result orally at the 2022 SPIE Medical Imaging Conference  


***Technology Used*** ⚙️
Python (PyTorch, pandas, OpenCV, PIL, sklearn, matplotlib), AWS, wandb, C++ (OpenCV)  

[Publication Link 📝](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12039/120390H/Thyroid-carcinoma-detection-on-whole-histologic-slides-using-hyperspectral-imaging/10.1117/12.2612963.short?SSO=1){: .btn .btn--success .btn--medium}

## U-Net for Image Segmentation

Implementation of U-Net for various image segmentation and image reconstruction tasks.
To validate my network, I generated synthesized segmentation data using OpenCV.

![UNet](/assets/about/unet.png)

![Output](/assets/about/output.png)

***Technology Used*** ⚙️
Python (PyTorch, OpenCV, PIL, sklearn, matplotlib), 

***GitHub Link*** 💻 [![View on Github](https://img.shields.io/badge/View on GitHub-mtran5.github.io-blue?logo=GitHub)](https://github.com/mtran5/UNet)

## Stanford CS231n: Convolutional Neural Networks for Visual Recognition

Solutions for Stanford CS231n coursework in Spring 2020.  

In this course, I used Python, NumPy, TensorFlow, and Keras to implement the following:
* Multi-layered networks
* Regularization techniques (batchnorm, dropout)
* Convolutional neural networks (CNNs)
* Recurrent neural networks (RNNs)
* Long-short term memory networks (LSTMs)
* Style Transfer

***GitHub Link*** 💻
[![View on Github](https://img.shields.io/badge/View on GitHub-mtran5.github.io-blue?logo=GitHub)](https://github.com/mtran5/CS231n)


# Machine Learning 👩‍🏫

## Sleep Quality Analysis

Sleep scoring is used to judge the quality of sleep and identify potential problems 
(sleep apnea, for example). In this study performed at Northwestern University, 
I used clustering to automatically classify 30-second epochs of sleep into 
different stages.

![Brain](/assets/about/brain.png)

![Stages](/assets/about/stages.jpg)

***Personal Contributions*** 👨🏻‍💻

* Downloaded and analyzed the ISRUC-SLEEP Dataset using MATLAB
* Analyzed 100 recordings totalling 500 hours. 
* Used wavelet transform to extract relevant features.
* Used hierarchical clustering to cluster sleep stages with 84% accuracy

***Technology Used*** ⚙️
MATLAB, Python (SciPy, OpenCV, pandas, matplotlibt)

## Posture Detection

At the University of Wyoming, I performed two studies on using inertial
measurement units (IMUs) to collect kinematic data from an exoskeleton wearer. 

***Personal Contributions*** 👨🏻‍💻
* Planned and conducted scientific studies on 17 volunteers wearing exoskeletons and inertial measurement units (IMU) sensors.
* Analyzed 20 hours of sensor data taken during various activities
* Reduced input feature space by a factor of 5 using stepwise forward selection method
* Used Naïve Bayes classifier to classify five activities (squatting, slouching, sitting, standing, and walking) with 92.2% accuracy.
* Presented research result at the IEEE Conference on Rehabilitation Robotics

***Technology Used*** ⚙️
MATLAB, R 

[Publication Link 📝](https://pubmed.ncbi.nlm.nih.gov/31374676/){: .btn .btn--success .btn--medium}


## [In Progress] Machine Learning From Scratch
Implementation from scratch using NumPy and PyTorch popular machine learning and deep learning algorithms:
* Machine learning algorithms
	* Linear regression
	* Decision tree
	* K-nearest neighbors
* Deep learning architechtures
	* ResNet
	* U-Net
	* MobileNet
	* SqueezeNet
	* Vision transformers

***GitHub Link*** 💻
[![View on Github](https://img.shields.io/badge/View on GitHub-mtran5.github.io-blue?logo=GitHub)](https://github.com/mtran5/PyTorchNeuralNets)

# Natural Language Processing 🗪
## Question-Answering Network

![QA](/assets/about/QA.png)

I used BERT, a language model pretrained on hundreds of millions of natural language text,
for the task of implementing a question-answering network. The network is finetuned 
on BioASQ, a dataset of biomedical questions. 

***Technology Used*** ⚙️
Python (HuggingFace, Gensim, SpaCy, sklearn), AWS EC2.

***GitHub Link*** 💻
[![View on Github](https://img.shields.io/badge/View on GitHub-mtran5.github.io-blue?logo=GitHub)](https://github.com/mtran5/PubMedQA)

## [In Progress] CS224n: Natural Language Processing with Deep Learning

Solutions to answers for CS224n. In this course I used PyTorch to implement:
* Word embedding
* Attention models (transformers)
* Question-answering networks
* Language model pipelines (sentiment analysis)

# Publications 📝

## Journal/Conference Papers
* Tran, Minh Ha, et al. "Thyroid carcinoma detection on whole histologic slides using hyperspectral imaging and deep learning." Medical Imaging 2022: Digital and Computational Pathology. Vol. 12039. SPIE, 2022.
* Ma, Ling, et al. "Unsupervised super-resolution reconstruction of hyperspectral histology images for whole-slide imaging." Journal of Biomedical Optics 27.5 (2022): 056502.
* Tran, Minh Ha, et al. "Toward real-world evaluations of trunk exoskeletons using inertial measurement units." 2019 IEEE 16th International Conference on Rehabilitation Robotics (ICORR). IEEE, 2019.
* Goršič, Maja, Minh Ha Tran, and Domen Novak. "A novel virtual environment for upper limb rehabilitation." 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2018.
 
## Blog/Technical Writings
I wrote almost exclusively for Towards Data Science, a technical blog aimed at data science community.
* [Understanding U-Net](https://towardsdatascience.com/understanding-u-net-61276b10f360)
* [Not Just PyTorch and TensorFlow: 4 Other Deep Learning Libraries You Should Know](https://medium.com/towards-data-science/not-just-pytorch-and-tensorflow-4-other-deep-learning-libraries-you-should-lnow-a72cf8be0814)
* [What did a chat bot made in 1966 tell us about human language?](https://medium.com/towards-data-science/what-did-a-chat-bot-made-in-1966-tell-us-about-human-language-886613a16a7f)
* [Three Things to Know about Binary Search](https://medium.com/towards-data-science/three-things-to-know-about-binary-search-cf3b00971c2c)
* [A Visual Understanding of Bias and Variance](https://medium.com/towards-data-science/a-visual-understanding-of-bias-and-variance-66179f16be32)