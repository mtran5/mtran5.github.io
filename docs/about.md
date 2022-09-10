---
layout: page
title: About
permalink: /about/
---

Hello!

My name is Minh Tran (pronounced *Ming*, last name rhymes with *Jordan*). I'm  a second-year PhD candidate in bioengineering.

In this blog I want to write about the things that interest me 
the most: healthcare, machine learning, self-improvement, engineering, and many more.

I hope that each of you find something here that spark your interest.

Cheers,

Minh

* [LinkedIn](https://www.linkedin.com/in/minhtran97/)
* [Toward Data Science](https://medium.com/@minh.tran2_11384)
* [Resume](/assets/about/resume.pdf)  

# My Projects

## Thyroid Cancer Identification with Deep Learning and Multispectral Imaging

With multispectral imaging, we can capture images that have not 
just three channels, but multiple channels ranging from visible to near infrared. Previously, our research lab (Quantitative Imaging Laboratory) has found that multispectral
imaging can potentially detect cancer. My research involved gathering multispectral data of thyroid cancer tissues, and training different neural network to improve upon
current technologies.

I started with using a modified VGG-19 network and achieved better ROC-AUC than that of regular RGB images. Then, I explored using vision transformer and found the 3-dimensional
TimesFormer. I adapted the system to training spectral data with great success and found improvement. I also introduced new data augmentation techniques to imnprove further the quality
of classification. I found that data augmentation can sometimes improve the test accuracy by up to 4%.

![Tissue](/assets/about/tissue.png)

**Technology used**: 
Python (PyTorch, pandas, OpenCV, PIL, sklearn, matplotlib), AWS, wandb, C++ (OpenCV)  

**Publications**:
Tran, Minh Ha, et al. "Thyroid carcinoma detection on whole histologic slides using hyperspectral imaging and deep learning." Medical Imaging 2022: Digital and Computational Pathology. Vol. 12039. SPIE, 2022.

## Sleep Stages Classification with Clustering

We have roughly divided sleep into multiple stages, from light sleep (N1) to deep sleep (N3) and dreaming (REM). This division is made possible by measuring the electrical signals
of the eye and the brain. In this project, I downloaded and analyzed the ISRUC-SLEEP Dataset, which consists of 100 recordings totalling 500 hours. 

I used wavelet transform to extract relevant features. Then, I discovered that stages of sleep can be clustered without labelling. I decided to explore and found that overall, 
sleep stages can be classified with up to 84% accuracy. 

**Technology used**:
MATLAB, Python (SciPy, OpenCV, pandas, matplotlib)

![Brain](/assets/about/brain.png)

## Posture Detection with Inertial Measurement Unit

Trunk exoskeletons can potentially reduce spinal injury and stress. However, they have not been properly tested in longer-term real-world studies. 
Our reserach lab at the University of Wyoming performed two preliminary studies on using inertial
measurement units (IMUs) to collect kinematic data from an exoskeleton wearer. 
Participants performed multiple activities (standing, sitting, walking, lifting with back, lifting with leg) wearing both the exoskeleton and the IMU sensors.

I used MATLAB to analyze 20 hours of sensor data taken during various activities. Stepwise forward selection was used to select the most relevant features out of 120 total features.
Naïve Bayes classifier was used to classify five activities (squatting, slouching, sitting, standing, and walking) with 92.2% accuracy.

![Lift](/assets/about/lift.PNG)

**Technology used**:
MATLAB, R

**Publications**:
Tran, Minh Ha, et al. "Toward real-world evaluations of trunk exoskeletons using inertial measurement units." 2019 IEEE 16th International Conference on Rehabilitation Robotics (ICORR). IEEE, 2019.
