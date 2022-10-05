# Cross-X-AI
## Cross X-AI: Explainable semantic segmentation of laparoscopic images in relation with depth estimations.

Francesco Bardozzo, Mattia Delli Priscoli, Toby Collins, Antonello Forgione, Alexandre Hostettler, Roberto Tagliaferri

submitted to WCCI 2022



This is the reference Python repository 
for training and testing depth estimation and segmentation models
for Cross Explainable AI.

```
In this work, two deep learning models, trained to segment the liver and perform
its depth reconstruction, are compared and analysed by means of their post-hoc explanation
interplay. The first model (a \emph{U-Net}) is designed to perform liver semantic
segmentation over different subjects and scenarios. In detail, the image pixels representing
the liver are classified and separated by the surrounding pixels. Meanwhile, with the second
model, a depth estimation is performed to classify the z-position pixel intensities 
(relative depths). In general, these two models apply a sort of classification task which
can be explained for each model individually and that can be combined to show additional
relations and insights between the most relevant features observed during the model learning
process.  In detail, this work will show how post-hoc explainable AI systems (X-AI) based 
on Grad CAM and Grad CAM++ can be compared by introducing Cross X-AI (CX-AI). Typically 
the post-hoc explanation maps provide different visual explanations on the way they take
their decisions based on the two proposed approaches. Despite that, our results show that
the Grad Cam++ segmentation explanation maps present cross-learning/classification 
strategies similar to disparity explanations (and vice versa).
```



![Test Image 6](https://github.com/lodeguns/Cross-X-AI/blob/main/pipexai.png)




## Small sample dataset
10 images are represented into the numpy files: X_test.npy and Y_test.npy for both X-Depth explanations and
X-Seg explanations. To access the whole repository please contact: Francesco Bardozzo - fbardozzo at unisa dot com

## Pre-trained models are on:
Supervised depth estimation : depth_ep18.hdf5
Supevised U-Net liver segmentation : unet_weights folder

## How to run our evaluations?

```
python3 evaluate_depth.py
python3 evaluate_seg.py

#to test training U-Net metrics (segmentation task)
python3 read_unet_metrics.py

# to test training Siamese metrics (depth estimation task)
tensorboard --logdir metrics_tensorboard_siamese/validation/     
```

## While to run our models for Cross-X-AI

```
python3 cross_xai.py
```

in the folder X_test is provided a sample file in output from cross_xai.py.
 
 

## Citation

```
- Bardozzo, Francesco, et al. "Cross X-AI: Explainable Semantic Segmentation of 
  Laparoscopic Images in Relation to Depth Estimation." 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022.

- Bardozzo, Francesco, et al. "StaSiS-Net: A stacked and siamese disparity estimation network for 
  depth reconstruction in modern 3D laparoscopy." Medical Image Analysis 77 (2022): 102380.
```

