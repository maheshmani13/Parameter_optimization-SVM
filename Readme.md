# SVM Classifier Optimization with UCIML Dataset

This repository contains code for optimizing an SVM (Support Vector Machine) classifier using random values and iterations on a dataset sourced from the UCIML repository.

## Introduction
Support Vector Machines (SVMs) are powerful supervised learning models used for classification tasks. They work by finding the optimal hyperplane that best separates classes in a high-dimensional space. Optimization of SVM parameters, such as the kernel type, regularization parameter (C), and kernel coefficient (gamma), is crucial for achieving optimal performance.

## Dataset
The dataset used in this project is sourced from the UCIML repository. It provides information on various features of dry beans, aiming to classify them into different classes based on these features. The dataset consists of features such as area, perimeter, compactness, length, width, asymmetry coefficient, and length of kernel groove. For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.

Dataset Looks like this :
![](https://github.com/maheshmani13/Parameter_optimization-SVM/blob/main/Dry_bean.png)



## Code Overview
The main code file, svm_classifier_optimization.py, contains the implementation of SVM classifier optimization using random values and iterations. Here's a brief overview of the code:

`Data Loading`: The dataset is fetched from the UCIML repository using the fetch_ucirepo function provided by the ucimlrepo library. The dataset is then loaded into pandas DataFrames.

`Data Preprocessing`: No explicit preprocessing steps are shown in the provided code snippet. However, preprocessing steps such as data cleaning, feature scaling, or encoding categorical variables can be applied based on the dataset's requirements.

`SVM Classifier Optimization`:
  -Random values for SVM parameters (kernel type, C, gamma) are generated.
  -SVM classifier is trained using these random parameters.
  -Performance metrics (accuracy) are evaluated.
  -The process is repeated for a specified number of iterations.
  
`Results Visualization`: The best accuracy achieved during optimization is displayed, along with the analysis table showing the best parameters and a convergence plot illustrating the accuracy over iterations.

## Dependencies
`pandas`
`numpy`
`scikit-learn`
`matplotlib`
`ucimlrepo`

## Results
Following Analysis Table and Convergence Graph were generated for the this project.
Convergence Graph is for the sample having overall best accuracy.

![](https://github.com/maheshmani13/Parameter_optimization-SVM/blob/main/Analysis_table.png)

![](https://github.com/maheshmani13/Parameter_optimization-SVM/blob/main/Convergence_Graph.png)

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.


