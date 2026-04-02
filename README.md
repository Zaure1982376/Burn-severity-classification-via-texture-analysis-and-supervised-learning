# Burn-severity-classification-via-texture-analysis-and-supervised-learning
This repository contains the implementation of an automated framework for burn severity classification based on medical image analysis, texture feature extraction, and supervised machine learning methods. The proposed approach focuses on interpretable feature-based modeling and includes the full pipeline from image preprocessing to classification.

How to Use This Repository
1. Dataset Preparation
Prepare a dataset with three folders, each representing a burn severity class (e.g., `class0`, `class1`, `class2`).  
Each folder should contain approximately 500 images per class.
Images must be color burn images
Regions of Interest (ROIs) must be manually selected and cropped by medical experts
Ensure consistent labeling and image quality

2. Image Preprocessing
Run the script:
    scaling_and_conversion.py

This step performs: 
- Conversion to 8-bit grayscale
- Image resizing
- Normalization
Output: processed grayscale ROI images.

3. Feature Extraction
Run the script:
    data_extraction.py

This will generate:
    DATA_FULL.csv containing extracted texture features.

4. Data Preparation
Run the script:
    data_preprocessing.py

This step produces:
    DATA_TRAIN.csv  
    DATA_TEST.csv

5. Feature Selection and Classification
After preprocessing:
- Apply feature selection (ANOVA, Fisher, Relief and other selection methods)
- Train classification models (KNN, Random Forest, MLP, etc.)

Notes
Ensure file paths are correctly set\ Maintain consistent class labels\ Use a fixed random seed (e.g., 100) for reproducibility

Citation
If you use this work, please cite: Accurate burn severity classification via texture analysis and supervised learning
