# Partial Assignment – Machine Learning
**Facial Identity Classification with CelebA**

## Problem Description

This work addresses a **supervised image classification** problem using a subset of the **CelebA (CelebFaces Attributes Dataset)**. The goal is to correctly identify the **identity** associated with a facial image based on automatically extracted visual features.

The CelebA dataset contains more than 200,000 facial images of over 10,000 distinct identities, along with additional annotations such as binary attributes, facial landmarks, and predefined train/test splits. Due to the large volume of data, only a **balanced subset** of identities was used to make the experiments feasible within computational constraints.

Each class in the problem corresponds to a **distinct identity**, and the model must learn to associate a facial image with the correct identity.

In this work, the focus is specifically on the **face identification** task, where each image must be associated with a known identity present in the training set. The face verification task is discussed conceptually but was not experimentally explored in this implementation.

---

## Dataset

- **Dataset:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Total number of images:** 202,599
- **Total number of identities:** 10,177
- **Annotations used:**
  - `identity_CelebA.txt` (image → identity mapping)
  - Auxiliary attribute and landmark files (available but not directly used in this work)

To avoid manual image selection, the `identity_CelebA.txt` file was used to **automatically group images by identity** and construct a subset containing only a limited number of classes, with a controlled number of images per class.

---

## Methodology

The adopted machine learning pipeline follows these steps:

1. **Automatic subset selection**
   - Reading the `identity_CelebA.txt` file
   - Grouping images by identity
   - Selecting a fixed number of identities and images per identity

2. **Preprocessing**
   - Image resizing
   - Conversion to grayscale
   - Data normalization

3. **Feature extraction**
   - Use of the **HOG (Histogram of Oriented Gradients)** descriptor to transform each image into a numerical feature vector

4. **Data splitting**
   - Separation into training and testing sets
   - Stratified split to preserve class proportions

5. **Classification**
   - Use of a **linear SVM classifier (LinearSVC)** trained on HOG features

6. **Evaluation**
   - Performance evaluation using **accuracy** on the test set

---

## Implemented Models

The project uses classical supervised models, including:

- **Linear Models (LinearSVC / SVM C-SVC)**
- Full pipeline:
  - Descriptor extraction
  - Normalization (StandardScaler)
  - Training
  - Evaluation

Parameter selection is performed in a controlled manner, following best practices for balancing and cross-validation.

---

## Training and Evaluation Strategy

- Stratified train/test split
- The code structure allows extension to **k-fold cross-validation (k=5)**, as suggested in the assignment
- Automatic saving of:
  - Configuration (`config.txt`)
  - Error evolution (`error.txt`)
  - Trained model (`model.dat`)

Multiple experimental scenarios are generated, including:
- Best-performing model
- Worst-performing model
- Different feature descriptors

---

## Reproducibility

All runs:

- Use a **fixed random seed**
- Save the full experiment configuration
- Allow reapplication of the trained model on new images

---

## Dependencies

Main libraries used:

- Python ≥ 3.10
- NumPy
- scikit-learn
- scikit-image
- OpenCV (optional)

---

## How to Run

1. Create a virtual environment and install dependencies
2. Run the subset creation script:

```bash
python make_subset.py
```
