# Exploring-GAN-Variants
📌 Project Overview
This project explores the application of Generative Adversarial Networks (GANs) to address class imbalance by generating synthetic data for underrepresented classes. The study compares three GAN models: Vanilla GAN, DCGAN, and Conditional GAN (CGAN) to assess their effectiveness in balancing the dataset and improving classification performance.

📂 Repository Structure
graphql
Copy
Edit
├── Dataset_Imbalance_Visualizations.ipynb     # Visualizes and analyzes dataset imbalance
├── Vanilla GAN Implementation.ipynb           # Implementation and training of Vanilla GAN
├── VanillaGAN Classifier.ipynb                # Classifier trained with Vanilla GAN data
├── DCGAN imp.ipynb                            # DCGAN implementation and training
├── DCGAN Classifier.ipynb                     # Classifier trained with DCGAN data
├── CGAN Implementaition notebook.ipynb        # CGAN implementation and training
├── CGAN Classifier.ipynb                      # Classifier trained with CGAN data
├── cnn_imbalanced_classifier.py               # Baseline classifier on original imbalanced data
├── FINAL VANILLA CLASSIFIER.ipynb             # Unified classifier used for all GAN variants
├── README.md                                  # Project description and usage instructions
🧠 Objectives
Tackle the class imbalance problem using synthetic data generation with GANs.

Implement and compare three GAN architectures for data augmentation.

Evaluate classification performance with and without synthetic data augmentation using consistent metrics.

🧪 GAN Models Implemented
Model	Description
Vanilla GAN	Basic GAN with fully connected layers.
DCGAN	Deep Convolutional GAN using convolutional networks for image generation.
CGAN	Conditional GAN, which generates class-specific samples using label inputs.

🖼️ Dataset & Preprocessing
The dataset used contains a significant class imbalance in image classes.

Visual analysis of imbalance is provided in Dataset_Imbalance_Visualizations.ipynb.

🧮 Classifier Architecture
A consistent Convolutional Neural Network (CNN) architecture was used across all experiments.
The classifier was implemented using TensorFlow/Keras and is detailed in:

FINAL VANILLA CLASSIFIER.ipynb — the standard classifier used for all GAN-generated datasets.

cnn_imbalanced_classifier.py — classifier trained on the original imbalanced dataset only.

📊 Evaluation Metrics
Each model was evaluated using the following classification metrics:

Accuracy

Precision

Recall

F1-Score

AUC-ROC

Confusion Matrix

