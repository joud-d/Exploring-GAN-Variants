import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import os
from collections import Counter

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class AnimalCNNClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        """
        Initialize CNN classifier for animal images
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes (dog, spider, elephant)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = ['dog', 'spider', 'elephant']
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build CNN architecture optimized for animal classification
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling (alternative to Flatten)
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.25):
        """
        Prepare and split data for training with proper stratification for imbalanced data
        
        Args:
            X: Input images
            y: Labels
            test_size: Proportion for test set (20% of total data)
            val_size: Proportion for validation set (20% of total data, 25% of remaining after test split)
        
        Returns:
            Split datasets: X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Preparing data splits...")
        print(f"Original dataset size: {len(X)} samples")
        
        # Check class distribution before splitting
        unique_classes, counts = np.unique(y, return_counts=True)
        print("Original class distribution:")
        for i, class_name in enumerate(self.class_names):
            if i in unique_classes:
                count = counts[list(unique_classes).index(i)]
                percentage = (count / len(y)) * 100
                print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # First split: separate test set (20% of total data)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        # val_size=0.25 means 25% of remaining 80% = 20% of original data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=42, 
            stratify=y_temp
        )
        
        print(f"\nData split completed:")
        print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Normalize pixel values to [0, 1]
        print("Normalizing pixel values...")
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Verify splits maintain class distribution
        print("\nVerifying stratification worked correctly:")
        for split_name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"{split_name} set class distribution:")
            for i, class_name in enumerate(self.class_names):
                if i in unique:
                    count = counts[list(unique).index(i)]
                    percentage = (count / len(labels)) * 100
                    print(f"    {class_name}: {count} samples ({percentage:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def analyze_class_distribution(self, y_train, y_val, y_test):
        """
        Analyze and visualize class distribution in the dataset
        """
        train_counts = Counter(y_train)
        val_counts = Counter(y_val)
        test_counts = Counter(y_test)
        
        print("Class Distribution Analysis:")
        print("=" * 50)
        print(f"Training set:")
        for i, class_name in enumerate(self.class_names):
            count = train_counts.get(i, 0)
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nValidation set:")
        for i, class_name in enumerate(self.class_names):
            count = val_counts.get(i, 0)
            percentage = (count / len(y_val)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nTest set:")
        for i, class_name in enumerate(self.class_names):
            count = test_counts.get(i, 0)
            percentage = (count / len(y_test)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Plot distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        datasets = [('Train', y_train), ('Validation', y_val), ('Test', y_test)]
        
        for idx, (name, labels) in enumerate(datasets):
            counts = [Counter(labels).get(i, 0) for i in range(self.num_classes)]
            axes[idx].bar(self.class_names, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[idx].set_title(f'{name} Set Distribution')
            axes[idx].set_ylabel('Number of Samples')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the CNN model
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """
        Plot training and validation metrics
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation with all required metrics
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Basic metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print("Model Evaluation Results:")
        print("=" * 50)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        # ROC Curves and AUC
        self.plot_roc_curves(y_test, y_pred_proba)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Calculate and display per-class metrics
        print("\nPer-class Metrics from Confusion Matrix:")
        print("-" * 40)
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
    
    def plot_roc_curves(self, y_test, y_pred_proba):
        """
        Plot ROC curves for each class and calculate AUC
        """
        # Binarize the output
        y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
        
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'green', 'blue']
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def predict_sample_images(self, X_test, y_test, num_samples=9):
        """
        Display sample predictions with confidence scores
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Randomly select samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        sample_images = X_test[indices]
        sample_labels = y_test[indices]
        
        # Get predictions
        predictions = self.model.predict(sample_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Plot results
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(sample_images[i])
            
            true_label = self.class_names[sample_labels[i]]
            pred_label = self.class_names[predicted_classes[i]]
            confidence = predictions[i][predicted_classes[i]] * 100
            
            color = 'green' if predicted_classes[i] == sample_labels[i] else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                            color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Data loading function for Google Colab
def load_animal_data(data_path='/content/drive/MyDrive/animals'):
    """
    Load animal images from Google Drive directory structure
    Expected structure:
    /content/drive/MyDrive/animals/
    ├── dog/
    ├── spider/
    └── elephant/
    """
    from tensorflow.keras.preprocessing import image
    import glob
    
    print(f"Loading data from: {data_path}")
    
    # Mount Google Drive if not already mounted
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
    except:
        print("Google Drive already mounted or not in Colab environment")
    
    # Check if path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist!")
    
    # Define class mapping
    class_names = ['dog', 'spider', 'elephant']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    images = []
    labels = []
    class_counts = {}
    
    print("Loading images...")
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist! Skipping {class_name}")
            continue
        
        # Get all image files (jpg, jpeg, png)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        print(f"Found {len(image_files)} {class_name} images")
        class_counts[class_name] = len(image_files)
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                
                images.append(img_array)
                labels.append(class_to_idx[class_name])
                
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total images: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Class distribution:")
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        percentage = (count / len(y)) * 100 if len(y) > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    return X, y, class_counts

def verify_data_structure(data_path='/content/drive/MyDrive/animals'):
    """
    Verify the data directory structure and show file counts
    """
    print("Verifying data structure...")
    print("=" * 50)
    
    if not os.path.exists(data_path):
        print(f"❌ Main directory {data_path} does not exist!")
        return False
    
    print(f"✅ Main directory exists: {data_path}")
    
    class_names = ['dog', 'spider', 'elephant']
    all_good = True
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            # Count image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_count = 0
            for ext in image_extensions:
                image_count += len(glob.glob(os.path.join(class_path, ext)))
            
            print(f"✅ {class_name}/ directory: {image_count} images")
        else:
            print(f"❌ {class_name}/ directory missing!")
            all_good = False
    
    return all_good

# Main execution
if __name__ == "__main__":
    # Data path configuration
    DATA_PATH = '/content/drive/MyDrive/animals'
    
    print("Starting Animal Classification Training on Original Imbalanced Dataset")
    print("=" * 70)
    
    # Step 1: Verify data structure
    if not verify_data_structure(DATA_PATH):
        print("❌ Data structure verification failed!")
        print("Please ensure your data is organized as:")
        print("/content/drive/MyDrive/animals/")
        print("├── dog/")
        print("├── spider/")
        print("└── elephant/")
        exit(1)
    
    # Step 2: Load data
    try:
        X, y, class_counts = load_animal_data(DATA_PATH)
        print(f"✅ Data loaded successfully: {len(X)} total images")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit(1)
    
    # Step 3: Initialize classifier
    classifier = AnimalCNNClassifier()
    
    # Step 4: Display model architecture
    print("\nCNN Model Architecture:")
    print("=" * 50)
    classifier.get_model_summary()
    
    # Step 5: Prepare data with proper splitting
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(X, y)
    
    # Step 6: Analyze class distribution
    classifier.analyze_class_distribution(y_train, y_val, y_test)
    
    # Step 7: Train model
    print("\nStarting model training...")
    history = classifier.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Step 8: Plot training history
    print("\nDisplaying training history...")
    classifier.plot_training_history()
    
    # Step 9: Evaluate model
    print("\nEvaluating model performance...")
    results = classifier.evaluate_model(X_test, y_test)
    
    # Step 10: Show sample predictions
    print("\nDisplaying sample predictions...")
    classifier.predict_sample_images(X_test, y_test)
    
    # Step 11: Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED - ORIGINAL IMBALANCED DATASET RESULTS")
    print("=" * 70)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test Loss: {results['loss']:.4f}")
    print("\nThese results will serve as baseline for comparison with GAN-augmented datasets!")
    
    # Save results for later comparison
    import pickle
    baseline_results = {
        'model_type': 'CNN_Original_Imbalanced',
        'test_accuracy': results['accuracy'],
        'test_loss': results['loss'],
        'class_distribution': class_counts,
        'confusion_matrix': results['confusion_matrix'],
        'training_history': history.history if history else None
    }
    
    try:
        with open('/content/baseline_results.pkl', 'wb') as f:
            pickle.dump(baseline_results, f)
        print("✅ Baseline results saved to /content/baseline_results.pkl")
    except Exception as e:
        print(f"⚠️ Could not save results: {str(e)}")
    
    print("\nNext steps:")
    print("1. Note down these baseline metrics")
    print("2. Train your Vanilla GAN and GAN variant")
    print("3. Use generated data to balance the dataset")
    print("4. Retrain this CNN on balanced datasets")
    print("5. Compare performance improvements!")
