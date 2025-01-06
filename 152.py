import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

train_dir = pathlib.Path("C:/MPOXCLASS/DATA_AFTER_SPLIT/aug_train")
val_dir = pathlib.Path("C:/MPOXCLASS/DATA_AFTER_SPLIT/Valid")
test_dir = pathlib.Path("C:/MPOXCLASS/DATA_AFTER_SPLIT/Test")

# Load disease mappings from CSV files
def load_disease_mapping(file_path):
    mapping_df = pd.read_csv(file_path)
    return dict(zip(mapping_df['Filename'], mapping_df['Disease']))

# Load mappings for train, val, and test datasets
train_mapping = load_disease_mapping("C:/MPOXCLASS/train_disease_mapping.csv")
val_mapping = load_disease_mapping("C:/MPOXCLASS/val_mapping.csv")
test_mapping = load_disease_mapping("C:/MPOXCLASS/test_mapping.csv")

# Set parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 20
initial_lr = 1e-4  # Set the initial learning rate

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)
val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)
test_ds = image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Extract class names
class_names = train_ds.class_names
print("Class names:", class_names)

# Display sample images
def display_sample_images(dataset, num_samples=5):
    plt.figure(figsize=(15, 5))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

display_sample_images(train_ds)

# Build and compile the ResNet152 model
def build_resnet152():
    base_model = keras.applications.ResNet152(
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        include_top=False
    )
    base_model.trainable = True
    for layer in base_model.layers[:-100]:  # Unfreeze only top layers
        layer.trainable = False

    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs, x)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Initialize the model
model_resnet = build_resnet152()

# Learning rate scheduler and early stopping callbacks
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history_resnet = model_resnet.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the trained model
model_resnet.save("resnet152.keras")

# Evaluate on test set
test_loss, test_accuracy = model_resnet.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions and evaluate
test_images, test_labels, image_paths = [], [], []

for images, labels in test_ds:
    # Convert tensors to numpy arrays
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

    # Collect image paths from the dataset
    for i in range(len(images)):
        image_path = test_ds.file_paths[len(image_paths)]  # Adjusted to get the correct image path
        image_paths.append(image_path)

test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# Make predictions
resnet_predictions = model_resnet.predict(test_images)
predicted_labels = np.argmax(resnet_predictions, axis=1)
prediction_probabilities = np.max(resnet_predictions, axis=1)

# Create a DataFrame to hold the results
results_df = pd.DataFrame({
    "Image Path": [f"{pathlib.Path(path).name}" for path in image_paths],  # Only get the filename
    "True Label": [test_mapping[pathlib.Path(path).name] for path in image_paths],  # Get true label from mapping
    "Predicted Label": [class_names[pred] for pred in predicted_labels],
    "Prediction Probability": prediction_probabilities
})

# Save results to a CSV file
results_df.to_csv("resnet152_result100.csv", index=False)
print("Results saved to resnet152_result.csv")

# Print the results DataFrame
print(results_df)

# Classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_labels, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix Heatmap")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
