import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# # Create directories if they don't exist
# os.makedirs('static', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Constants
IMG_SIZE = (160, 160)  # MobileNetV2 optimal input size
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7  # angry, disgust, fear, happy, sad, surprise, neutral

# Paths
train_dir = '../data/train'
test_dir = '../data/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 127.5,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1. / 127.5)

# Data Generators
print("Setting up data generators...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)


# MobileNetV2 Model
def build_mobilenet_model():
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # Smaller version (35% of original)
    )

    # Freeze base model layers
    base_model.trainable = False

    # Build new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


print("Building MobileNetV2 model...")
model = build_mobilenet_model()
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(
        'static/best_mobilenet_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Training
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc:.4f}")


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plot_path = 'static/mobilenet_training_history.png'
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()


plot_training_history(history)

# Save final model
final_model_path = 'static/final_mobilenet_model.h5'
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")


# Optional: Fine-tuning
def fine_tune_model(model):
    # Unfreeze some layers
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:100]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Uncomment to enable fine-tuning
print("\nFine-tuning model...")
model = fine_tune_model(model)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)