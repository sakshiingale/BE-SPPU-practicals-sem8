# -*- coding: utf-8 -*-
"""
Face Emotion Recognition Model Training
Updated to use directory structure with train/test folders
"""

import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train face emotion recognition model')
    parser.add_argument('--train_dir', type=str, default='train', 
                        help='Path to the training data directory')
    parser.add_argument('--test_dir', type=str, default='test', 
                        help='Path to the test data directory')
    parser.add_argument('--epochs', type=int, default=15, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
                        
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Output directory for saving the model')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Initialize Parameters
    num_classes = 7  # 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
    epochs = args.epochs
    batch_size = args.batch_size
    num_features = 64
    width, height = 48, 48
    
    print(f"Training data directory: {args.train_dir}")
    print(f"Test data directory: {args.test_dir}")
    
    # Check if directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory {args.train_dir} not found")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory {args.test_dir} not found")
        return
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load images from directories
    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(width, height),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(width, height),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(width, height),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size
    )
    
    # Print class mapping
    print("Class mapping:")
    for class_name, class_index in train_generator.class_indices.items():
        print(f"{class_index}: {class_name}")
    
    # Build the model
    print("Building the model...")
    model = Sequential()
    
    # Module 1: conv<<conv<<batchnorm<<relu<<maxpooling<<dropout
    model.add(Conv2D(num_features, kernel_size=(3,3), padding='same', input_shape=(width, height, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(num_features, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    # Module 2: conv<<conv<<batchnorm<<relu<<maxpool<<dropout
    model.add(Conv2D(2*num_features, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(2*num_features, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    # Module 3: conv<<conv<<batchnorm<<relu<<maxpool<<dropout
    model.add(Conv2D(4*num_features, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(4*num_features, kernel_size=(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    # Module 4: fc<<batchnorm<<fc<<batchnorm<<dropout<<softmax
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.output_dir, 'emotion_model_checkpoint.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Calculate steps per epoch and validation steps
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    # Training
    print(f"Training the model for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        validation_data=validation_generator,
        validation_steps=validation_steps
    )
    
    # Evaluate the model
    test_steps = test_generator.samples // batch_size
    loss_and_metrics = model.evaluate(test_generator, steps=test_steps, verbose=1)
    print(f"Test loss: {loss_and_metrics[0]:.4f}")
    print(f"Test accuracy: {loss_and_metrics[1]:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    print(f"Training history plot saved to {os.path.join(args.output_dir, 'training_history.png')}")
    
    # Create a dictionary to map indices back to emotion names
    # This ensures the app uses the correct emotion labels
    emotion_dict = {}
    for emotion_name, index in train_generator.class_indices.items():
        emotion_dict[index] = emotion_name
    
    # Save the emotion mapping
    with open(os.path.join(args.output_dir, "emotion_map.txt"), "w") as f:
        for index, emotion in emotion_dict.items():
            f.write(f"{index}: {emotion}\n")
    
    # Save model and weights
    model_json = model.to_json()
    with open(os.path.join(args.output_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output_dir, "model.h5"))
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()