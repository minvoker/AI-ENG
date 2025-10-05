"""Portfolio 4

This script trains two different CNN models to classify images of metallic surfaces as either *rust* or *no rust*.

When run, the script will create the folders `cnn_test` and `resnet50_test` in the
project root (if they do not already exist) and populate them with the
prediction images.
"""

import os
import random
import shutil
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, GlobalAveragePooling2D)
from tensorflow.keras.applications import ResNet50


# Paths ----------------------------------------------------------------------
# Adjust this path to point at the folder containing the corrosion data.  The
# expected structure is described in the accompanying README.md.
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, 'Corrosion-dataset', 'Corrosion')
DATA_ROOT = os.path.normpath(DATA_ROOT)

# Output directories for predicted images
CNN_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, 'cnn_test')
RESNET_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, 'resnet50_test')


def ensure_test_split(base_dir: str, test_count: int = 10) -> None:
    """Ensure there is a test set with a fixed number of samples per class.

    If the dataset already contains a `test` subfolder, nothing is done.  If
    not, this function creates it by randomly selecting `test_count` images
    from each class in the `train` folder.  Images are copied (not moved)
    into the new test folders so that the original training data remain
    intact.

    Parameters
    ----------
    base_dir: str
        Path to the `Corrosion` directory containing `train` and optionally
        `test` subdirectories.
    test_count: int, optional
        Number of images per class to include in the test split.  Default is 10.
    """
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    class_names = ['rust', 'no rust']

    # If the test directory already exists and appears populated, do nothing
    if os.path.isdir(test_dir):
        has_data = all(os.path.isdir(os.path.join(test_dir, cls)) and
                        len(os.listdir(os.path.join(test_dir, cls))) >= test_count
                        for cls in class_names)
        if has_data:
            return

    # Otherwise create the test folder structure
    for cls in class_names:
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Copy a fixed number of images per class into the test folder
    for cls in class_names:
        class_train_dir = os.path.join(train_dir, cls)
        class_test_dir = os.path.join(test_dir, cls)
        # Gather all image file names in the training directory
        images = [f for f in os.listdir(class_train_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        selected = images[:test_count]
        for fname in selected:
            src = os.path.join(class_train_dir, fname)
            dst = os.path.join(class_test_dir, fname)
            if not os.path.exists(dst):
                shutil.copy(src, dst)


def build_basic_classifier(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """Construct a simple CNN model.

    The architecture loosely follows the pattern of the MNIST classifier but
    increases the number of filters to handle colour images.  The final
    Dense layer uses a softmax activation so that predictions form a
    probability distribution over the classes.

    Parameters
    ----------
    input_shape: tuple
        The shape of each input image (height, width, channels).
    num_classes: int
        Number of target classes.

    Returns
    -------
    tensorflow.keras.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_resnet_classifier(input_shape: Tuple[int, int, int], num_classes: int) -> Model:
    """Create a ResNet50‑based model for two‑class classification.

    A pre‑trained ResNet50 network is used as a feature extractor.  Its
    convolutional layers are frozen so that only the newly added classification
    head is trained.  Global average pooling is applied before the dense
    layer to reduce the spatial dimensions.

    Parameters
    ----------
    input_shape: tuple
        Dimensions of the input images.
    num_classes: int
        Number of target classes.

    Returns
    -------
    tensorflow.keras.Model
        Compiled model ready for training.
    """
    base_model = ResNet50(weights='imagenet', include_top=False,
                           input_shape=input_shape)
    base_model.trainable = False  # freeze the convolutional base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_generators(base_dir: str,
                       image_size: Tuple[int, int] = (64, 64),
                       batch_size: int = 16) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """Create training, validation and test data generators.

    Keras' `ImageDataGenerator` is used to load images from disk and
    optionally apply basic normalisation.  A validation split of 20 % is
    applied to the training data; the test data use an independent
    generator without data augmentation.

    Parameters
    ----------
    base_dir: str
        Path to the `Corrosion` folder containing `train` and `test` subfolders.
    image_size: tuple, optional
        Size to which all images will be resized.
    batch_size: int, optional
        Number of images per batch.

    Returns
    -------
    tuple
        A triple `(train_gen, val_gen, test_gen)` of Keras generators.
    """
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen


def train_and_evaluate(model: tf.keras.Model,
                        train_gen: ImageDataGenerator,
                        val_gen: ImageDataGenerator,
                        test_gen: ImageDataGenerator,
                        epochs: int,
                        output_dir: str) -> None:
    """Train a model and evaluate it on a test generator.

    After fitting the model, this function computes a confusion matrix and a
    classification report, prints them to the console and saves the test
    images with predicted labels appended to the filename.

    Parameters
    ----------
    model: tensorflow.keras.Model
        The model to train.
    train_gen: ImageDataGenerator
        The generator yielding training batches.
    val_gen: ImageDataGenerator
        The generator yielding validation batches.
    test_gen: ImageDataGenerator
        The generator yielding single images for evaluation.
    epochs: int
        Number of training epochs.
    output_dir: str
        Directory where predicted images will be saved.
    """
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1
    )

    # Evaluate on the test data
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=0)
    y_true = test_gen.classes
    y_pred = np.argmax(predictions, axis=1)

    # Print metrics
    labels = list(test_gen.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall accuracy: {accuracy:.2%}")

    # Save predicted images to disk
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(len(test_gen.filenames)):
        img_array, _ = test_gen[idx]
        fname = test_gen.filenames[idx]
        predicted_label = labels[y_pred[idx]]
        # Remove any directory components from the original filename
        base_name = os.path.basename(fname)
        root, ext = os.path.splitext(base_name)
        new_name = f"{root}_{predicted_label}{ext}"
        save_path = os.path.join(output_dir, new_name)
        # `img_array` has shape (1, h, w, 3) because batch_size=1
        tf.keras.utils.save_img(save_path, img_array[0])


def main() -> None:
    """Entry point for the script.

    This function orchestrates the data preparation, model construction,
    training and evaluation.  It calls the helper functions defined above
    with sensible defaults.
    """
    # Ensure a consistent test split
    ensure_test_split(DATA_ROOT, test_count=10)

    # Create data generators
    train_gen, val_gen, test_gen = prepare_generators(DATA_ROOT, image_size=(64, 64), batch_size=16)
    num_classes = len(train_gen.class_indices)
    input_shape = (64, 64, 3)

    # Train and evaluate a small CNN
    print("\nTraining the simple CNN model…")
    simple_model = build_basic_classifier(input_shape, num_classes)
    train_and_evaluate(simple_model, train_gen, val_gen, test_gen,
                       epochs=10, output_dir=cnn_output_path())

    # Train and evaluate a ResNet50‑based model
    print("\nTraining the ResNet50 transfer learning model…")
    resnet_model = build_resnet_classifier(input_shape, num_classes)
    train_and_evaluate(resnet_model, train_gen, val_gen, test_gen,
                       epochs=10, output_dir=resnet_output_path())


def cnn_output_path() -> str:
    """Return the absolute path to the CNN output directory."""
    return os.path.normpath(CNN_OUTPUT_DIR)


def resnet_output_path() -> str:
    """Return the absolute path to the ResNet output directory."""
    return os.path.normpath(RESNET_OUTPUT_DIR)


if __name__ == '__main__':
    main()