"""
Michael Omori
Make predictions
"""

from keras import backend as K
from inception import inception_v4
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time

# If you want to use a GPU set its index here
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    image_dir = "test_images/"
    metadata_dir = "geoimages_regional/photo_metadata.csv"
    model_fn = "network-weights/geo_regression.h5"

    # Contains the image file names, along with the targets
    test_df = pd.read_csv(metadata_dir, nrows=size)
    datagen = ImageDataGenerator(rescale=1. / 255.)
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col="id",
        y_col=None,
        has_ext=False,
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(299, 299))
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    model = load_model(model_fn)
    pred = model.predict_generator(test_generator, verbose=1)


if __name__ == "__main__":
    main()
