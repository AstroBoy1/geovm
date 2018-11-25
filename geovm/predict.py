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
from keras.models import load_model
from closest_location import closest

# If you want to use a GPU set its index here
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    image_dir = "geoimages_all/"
    metadata_dir = "geoimages_regional/photo_metadata.csv"
    model_fn = "network-weights/geo_regression_latlong.h5"
    size = 10
    find_closest = True

    # Contains the image file names, along with the targets
    # batch size needs to be a multiple of the # of images
    df  = pd.read_csv(metadata_dir, nrows=size)
    test_df = df[:64]
    datagen = ImageDataGenerator(rescale=1. / 255.)
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col="id",
        y_col=None,
        has_ext=False,
        batch_size=2,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(299, 299))
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    model = load_model(model_fn)
    print("Loaded model")
    pred = model.predict_generator(test_generator, verbose=1, steps=STEP_SIZE_TEST)
    print("Finished predicting")
    #print(list(pred))
    for pair in list(pred):
        latitude = pair[0]
        longitude = pair[1]
        print(latitude, longitude)
        if find_closest:
            output_fn = "closest_image_dfs/" + str(round(latitude)) + str(round(longitude)) + ".csv"
            closest(latitude=latitude, longitude=longitude, output_fn=output_fn)
    K.clear_session()


if __name__ == "__main__":
    main()
