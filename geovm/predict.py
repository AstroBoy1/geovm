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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    image_dir = "geoimages_all/"
    metadata_dir = "geoimages_regional/photo_metadata.csv"
    model_fn = "network-weights/10kimages__100epochs_all_model.h5"
    #size = 10
    find_closest = True
    num_test = 10000

    # Contains the image file names, along with the targets
    # batch size needs to be a multiple of the # of images
    df  = pd.read_csv(metadata_dir, nrows=num_test)
    test_df = df[:num_test]
    datagen = ImageDataGenerator(rescale=1. / 255.)
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col="id",
        y_col=None,
        has_ext=False,
        batch_size=1,
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
    # Save the best prediction and worst prediction
    errors = []
    #print("predictions", pred)
    index = 0
    for pair in list(pred):
        latitude = pair[0]
        longitude = pair[1]
        # print(latitude, longitude)
        real_lat = df['latitude'][index]
        real_long = df['longitude'][index]
        error = pow((latitude - real_lat), 2) + pow((longitude - real_long), 2)
        errors.append(error)
        index += 1

    output = pd.DataFrame()
    output['errors'] = errors
    output['id'] = df['id'][:num_test]
    output['latitude'] = df['latitude'][:num_test]
    output['longitude'] = df['longitude'][:num_test]
    output['lat_preds'] = [p[0] for p in list(pred)]
    output['long_preds'] = [p[1] for p in list(pred)]

    output_sorted = output.sort_values(by=['errors'], ascending=True)
    #print("Best output", output_sorted[0])
    #print("Worst output", output_sorted[-1])
    output_sorted.to_csv("closest_image_dfs/predictions_3.csv")

        #if find_closest:
            #output_fn = "closest_image_dfs/" + str(round(latitude)) + str(round(longitude)) + ".csv"
            #closest(latitude=latitude, longitude=longitude, output_fn=output_fn)
    K.clear_session()


if __name__ == "__main__":
    main()
