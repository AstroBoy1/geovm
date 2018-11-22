"""
Copyright 2017 TensorFlow Authors and Kent Sommer
Edited by Michael Omori
http://www.apache.org/licenses/LICENSE-2.0
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


def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    im = cv2.resize(im, (299, 299))
    im = inception_v4.preprocess_input(im)
    if K.image_data_format() == "channels_first":
        im = np.transpose(im, (2, 0, 1))
        im = im.reshape(-1, 3, 299, 299)
    else:
        im = im.reshape(-1, 299, 299, 3)
    return im


def remove_broken_links(fn, threshold=5000):
    # Get the size of an image in bytes, anything less than 5000 bytes is a broken link
    if os.stat(fn).st_size < threshold:
        os.remove(fn)


def main():
    # https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
    # Use better and more images, at least take the ones without anything
    """TODO: Save only weights from geolocation regression model. Try sigmoid activation * max_value"""

    image_dir = "geoimages_all/"
    metadata_dir = "geoimages_regional/photo_metadata.csv"
    wp = "network-weights/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5"
    epochs = 1

    # Create model and load pre-trained weights
    model = inception_v4.create_model(weights='imagenet', include_top=False, weights_path=wp)

    # Freeze the inception base, going to just train the dense network
    for i in range(0, len(model.layers) - 1):
         model.layers[i].trainable = False

    model.compile(optimizer='rmsprop', loss='mse')

    size = 500
    train_test_ratio = 0.8
    train_val_ratio = 0.8
    end = int(size * train_test_ratio)
    mid = int(end * train_val_ratio)
    df = pd.read_csv(metadata_dir, nrows=size)

    train_df = df[:mid]
    validtion_df = df[mid:end]
    test_df = df[end:]

    datagen = ImageDataGenerator(rescale=1. / 255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="id",
        y_col="latitude",
        has_ext=False,
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(299, 299))

    valid_generator = datagen.flow_from_dataframe(
        dataframe=validtion_df,
        directory=image_dir,
        x_col="id",
        y_col="latitude",
        has_ext=False,
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(299, 299))

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

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    start = time.time()
    history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, epochs=epochs)
    end = time.time()
    print("Time to train", round(end - start))

    #plt.plot(history.history['loss'], label='Training loss')
    #plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    #plt.title('Model Loss')
    #plt.ylabel('RMSE')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()

    #test_generator.reset()
    # pred = model.predict_generator(test_generator, verbose=1)

    model.save("network-weights/geo_regression.h5")
    print("Saved model")
    K.clear_session()
    # with open("history", 'wb') as f:
    #     pickle.dump(history, f)

    # with open('history', 'rb') as f:
    #     history = pickle.load(f)

    # Open Class labels dictionary. (human readable label given ID)
    # classes = eval(open('validation_utils/class_names.txt', 'r').read())

    # Load test image!
    # img_path = 'elephant.jpg'
    # img = get_processed_image(img_path)

    # Run prediction on test image
    # preds = model.predict(img)
    # print("Class is: " + classes[np.argmax(preds) - 1])
    # print("Certainty is: " + str(preds[0][np.argmax(preds)]))

if __name__ == "__main__":
    main()
