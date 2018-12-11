import pandas as pd
import sys, getopt
from PIL import Image


def closest(fn="geoimages_regional/10kimages.csv", image_fn= "images.csv", latitude=43.5, longitude=75.2, output_fn= "closest_location_output.csv", chunks=1000, n_images=10, im_out="closest_images"):
    """Finds the closest images to the input
    i: input file that contains the images with lat, long, id, and accuracy
    a: latitude
    l: longitude
    o: output filename
    Returns: sorted dataframe of closest images""" 
    
    reader = pd.read_csv(fn, chunksize=chunks)
    count = 0
    best_distance = float('inf')
    best_index = 0

    df_distance = pd.DataFrame()
    fn_list = []
    distance_list = []

    for chunk_index, df in enumerate(reader):
        #print("Chunk #:", chunk_index)
        df = df.sample(frac=1, replace=False)
        for index, row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            fn_list.append(int(row['id']))
            distance = pow(pow((latitude - lat), 2) + pow((longitude - lon), 2), 1/2)
            distance_list.append(distance)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        
        #if chunk_index > 1000:
            #break
    print("Closest distance:", best_distance)
    #print("Index", best_index)
    df_distance['id'] = fn_list
    df_distance['distance'] = distance_list
    sorted_df = df_distance.sort_values(by='distance', ascending=True)
    #sorted_df = sorted_df[:n_images]
    print("sorted values")
    sorted_df.to_csv(output_fn)
    count = 0
    print("10 filenames", fn_list[:10])
    while count < n_images:
        for i in range(0, len(sorted_df)):
            fn = str(sorted_df['id'][i]) + ".jpg"
            try:
                im = Image.open("geoimages_all" + "/" + fn)
                #im.save(im_out + "/" + fn)
                count += 1
                #print("file found")
                if count > n_images:
                    break
            except FileNotFoundError:
                print("File not found", fn)
    return sorted_df


def main(argv):
    """Finds the closest images to the input
    i: input file that contains the images with lat, long, id, and accuracy
    a: latitude
    l: longitude
    o: output filename
    Returns: sorted dataframe of closest images"""
    fn = "geoimages_regional/10kimages.csv"
    image_fn = "images.csv"
    latitude = 43.5
    longitude = 75.2
    output_fn = "closest_location_output.csv" 
    try:
        opts, args = getopt.getopt(argv, "iaol", ["ifile="])
    except getopt.GetoptError:
        print("Incorrect options")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            fn = arg
        elif opt in ("-a"):
            latitude = arg
        elif opt in ("l"):
            longitude = arg
        elif opt in ("o"):
            output_fn = arg
    print("File name", fn)
    
    chunks = 1000
    reader = pd.read_csv(fn, chunksize=chunks)
    count = 0
    best_distance = float('inf')
    best_index = 0

    df_distance = pd.DataFrame()
    fn_list = []
    distance_list = []

    for chunk_index, df in enumerate(reader):
        #print("Chunk #:", chunk_index)
        df = df.sample(frac=0.1, replace=False)
        for index, row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            fn_list.append(int(row['id']))
            distance = pow(pow((latitude - lat), 2) + pow((longitude - lon), 2), 1/2)
            distance_list.append(distance)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if chunk_index > 1000:
            break
    #print("Closest distance:", best_distance)
    #print("Index", best_index)
    df_distance['id'] = fn_list
    df_distance['distance'] = distance_list
    sorted_df = df_distance.sort_values(by='distance', ascending=True)
    print("sorted values")
    sorted_df.to_csv(output_fn)
    count = 0
    print("10 filenames", fn_list[:10])
    while count < 10:
        for i in range(0, len(sorted_df)):
            fn = str(sorted_df['id'][i]) + ".jpg"
            try:
                im = Image.open("geoimages_all" + "/" + fn)
                #im.save("closest_images/" + fn)
                count += 1
                if count > 10:
                    break
            except FileNotFoundError:
                print("File not found", fn)
    return sorted_df


if __name__ == "__main__":
    #main(sys.argv[1:])
    df = pd.read_csv("closest_image_dfs/predictions_3.csv")
    ids = ['6385911009.jpg', '6341360361.jpg', '6341358361.jpg', '3569191177.jpg', '3570004142.jpg', '3570003476.jpg']
    for name in ids:
        output_fn = "closest_image_dfs/" + name[:-4] + "predictions" + ".csv"
        im_out = "closest_images/" + name
        real_lat = df.loc[df['id'] == int(name[:-4])]['lat_preds'].values[0]
        real_long = df.loc[df['id'] == int(name[:-4])]['long_preds'].values[0]
        print(real_lat)
        print(real_long)
        closest(output_fn=output_fn, im_out=im_out, latitude=real_lat, longitude=real_long)
        print("Finished for one image")

