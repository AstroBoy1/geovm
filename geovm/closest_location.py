import pandas as pd
import sys, getopt


def main(argv):
    """TODO:just take the first 10k files and also open up the images"""
    fn = "geoimages_regional/photo_metadata.csv"

    try:
        opts, args = getopt.getopt(argv, "i", ["ifile="])
    except getopt.GetoptError:
        print("Incorrect options")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            fn = arg
    print("File name", fn)
    latitude = 43.5
    longitude = 75.2
    chunks = 1000
    reader = pd.read_csv(fn, chunksize=chunks)
    count = 0
    best_distance = float('inf')
    best_index = 0

    df_distance = pd.DataFrame()
    fn_list = []
    distance_list = []

    for chunk_index, df in enumerate(reader):
        print("Chunk #:", chunk_index)
        df = df.sample(frac=0.1, replace=False)
        for index, row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            fn_list.append(row['id'])
            distance = pow(pow((latitude - lat), 2) + pow((longitude - lon), 2), 1/2)
            distance_list.append(distance)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if chunk_index > 1000:
            break
    print("Closest distance:", best_distance)
    print("Index", best_index)
    df_distance['id'] = fn_list
    df_distance['distance'] = distance_list
    sorted_df = df_distance.sort_values(by='distance', ascending=True)
    sorted_df.to_csv("close_images.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
