import pandas as pd


def main():
    """TODO: Need to ensure that the files returned are in the GCS"""
    latitude = 43.5
    longitude = 75.2
    chunks = 1000
    reader = pd.read_csv("photo_metadata.csv", chunksize=chunks)
    count = 0
    best_distance = float('inf')
    best_index = 0

    df_distance = pd.DataFrame()
    fn_list = []
    distance_list = []

    for index, df in enumerate(reader):
        for i in range(chunks * index, chunks * index + len(df)):
            count += 1
            lat = df['latitude'][i]
            lon = df['longitude'][i]
            fn_list.append(df['id'][i])
            distance = pow(pow((latitude - lat), 2) + pow((longitude - lon), 2), 1/2)
            distance_list.append(distance)
            if distance < best_distance:
                best_distance = distance
                best_index = i
        if count > 100000:
            break
    print("Closest distance:", best_distance)
    print("Index", best_index)
    df_distance['id'] = fn_list
    df_distance['distance'] = distance_list
    sorted_df = df_distance.sort_values(by='distance', ascending=True)


if __name__ == "__main__":
    main()
