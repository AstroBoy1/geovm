import pandas as pd


def main():
    """TODO: Get the closest image fn"""
    latitude = 43.5
    longitude = 75.2
    chunks = 1000
    reader = pd.read_csv("photo_metadata.csv", chunksize=chunks)
    count = 0
    best_distance = float('inf')
    best_index = 0
    for index, df in enumerate(reader):
        for i in range(chunks * index, chunks * index + len(df)):
            count += 1
            lat = df['latitude'][i]
            lon = df['longitude'][i]
            distance = pow(pow((latitude - lat), 2) + pow((longitude - lon), 2), 1/2)
            if distance < best_distance:
                best_distance = distance
                best_index = i
        if count > 100000:
            break
    print("Closest distance:", best_distance)
    print("Index", best_index)


if __name__ == "__main__":
    main()
