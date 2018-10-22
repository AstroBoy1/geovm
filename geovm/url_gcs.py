import pandas as pd
import urllib.request
import time
import csv
import sys


def download_image(fn, chunk=10, number=100):
    """Download images locally from csv file with id and url column"""
    reader = pd.read_csv(fn, chunksize=chunk)
    count = 0
    start = time.time()
    for df in reader:
        for i in range(len(df)):
            url = df["url"][i + count]
            urllib.request.urlretrieve(url, "images/" + str(df["id"][i + count]) + ".jpg")
        count += chunk
        print(count)
        if count > number:
            break
    end = time.time()
    with open('timing.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([10, "seconds"])


def main():
    params = sys.argv[1:]
    c = 5
    n = 10
    if len(params) >= 1:
        c = int(params[0])
    if len(params) == 2:
        n = int(params[1])
    download_image("urls.csv", c, n)


if __name__ == "__main__":
    main()
