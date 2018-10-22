import pandas as pd
import urllib.request
import mysql.connector
from mysql.connector import errorcode
import csv
import time


def read_sql():
    try:
        cnx = mysql.connector.connect(user='root', host='195.206.104.109', passwd='1fbDKLcEvKfBiItK')
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        cnx.close()


def download_image(chunk=100, number=1000):
    """Download images locally from csv file with id and url column"""
    reader = pd.read_csv('urls.csv', chunksize=chunk)
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
    print("Time to download", number, "images:", round(end - start), " seconds")


def bulk_url_builder(chunk=10000):
    """Lags at chunk size 1M, 100k is good"""
    reader = pd.read_csv('photo_metadata.csv', chunksize=chunk)
    count = 0
    with open('urls.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "url"])
        for df in reader:
            for i in range(len(df)):
                farm_id = str(df['flickr_farm'][i + count])
                server_id = str(df['flickr_server'][i + count])
                identity = str(df['id'][i + count])
                secret = str(df['flickr_secret'][i + count])
                url = "https://farm" + farm_id + ".staticflickr.com/" + server_id + "/" + identity + "_" + secret + ".jpg"
                # urllib.request.urlretrieve(url, str(count + i) + ".jpg")
                writer.writerow([identity, url])
            count += chunk
            print(count)


def url_check(file):
    df = pd.read_csv(file, nrows=10)
    return df['id']


# class MyTest(unittest.TestCase):
#     def test(self):
#         self.assertEqual(url_check("urls.csv"), url_check("photo_metadata.csv"))


def main():
    """110 seconds per 100 images. 1M images, 1M seconds"""
    # read_sql()
    # bulk_url_builder()
    # try:
    #     os.remove("1.jpg")
    # except FileNotFoundError:
    #     print("File not found")
    # return 1
    download_image()


if __name__ == "__main__":
    main()
