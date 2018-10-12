import pandas as pd
import urllib.request
import mysql.connector
from mysql.connector import errorcode
import numpy as np

def main():
    cnx = mysql.connector.connect(user='root', host='35.232.221.209', passwd='1fbDKLcEvKfBiItK')
    cursor = cnx.cursor()
    df = pd.read_sql('SELECT * FROM metadata.imagemetadata LIMIT 10', con=cnx)
    cursor.execute("DROP TABLE IF EXISTS metadata.url")
    statement = "CREATE TABLE metadata.url(id BIGINT, url VARCHAR(100))"
    cursor.execute(statement)
    print("Metadata table", df)
    for i in range(len(df)):
        id = df["id"][i]
        farm_id = str(df['flickr_farm'][i])
        server_id = str(df['flickr_server'][i])
        identity = str(df['id'][i])
        secret = str(df['flickr_secret'][i])
        url = "https://farm" + farm_id + ".staticflickr.com/" + server_id + "/" + identity + "_" + secret + ".jpg" 
        sql = "INSERT INTO metadata.url (id, url) VALUES(%s, %s)"
        val = (int(id), url)
        cursor.execute(sql, val)
    cnx.commit()
    df = pd.read_sql("SELECT t1.id, t1.url FROM metadata.url AS t1, metadata.imagemetadata AS t2 where t1.id = t2.id", con=cnx)
    print("URL Table", df)
    cnx.close()


if __name__ == "__main__":
    main()

