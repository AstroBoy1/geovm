# First, you should install flickrapi
# pip install flickrapi

import flickrapi
import urllib.request
from PIL import Image

# Flickr api access key
flickr = flickrapi.FlickrAPI(u'a7abf9679953e2cb03fe748fad23f798', u'ccbbdce75269f4b8', cache=True)

keyword = 'United States'

# photos = flickr.walk(text=keyword,
#                      tag_mode='all',
#                      tags=keyword,
#                      extras='url_c',
#                      per_page=100,  # may be you can try different numbers..
#                      sort='relevance')
# photos = flickr.photos.getPopular()
photos = flickr.photos.getWithGeoData(privacy_filter=1, media='photos', extras='url_o')

urls = []
for i, photo in enumerate(photos):
    print(i)

    url = photo.get('url_o')
    urls.append(url)

    # get 50 urls
    if i > 50:
        break

print(urls)

# Download image from the url and save it to '00001.jpg'
urllib.request.urlretrieve(urls[1], '00001.jpg')

# Resize the image and overwrite it
image = Image.open('00001.jpg')
image = image.resize((256, 256), Image.ANTIALIAS)
image.save('00001.jpg')