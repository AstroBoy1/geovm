import flickrapi


api_key = u'a7abf9679953e2cb03fe748fad23f798'
api_secret = u'ccbbdce75269f4b8'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

photoId = 1418878
photoSizes = flickr.photos.getSizes(photo_id=photoId)

photos = flickr.photos.search(user_id='73509078@N00', per_page='10', extras='url_o')
sets = flickr.photosets.getList(user_id='73509078@N00')

title = sets['photosets']['photoset'][0]['title']['_content']

print('First set title: %s' % title)

# location = flickr.photos.geo.getLocation(photo_id=1)

flickr.authenticate_via_browser(perms='read')

# Get public photos and geo-data
sets2 = flickr.photos.getWithGeoData(privacy_filter=1, media='photos')

for node in sets2:
    print(node)


photos = flickr.photos.geo.photosForLocation(lat=21.2870198, lon=-157.839106)
