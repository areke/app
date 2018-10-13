import urllib2
import json
files = json.load(open("data.json", "r"))
for i in range(len(files)):
    print i
    response = urllib2.urlopen(files[i]["thumbnail_image_url"])
    html = response.read()
    f = open("imgs/" + str(i) + ".jpg", "w")
    f.write(html)
    f.close()