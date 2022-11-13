import base64
import mysql.connector
import urllib.request

def url_to_jpg(ID, title, url):
    filepath = 'manga_images/'
    filename = '{}.jpg'.format(ID)
    fullpath = filepath+filename
    try:
        urllib.request.urlretrieve(url, fullpath)
        print('{} saved from url: {}'.format(fullpath, url))
    except:
        print('----------------------------------------------\nfailed to save {} from url: {}'.format(fullpath, url))
        return 'failed to save {} from url: {}'.format(title, url)

dataBase = mysql.connector.connect(
    host="washington.uww.edu",
    user="stremmeltr18",
    passwd=base64.b64decode(b'dHM1NjEy').decode("utf-8"),
    database="manga_rec"
)
myCursor = dataBase.cursor()

myCursor.execute("select id, title, pictureLink from manga;")
mangaImgURLs = [x for x in myCursor]
print(len([x for x in mangaImgURLs if x[1] is not None]))
brokenLinks = []

for mangaId, mangaTitle, mangaULR in mangaImgURLs:
    if mangaULR is not None:
        brokenLinks.append(url_to_jpg(mangaId, mangaTitle, mangaULR))
print(brokenLinks)
