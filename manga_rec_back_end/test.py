import csv
import mysql.connector
import json
import base64
'''count = 0
with open('manga_results/manga_cleaned.csv', encoding='utf-8') as csv_file_in:
    csv_reader = csv.reader(csv_file_in, delimiter=',')
    for row in csv_reader:
        count += 1
        #print(row)
        if len(row) != 11:
            print(len(row))

print(count)'''

def compareSets(knn, LSH, length):
    overlap = 0
    for idx in range(length):
        if knn[idx] in LSH[0:length]:
            overlap += 1
    print("%s manga overlapped out of %s aka %s percent" % (overlap, length, (overlap*100/length)))


knnResults = [2168, 548, 184, 224, 488, 2366, 5995, 674, 2043, 777, 1113, 2113, 2049, 136, 2038, 2125, 2236, 553, 827, 339, 1739, 789, 1162, 33, 466, 963, 1670, 3296, 249, 4430, 4, 203, 1590, 1845, 1996, 2965, 313, 529, 250, 1457, 1858, 2613, 5063, 1165, 1636, 1959, 2817, 6747, 7326, 115, 480, 929, 1074, 1412, 177, 333, 486, 591, 1203, 1335, 2301, 3694, 1291, 611, 2412, 6292, 152, 717, 1119, 3302, 3583, 1523, 2001, 133, 577, 4724, 103, 378, 517, 1081, 1377, 1854, 641, 735, 592, 1108, 191, 111, 1861, 632, 1318, 733, 1025, 1638, 2611, 14, 1334, 47, 157, 462]

LSHResults = [2128, 963, 1162, 17, 1165, 1579, 2049, 1861, 184, 2250, 548, 136, 529, 777, 2113, 688, 2366, 1074, 1458, 201, 332, 378, 894, 2125, 313, 717, 7, 463, 611, 1081, 1167, 1845, 2043, 122, 1996, 27, 19, 224, 339, 826, 1113, 2300, 4724, 1057, 177, 4055, 491, 111, 229, 234, 249, 260, 480, 623, 643, 250, 412, 592, 733, 2739, 3245, 3694, 5276, 1380, 4, 170, 267, 601, 789, 1334, 1636, 2729, 3424, 642, 1377, 1590, 2114, 6333, 551, 945, 1291, 1625, 1667, 1726, 2627, 2965, 3583, 309, 2441, 1318, 365, 2746, 436, 11, 33, 62, 211, 886, 1670, 2038]

lengths = [25, 50, 100]
for i in lengths:
    compareSets(knnResults, LSHResults, i)



def getColumnLengths():
    dataBase = mysql.connector.connect(
        host="washington.uww.edu",
        user="stremmeltr18",
        passwd=base64.b64decode(b'dHM1NjEy').decode("utf-8"),
        database="manga_rec"
    )
    myCursor = dataBase.cursor()
    # get user's manga ratings

    myCursor.execute("select min(popularity), max(popularity) from manga")
    print("popularity min,max values:")
    print([x for x in myCursor][0])

    myCursor.execute("select min(releaseDate), max(releaseDate) from manga")
    print("releaseDate min,max values:")
    print([x for x in myCursor][0])

    myCursor.execute("select min(chapterCount), max(chapterCount) from manga")
    print("chapterCount min,max values:")
    print([x for x in myCursor][0])

    myCursor.execute("select DISTINCT status from manga")
    print("status values:")
    print([x[0].replace("\"", "") for x in myCursor])

    myCursor.execute("select DISTINCT genre from manga")
    print("genre values:")
    genreClumps = '|'.join([x[0].replace("\"", "") for x in myCursor if x[0] is not None])
    genreSet = set(genreClumps.split("|"))
    print(genreSet)
    print("genre count:", len(genreSet))

    myCursor.execute("select DISTINCT theme from manga")
    print("theme values:")
    themeClumps = '|'.join([x[0].replace("\"", "") for x in myCursor if x[0] is not None])
    themeSet = set(themeClumps.split("|"))
    print(themeSet)
    print("theme count:", len(themeSet))

    myCursor.execute("select DISTINCT demographic from manga")
    print("demographic values:")
    demographicClumps = '|'.join([x[0].replace("\"", "") for x in myCursor if x[0] is not None])
    demographicSet = set(demographicClumps.split("|"))
    print(demographicSet)
    print("demographic count:", len(demographicSet))


#getColumnLengths()
