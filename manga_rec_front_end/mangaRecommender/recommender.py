import json
import base64
import mysql.connector
import sys
import math
import time
from random import randint
import numpy as np
from datasketch import MinHashLSHForest, MinHash
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from collections import Counter
import pickle

np.set_printoptions(suppress=True)

ratingMap = {5: 'likes', 4: 'interested', 3: 'neutral', 2: 'dislikes', 1: 'not-interested'}


def convertRating(ratingTuple):
    mangaId = ratingTuple[2]
    status = ratingTuple[3]
    originalRating = ratingTuple[4]

    if originalRating is not None:
        if originalRating >= 7:
            return mangaId, 5  # 5
        if originalRating >= 5:
            return mangaId, 3  # 3
        else:  # if rating < 5
            return mangaId, 2  # 2
    if status == 'reading' or status == 'completed':
        return mangaId, 4  # 4
    if status == 'plan_to_read':
        return mangaId, 4  # 4
    if status == 'on_hold':
        return mangaId, 3  # 3
    if status == 'dropped':
        return mangaId, 2  # 2
    if status == 'not interested':
        return mangaId, 1
    return mangaId, 4  # 4  # happens if status and rating are none.
    # list of assumptions: if you finished it you liked it
    # if you are reading it or plan to read it you are interested in it
    # if you rated it we map that into liked, disliked, neutral
    # on hold maps to neutral (we cannot make any assumptions with on hold imo so neutral is best)
    # dropped maps to disliked since we assume there was something you disliked about it that caused you to drop it


def convertRating_forKnn(ratingTuple):
    convertedRating = convertRating(ratingTuple)
    return ratingTuple[1], convertedRating[0], convertedRating[1]  # userId, mangaId, convertedRatingValue


def satisfiesFilters(mangaInstance, filters):
    # if I want include check for include exclude overlap
    if not (filters[0][0] <= mangaInstance[1] <= filters[0][1]):
        return False  # popularity
    if mangaInstance[2] is not None:
        if not (filters[1][0] <= mangaInstance[2] <= filters[1][1]):
            return False  # releaseDate
    if mangaInstance[3] is not None:
        if not (filters[2][0] <= mangaInstance[3] <= filters[2][1]):
            return False  # chapterCount
    for i in range(len(filters[3])):
        if filters[3][i] is True and mangaInstance[4 + i] == 1:
            return False  # exclude status
    for i in range(len(filters[4])):
        if filters[4][i] is True and mangaInstance[8 + i] == 0:
            return False  # include genre
    for i in range(len(filters[5])):
        if filters[5][i] is True and mangaInstance[26 + i] == 0:
            return False  # include theme
    for i in range(len(filters[6])):
        if filters[6][i] is True and mangaInstance[77 + i] == 0:
            return False  # include demographic
    for i in range(len(filters[7])):
        if filters[7][i] is True and mangaInstance[8 + i] == 1:
            return False  # exclude genre
    for i in range(len(filters[8])):
        if filters[8][i] is True and mangaInstance[26 + i] == 1:
            return False  # exclude theme
    for i in range(len(filters[9])):
        if filters[9][i] is True and mangaInstance[77 + i] == 1:
            return False  # exclude demographic
    return True


def getExcludedIds(encodedManga, rawFeaturesThatWereNormalized, filters):
    excludedIds = []
    for i in range(len(encodedManga)):
        if not satisfiesFilters(np.concatenate((rawFeaturesThatWereNormalized[i][:4], encodedManga[i][4:])), filters):
            excludedIds.append(encodedManga[i][0])
    return excludedIds


def getStatusSet():
    return ['On Hiatus', 'Finished', 'Publishing', 'Discontinued']


def getGenreSet():
    return ['Adventure', 'Comedy', 'Slice of Life', 'Boys Love', 'Sci-Fi', 'Action', 'Horror', 'Suspense', 'Girls Love',
            'Gourmet', 'Sports', 'Avant Garde', 'Supernatural', 'Fantasy', 'Romance', 'Ecchi', 'Drama', 'Mystery']


def getThemeSet():
    return ['Historical', 'Time Travel', 'Visual Arts', 'Military', 'Love Polygon', 'Mecha', 'Martial Arts', 'Racing',
            'Samurai', 'Strategy Game', 'CGDCT', 'Mythology', 'High Stakes Game', 'Idols (Male)', 'Reincarnation',
            'Pets', 'Team Sports', 'Workplace', 'Isekai', 'Gag Humor', 'Memoir', 'Harem', 'Villainess', 'Detective',
            'Performing Arts', 'Reverse Harem', 'Childcare', 'Otaku Culture', 'Mahou Shoujo', 'Anthropomorphic',
            'Survival', 'Magical Sex Shift', 'Music', 'Delinquents', 'Organized Crime', 'Adult Cast', 'Medical',
            'Showbiz', 'Crossdressing', 'Gore', 'Psychological', 'School', 'Combat Sports', 'Parody',
            'Romantic Subtext', 'Space', 'Iyashikei', 'Video Game', 'Educational', 'Vampire', 'Super Power']


def getDemographicSet():
    return ['Kids', 'Seinen', 'Shoujo', 'Josei', 'Shounen']


def encodeManga(manga, filters):
    mangaEncoded = []
    excludedIds = []
    statusSet = getStatusSet()
    genreSet = getGenreSet()
    themeSet = getThemeSet()
    demographicSet = getDemographicSet()
    maxPopularity = 27691
    minPopularity = 1
    maxReleaseDate = 2022
    minReleaseDate = 1946
    maxChapterCount = 6477
    minChapterCount = 1
    for mangaInstance in manga:
        # make fill release date for None type,
        # uses 2008 as fill for when release date is none. 2008 is average release data and there are only 100 manga...
        # with no release date
        # rn chapterCount is set to 40 if none, (avg is 40) ************************************************************
        instanceData = [mangaInstance[0], mangaInstance[1], mangaInstance[4] if mangaInstance[4] is not None else 2008,
                        mangaInstance[5] if mangaInstance[5] is not None else 40]

        mangaInstanceStatusSet = mangaInstance[6]
        for i in range(len(statusSet)):
            if statusSet[i] == mangaInstanceStatusSet:
                instanceData.append(1)
            else:
                instanceData.append(0)
        mangaInstanceGenreSet = mangaInstance[7].split('|') if mangaInstance[7] is not None else []
        for i in range(len(genreSet)):
            if genreSet[i] in mangaInstanceGenreSet:
                instanceData.append(1)
            else:
                instanceData.append(0)
        mangaInstanceThemeSet = mangaInstance[8].split('|') if mangaInstance[8] is not None else []
        for i in range(len(themeSet)):
            if themeSet[i] in mangaInstanceThemeSet:
                instanceData.append(1)
            else:
                instanceData.append(0)
        mangaInstanceDemographicSet = mangaInstance[9].split('|') if mangaInstance[9] is not None else []
        for i in range(len(demographicSet)):
            if demographicSet[i] in mangaInstanceDemographicSet:
                instanceData.append(1)
            else:
                instanceData.append(0)

        if not satisfiesFilters(instanceData, filters):
            excludedIds.append(mangaInstance[0])

        # normalize release date, chapter count, and popularity between 0 and 1 (new)
        #this is done after filtering because the filters use the raw popularity, chapter count, and release date values
        instanceData[1] = (instanceData[1] - minPopularity) / (maxPopularity - minPopularity)
        instanceData[2] = (instanceData[2] - minReleaseDate) / (maxReleaseDate - minReleaseDate)
        instanceData[3] = (instanceData[3] - minChapterCount) / (maxChapterCount - minChapterCount)

        mangaEncoded.append(instanceData)

    return mangaEncoded, excludedIds


def cosineSimilarity(vector1, vector2):
    sumXX, sumXY, sumYY = 0, 0, 0
    for i in range(len(vector1)):
        X = vector1[i]
        Y = vector2[i]
        sumXX += X * X
        sumYY += Y * Y
        sumXY += X * Y
    return sumXY / math.sqrt(sumXX * sumYY) if sumXX * sumYY > 0 else 0

def jaccardSimilarity(set1, set2):
    return float(len(set1.intersection(set2))) / float(len(set1.union(set2)))

def knn(myCursor, mangaIds, userId, testSet, k, useLocalRatings, runLSH, useLSH, runPackageLSHCode, numLSHPermutations):
    # 130 is sqrt(num users) 65 is sqrt(num users)/2  <- both common choice of k

    convertedRatings = []
    if useLocalRatings:
        convertedRatings = np.load('mangaRecommender/convertedRatings.npy')
    else:
        myCursor.execute("select * from ratings;")
        ratings = myCursor.fetchall()  # [x for x in myCursor]
        convertedRatings = [convertRating_forKnn(x) for x in ratings]
    myCursor.execute("select distinct userId from users;")
    userIds = myCursor.fetchall()
    userIds = [x[0] for x in userIds]
    knn_time = time.time()
    # user and manga dict are used to map from the user/manga ids to there index in the user_matrix
    mangaDict = dict()
    for index, value in enumerate(mangaIds):
        mangaDict[value] = index
    userDict = dict()
    for index, value in enumerate(userIds):
        userDict[value] = index
    userIndex = userDict[userId]

    # start of section unique to knn
    user_matrix = np.zeros((len(userIds), len(mangaIds)), dtype=np.int8)
    for i in range(len(convertedRatings)):
        if convertedRatings[i][0] == userId:
            if convertedRatings[i][1] in testSet:
                continue  # dont include manga in the user's test set
        user_matrix[userDict[convertedRatings[i][0]], mangaDict[convertedRatings[i][1]]] = convertedRatings[i][2]


    # I tried csr and csc.  csr: 18s  csc: 15.16s  coo: 14.6s
    coo_user_matrix = sparse.coo_matrix(user_matrix)
    similarities = cosine_similarity(coo_user_matrix)
    similarities_toUser = np.zeros(shape=(len(user_matrix), 2))
    for i in range(len(similarities[userDict[userId]])):
        similarities_toUser[i][0] = userIds[i]  # or i or userDict[i]
        if i == userDict[userId]:
            continue  # the user will have a 0 sim to its self
        similarities_toUser[i][1] = similarities[userIndex][i]
    similarities_toUser = similarities_toUser[similarities_toUser[:, 1].argsort()[::-1]]


    kNeighbors = []
    neighborsManga = []
    for x in range(k):
        kNeighbors.append(int(similarities_toUser[x][0]))
        neighborsManga.append(user_matrix[userDict[kNeighbors[x]]])

    if runLSH:
        similarUsers = LSH(userId, userIds, userDict, convertedRatings, numLSHPermutations, k, runPackageLSHCode)

        overlap = 0
        for i in range(len(kNeighbors)):
            if kNeighbors[i] in similarUsers:
                overlap += 1

        if useLSH:
            kNeighbors = similarUsers
            neighborsManga = []
            for i in range(len(similarUsers)):
                neighborsManga.append(user_matrix[userDict[similarUsers[i]]])

    return kNeighbors, neighborsManga


def LSH(userId, userIds, userDict, ratings, num_permutations, k, runPackageLSHCode):
    # k is number of similar users to return
    # num_permutations is number of hash functions, increase to improve results at cost of speed

    user_sets = [set() for i in range(len(userDict))]  # create empty set for each user
    for rating in ratings:
        if rating[2] >= 3:  # only include if the rating is 5: 'likes', 4: 'interested', 3: 'neutral'
            user_sets[userDict[rating[0]]].add(rating[1])

    if runPackageLSHCode:
        # ************** min hash generation using package code **************
        signature_matrix_package = []
        for user_set in user_sets:
            min_hash = MinHash(num_perm=num_permutations)
            for mangaId in user_set:
                min_hash.update(str(mangaId).encode('utf8'))
            signature_matrix_package.append(min_hash)
        # ************** min hash LSH forest generation using package code **************
        user_min_hash = None
        LSH_forest = MinHashLSHForest(num_perm=num_permutations)
        for i in range(len(signature_matrix_package)):  # should be equal to number of users
            if userIds[i] == userId:
                user_min_hash = signature_matrix_package[i]
                continue  # dont include the user in the forest
            LSH_forest.add(i, signature_matrix_package[i])  # stores user index not userId
        LSH_forest.index()
        # ************** get k neighbors using package code **************
        similarIndices = np.array(LSH_forest.query(user_min_hash, k + int(k/10)))  # include extra results
        similarUsers = [[userIds[x], user_sets[x]] for x in similarIndices]
    else:
        # ************** min hash generation using my code **************
        signature_matrix = []
        max_val = (2 ** 32) - 1
        perms = [(randint(0, max_val), randint(0, max_val)) for i in range(num_permutations)]
        for user_set in user_sets:
            min_hash = minhash(user_set, num_permutations, perms)
            signature_matrix.append(min_hash)
        # ************** min hash LSH generation using my code **************
        user_min_hash = None
        lsh = {}
        bandSize = 2  # larger number will reduce collision. worked with 2 ######################################
        for i in range(len(signature_matrix)):  # should be equal to number of users
            if userIds[i] == userId:
                user_min_hash = signature_matrix[i]  # save user min hash
                continue  # dont include the user in the lsh
            # stores userId not user index
            for x in range(len(signature_matrix[i]) - bandSize):
                hashCode = ""
                for y in range(bandSize):
                    hashCode += str(signature_matrix[i][x+y])
                if hashCode in lsh:
                    lsh[hashCode].append(userIds[i])
                else:
                    lsh[hashCode] = [userIds[i]]
        # ************** get k neighbors using my code **************
        similarUserIdCounts = Counter()
        for x in range(len(user_min_hash) - bandSize):
            hashCode = ""
            for y in range(bandSize):
                hashCode += str(user_min_hash[x + y])
            if hashCode in lsh:
                for similarUserId in lsh[hashCode]:
                    similarUserIdCounts[similarUserId] += 1
        similarUserIdCounts = dict(sorted(similarUserIdCounts.items(), key=lambda item: item[1], reverse=True))
        # ^ sorts similar users by frequency they appeared in same lsh buckets
        similarUserIds = list(similarUserIdCounts)
        similarUsers = [[x, user_sets[userDict[x]]] for x in similarUserIds]

    jaccardSims = np.array([[x[0], jaccardSimilarity(x[1], user_sets[userDict[userId]])] for x in similarUsers])
    jaccardSims = jaccardSims[jaccardSims[:, 1].argsort()[::-1]]

    return [int(jaccardSims[x][0]) for x in range(k)]

def minhash(user_set, num_permutations, perm_functions, prime=4294967311):  # no clue what to make prime 4294967311 429497
    # initialize a minhash vector of length num_permutations with positive infinity values
    vector = [float('inf') for i in range(num_permutations)]

    for val in user_set:

        # loop over each "permutation function"
        for perm_idx, perm_vals in enumerate(perm_functions):
            a, b = perm_vals

            # pass `val` through the `ith` permutation function
            output = (a * val + b) % prime

            # conditionally update the `ith` value of the vector
            if vector[perm_idx] > output:
                vector[perm_idx] = output

    # the returned vector represents the minimum hash of the user_set
    return vector

def clusteringResults(ratings, mangaIds, get_manga_array_index, clusterAlgName):
    numClusters = 100
    clusters = np.load('mangaRecommender/image_saved_data/{}.npy'.format(clusterAlgName))
    imagesMap = np.load('mangaRecommender/imagesMap.npy').tolist()
    #ResNet50Avg_100KmeansClustersBase
    #ResNet50Avg_100KmeansClustersScaled
    #ResNet152V2Avg_100HierarchicalClusters

    clusterScores = [0] * numClusters
    mangaScores = np.zeros(shape=(len(mangaIds), 2))
    for i in range(len(mangaScores)):
        mangaScores[i][0] = mangaIds[i]
    for manga in ratings:
        if '{}.jpg'.format(manga[0]) not in imagesMap:
            continue
        cluster = clusters[get_manga_index(manga[0], imagesMap)]
        clusterScores[cluster] += (manga[1] - 2)
        #does -2 so that 'negative' ratings decrease the clusterScore for that manga
    for i in range(len(clusters)):
        mangaScores[get_manga_array_index[get_manga_id(i, imagesMap)]][1] = clusterScores[clusters[i]]
        #every manga gets given the score of the cluster they belong to
    return mangaScores


def matrixResults(ratings, mangaIds, get_manga_array_index, imageFeatureSetName, matrixK):
    # value of k is very important for this algo
    # 76 is sqrt(num images) 38 is sqrt(num images)/2  <- both common choice of k
    distanceMatrix = np.load('mangaRecommender/image_saved_data/{}_distanceMatrix.npy'.format(imageFeatureSetName))
    imagesMap = np.load('mangaRecommender/imagesMap.npy').tolist()
    mangaScores = np.zeros(shape=(len(mangaIds), 2))
    for i in range(len(mangaScores)):
        mangaScores[i][0] = mangaIds[i]
    for manga in ratings:
        if '{}.jpg'.format(manga[0]) not in imagesMap:
            continue
        # KNM = k nearest manga
        KNM = np.argsort(distanceMatrix[get_manga_index(manga[0], imagesMap)])[0:matrixK]
        for index in KNM:
            mangaScores[get_manga_array_index[get_manga_id(index, imagesMap)]][1] += (manga[1] - 2)
            # does -2 so that 'negative' ratings decrease the clusterScore for that manga
    return mangaScores

def get_manga_index(manga_id, imagesMap):
    # converts manga_id (from database) to manga_index in the extracted_features array and clusters
    return imagesMap.index('{}.jpg'.format(manga_id))

def get_manga_id(manga_index, imagesMap):
    # convert manga_index (node names in the extracted_features array and clusters) to manga_id (from database)
    return int(imagesMap[manga_index][:-4])

def recommend(userId: int, filters, showResultsFromEachMethod=False, loadMangaFromLocal=False, forceTestSize=True, methodWeights=None, numMangaToReturn=50,
              k=65, runLSH=False, useLSH=False, runPackageLSHCode=False, numLSHPermutations=32, useLocalRatings=False,
              clusterAlgName='ResNet152V2Avg_100HierarchicalClusters', imageFeatureSetName='ResNet152V2Avg', matrixK=38):
    if methodWeights is None:  # this if statement works in place of default value
        methodWeights = [.7, 2, .7, .7]

    dataBase = mysql.connector.connect(
        host="washington.uww.edu",
        user="stremmeltr18",
        passwd=base64.b64decode(b'dHM1NjEy').decode("utf-8"),
        database="manga_rec"
    )
    myCursor = dataBase.cursor()

    # create one hot encoded (and other data alterations) matrix of manga table
    # possible include no_genres/no_themes column (i dont think it would be good but idk)
    manga = []
    mangaEncoded = []
    get_manga_array_index = []
    excludedIds = []
    if loadMangaFromLocal:  # does not work. idk why. must convert types into some weird shit
        with open('mangaRecommender/manga_raw.pkl', 'rb') as f:
            manga = pickle.load(f)
        with open('mangaRecommender/encodedManga.pkl', 'rb') as f:
            mangaEncoded = pickle.load(f)
        get_manga_array_index = np.load('mangaRecommender/get_manga_array_index.npy', allow_pickle=True)
        with open('mangaRecommender/rawFeaturesThatWereNormalized.pkl', 'rb') as f:
            rawFeaturesThatWereNormalized = pickle.load(f)
        excludedIds = list(getExcludedIds(mangaEncoded, rawFeaturesThatWereNormalized, filters))
    else:
        myCursor.execute("select * from manga;")
        manga = [x for x in myCursor]
        mangaEncoded, excludedIds = encodeManga(manga, filters)
        for i in range(len(manga)):
            manga[i] = list(manga[i])
            del manga[i][3]  # drop description for ease of viewing output
        # make manga_id -> manga_index np array
        get_manga_array_index = np.full(shape=10000, fill_value=-1, dtype=int)
        for i in range(len(manga)):
            get_manga_array_index[manga[i][0]] = i
    # 0:id, 1:popularity, 2: releaseDate, 3:chapterCount, 4-7:status, 8-25:genre, 26-76:theme, 77-81:demographic
    mapping = ['id', 'popularity', 'releaseDate', 'chapterCount'] + getStatusSet() + getGenreSet() + getThemeSet() + getDemographicSet()
    mangaIds = [i[0] for i in manga]

    # get user's manga ratings
    myCursor.execute("select * from ratings where userId = %s;", [userId])
    ratings = [x for x in myCursor]
    convertedRatings = [convertRating(x) for x in ratings]

    testSet = []  #only includes manga ids because we only include manga that should be recommended
    maxTestSetSize = int(len(convertedRatings)/10)
    if runExperiment:
        itemsToRemove = []
        for i in range(len(convertedRatings)):
            if len(testSet) >= maxTestSetSize:
                break
            if convertedRatings[i][1] == 4:  # aka if they are interested in the manga
                testSet.append(convertedRatings[i][0])
                itemsToRemove.append(convertedRatings[i])
        for item in itemsToRemove:
            convertedRatings.remove(item)
        itemsToRemove = []
        if forceTestSize:
            for i in range(len(convertedRatings)):
                if len(testSet) >= maxTestSetSize:
                    break
                if convertedRatings[i][1] > 2:  # aka as long as they dont dislike or are not interested in it
                    testSet.append(convertedRatings[i][0])
                    itemsToRemove.append(convertedRatings[i])
        for item in itemsToRemove:
            convertedRatings.remove(item)
        if len(convertedRatings) == 0:
            return False


    # create table of encoded manga the user has rated
    userTable = []  # list of manga the user has rated
    userRatings = []  # list of ratings for the manga they have rated
    for i in range(len(convertedRatings)):
        for j in range(len(mangaEncoded)):
            if convertedRatings[i][0] == mangaEncoded[j][0]:
                userTable.append(mangaEncoded[j])  # [4:]only uses the one hot values for now
                userRatings.append(list(convertedRatings[i]))
                break
    ratedMangaIds = [i[0] for i in userRatings]

    # create a user preference vector (average or dot product of all user ratings)
    weightedTotal = [0] * len(userTable[0])  # aka number of features
    featureCounts = [0] * len(userTable[0])
    for i in range(len(userTable)):
        for j in range(len(userTable[i])):
            if userTable[i][j] is not None:
                weightedTotal[j] += userTable[i][j] * (userRatings[i][1] - 3)
                featureCounts[j] += (userRatings[i][1] - 3)
    userProfile = [0] * len(userTable[0])
    for i in range(len(weightedTotal)):
        userProfile[i] = (weightedTotal[i] / featureCounts[i]) if featureCounts != 0 else 0


    # create recommendations
    similarityMeasures = np.zeros(shape=(len(mangaEncoded), 2))  # I think this is the right size
    # next 2 lines remove id for similarity calculation (it is not a feature) I will now try to remove them when sending the vectors to the cosign function
    userVector = userProfile[1:]  # change 1 to 2 to remove popularity feature
    for i in range(len(similarityMeasures)):
        similarityMeasures[i][0] = mangaIds[i]
        similarityMeasures[i][1] = cosineSimilarity(userVector, mangaEncoded[i][1:])

    if showResultsFromEachMethod:
        # get the manga rows that were recommended (and exclude those the user has already rated and that dont satisfy filters)
        contentBasedSorted = similarityMeasures[similarityMeasures[:, 1].argsort()[::-1]]  # sort similarityMeasures
        filteringTime = time.time()
        similarManga = []
        filteredSimilarityMeasures = []
        for i in range(len(contentBasedSorted)):
            if contentBasedSorted[i][0] not in ratedMangaIds and contentBasedSorted[i][0] not in excludedIds:
                filteredSimilarityMeasures.append(contentBasedSorted[i])
                similarManga.append(manga[get_manga_array_index[int(contentBasedSorted[i][0])]])
        print('\n'.join(["{:.3f}".format(filteredSimilarityMeasures[[i[0] for i in filteredSimilarityMeasures].index(similarManga[x][0])][1])
                         + "\t" + str(similarManga[x]) for x in range(5)]))  # HELPFUL
        print('filtering took:', time.time()-filteringTime)

    similarUsers, similarUsers_matrixRow = knn(myCursor, mangaIds, userId, testSet, k, useLocalRatings, runLSH, useLSH, runPackageLSHCode, numLSHPermutations)
    myCursor.close()
    similarMangaIdCounts = Counter()
    for i in range(len(similarUsers_matrixRow)):
        for j in range(len(similarUsers_matrixRow[i])):
            if similarUsers_matrixRow[i][j] == 4 or similarUsers_matrixRow == 5:
                # if they liked or are interested in the manga at index j
                similarMangaIdCounts[manga[j][0]] += 1  # add the mangaId for that index to list

    # next line sorts by frequency then by id (which is basically ranking) (aka ties broken by cite wide rating)
    # similarMangaIdCounts = {val[0]: val[1] for val in sorted(similarMangaIdCounts.items(), key=lambda x: (-x[1], x[0]))}
    # next line sorts by frequency but keeps insertion order (aka ties broken by the order of similarity of the user the manga comes from)
    similarMangaIdCounts = dict(sorted(similarMangaIdCounts.items(), key=lambda item: item[1], reverse=True))
    # the idea behind the above line is to sightly reduce the bias for popular manga compared to the line three above this

    similarUsersMangaScores = np.zeros(shape=(len(mangaIds), 2))
    for i in range(len(mangaIds)):
        similarUsersMangaScores[i][0] = mangaIds[i]
        if mangaIds[i] in similarMangaIdCounts:
            similarUsersMangaScores[i][1] = similarMangaIdCounts[mangaIds[i]]
    minValue = similarUsersMangaScores.min(axis=0)[1]
    maxValue = similarUsersMangaScores.max(axis=0)[1]
    for i in range(len(similarUsersMangaScores)):
        similarUsersMangaScores[i][1] = (similarUsersMangaScores[i][1] - minValue) / (maxValue - minValue)

    if showResultsFromEachMethod:
        # filter out manga that dont satisfy filters
        for i in range(len(excludedIds)):
            if excludedIds[i] in similarMangaIdCounts:
                # print(excludedIds[i], similarMangaIdCounts[excludedIds[i]])
                del similarMangaIdCounts[excludedIds[i]]
        # filter out manga the user has already interacted with
        for i in range(len(ratedMangaIds)):
            if ratedMangaIds[i] in similarMangaIdCounts:
                # print(ratedMangaIds[i], similarMangaIdCounts[ratedMangaIds[i]])
                del similarMangaIdCounts[ratedMangaIds[i]]
        print(similarMangaIdCounts)

        similarUsersManga = []
        for mangaId in similarMangaIdCounts.keys():
            # if similarMangaIdCounts[mangaId] == 1:
            #     break  #only include manga up till when there is only 1 user that interacted with that manga (new) (testing)
            similarUsersManga.append(manga[get_manga_array_index[mangaId]])
            if len(similarUsersManga) < 10:
                print(similarMangaIdCounts[mangaId], manga[get_manga_array_index[mangaId]])  # HELPFUL
        print("********************************************************************************")

    mangaImageClusterScores = clusteringResults(convertedRatings, mangaIds, get_manga_array_index, clusterAlgName)
    minValue = mangaImageClusterScores.min(axis=0)[1]
    maxValue = mangaImageClusterScores.max(axis=0)[1]
    for i in range(len(mangaImageClusterScores)):
        mangaImageClusterScores[i][1] = (mangaImageClusterScores[i][1] - minValue) / (maxValue - minValue)

    if showResultsFromEachMethod:
        # get the manga rows that were recommended (and exclude those the user has already rated and that dont satisfy filters)
        imageClusterBasedSorted = mangaImageClusterScores[mangaImageClusterScores[:, 1].argsort()[::-1]]  # sort mangaScores
        filteringTime = time.time()
        similarImageClusterManga = []
        filteredMangaImageClusterScores = []
        for i in range(len(imageClusterBasedSorted)):
            if imageClusterBasedSorted[i][0] not in ratedMangaIds and imageClusterBasedSorted[i][0] not in excludedIds:
                filteredMangaImageClusterScores.append(imageClusterBasedSorted[i])
                similarImageClusterManga.append(manga[get_manga_array_index[int(imageClusterBasedSorted[i][0])]])
        print('\n'.join(["{:.3f}".format(filteredMangaImageClusterScores[[i[0] for i in filteredMangaImageClusterScores].index(similarImageClusterManga[x][0])][1])
                         + "\t" + str(similarImageClusterManga[x]) for x in range(5)]))  # HELPFUL
        print('filtering took:', time.time() - filteringTime)

        print("********************************************************************************")

    mangaImageMatrixScores = matrixResults(convertedRatings, mangaIds, get_manga_array_index, imageFeatureSetName, matrixK)
    # does it matter if we normalize before or after adding filters?
    minValue = mangaImageMatrixScores.min(axis=0)[1]
    maxValue = mangaImageMatrixScores.max(axis=0)[1]
    for i in range(len(mangaImageMatrixScores)):
        mangaImageMatrixScores[i][1] = (mangaImageMatrixScores[i][1]-minValue)/(maxValue-minValue)

    if showResultsFromEachMethod:
        # get the manga rows that were recommended (and exclude those the user has already rated and that dont satisfy filters)
        imageMatrixBasedSorted = mangaImageMatrixScores[mangaImageMatrixScores[:, 1].argsort()[::-1]]  # sort mangaScores
        filteringTime = time.time()
        similarImageMatrixManga = []
        filteredMangaImageMatrixScores = []
        for i in range(len(imageMatrixBasedSorted)):
            if imageMatrixBasedSorted[i][0] not in ratedMangaIds and imageMatrixBasedSorted[i][0] not in excludedIds:
                filteredMangaImageMatrixScores.append(imageMatrixBasedSorted[i])
                similarImageMatrixManga.append(manga[get_manga_array_index[int(imageMatrixBasedSorted[i][0])]])
        print('\n'.join(["{:.3f}".format(filteredMangaImageMatrixScores[
                                 [i[0] for i in filteredMangaImageMatrixScores].index(similarImageMatrixManga[x][0])][
                                 1]) + "\t" + str(similarImageMatrixManga[x]) for x in range(5)]))  # HELPFUL
        print('filtering took:', time.time() - filteringTime)
        print("********************************************************************************")

    mangaScores = np.zeros(shape=(len(mangaEncoded), 2))
    for i in range(len(similarUsersMangaScores)):
        mangaScores[i][0] = mangaIds[i]
        mangaScores[i][1] = (similarityMeasures[i][1] * methodWeights[0]) + (similarUsersMangaScores[i][1] * methodWeights[1]) + \
                            (mangaImageClusterScores[i][1] * methodWeights[2]) + (mangaImageMatrixScores[i][1] * methodWeights[3])
    mangaScores = mangaScores[mangaScores[:, 1].argsort()[::-1]]
    recommendations = []
    filteredMangaScores = []
    for i in range(len(mangaScores)):
        if mangaScores[i][0] not in ratedMangaIds and mangaScores[i][0] not in excludedIds:
            filteredMangaScores.append(mangaScores[i])
            recommendations.append(manga[get_manga_array_index[int(mangaScores[i][0])]])
    # print("combined, content, collaborative, imageClusters, imageMatrix")
    # print('\n'.join(["{:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(
    #     filteredMangaScores[[i[0] for i in filteredMangaScores].index(recommendations[x][0])][1],
    #     similarityMeasures[[i[0] for i in similarityMeasures].index(recommendations[x][0])][1],
    #     similarUsersMangaScores[[i[0] for i in similarUsersMangaScores].index(recommendations[x][0])][1],
    #     mangaImageClusterScores[[i[0] for i in mangaImageClusterScores].index(recommendations[x][0])][1],
    #     mangaImageMatrixScores[[i[0] for i in mangaImageMatrixScores].index(recommendations[x][0])][1])
    #                  + "\t" + str(recommendations[x]) for x in range(5)]))  # HELPFUL

    # return list of json with manga info for the highest scored recommendations
    if callFromNode:
        results = []
        for x in recommendations[:numMangaToReturn]:
            results.append({"id": x[0], "title": x[2], "pictureLink": x[9]})
        return json.dumps(results)
    else:
        if runExperiment:
            evalTime = time.time()
            ks = range(1, 51)  # ########################################
            precision_at_ks = []
            recall_at_ks = []
            numRelevantItems = len(testSet)  # aka true positives + false negatives
            for k in ks:
                # k is also equal to the number of true positives + false positives
                truePositives = 0  # aka number of relevant items recommended at k
                recommendedIds = [x[0] for x in recommendations[:k]]
                for i in range(len(testSet)):
                    if testSet[i] in recommendedIds:
                        truePositives += 1
                precision_at_ks.append(truePositives/k)
                recall_at_ks.append(truePositives/numRelevantItems)

            recommendedIds = [x[0] for x in recommendations[:numMangaToReturn]]
            similarities = []
            #mangaDistanceMatrix = pairwise_distances(mangaEncoded)
            for i in range(len(recommendedIds)-1):
                similarity = []
                for j in range(i+1, len(recommendedIds)):
                    similarity.append(cosineSimilarity(mangaEncoded[get_manga_array_index[recommendedIds[i]]][1:],
                                                       mangaEncoded[get_manga_array_index[recommendedIds[j]]][1:]))
                    #distances.append(mangaDistanceMatrix[get_manga_array_index[recommendedIds[i]]][get_manga_array_index[recommendedIds[j]]])
                similarities.append(np.mean(np.array(similarity)))
            print(similarities)
            diversityValue = 1-np.mean(np.array(similarities))
            print('Diversity:', diversityValue)
            print('evaluation took,', time.time()-evalTime)
            return precision_at_ks, recall_at_ks, diversityValue, recommendedIds
        return recommendations[:numMangaToReturn]


callFromNode = True
runExperiment = False
runMultipleExperiments = False
includeAll = [[1, 27691], [1946, 2022], [1, 6477],
              [False] * 4, [False] * 18, [False] * 51, [False] * 5, [False] * 18, [False] * 51, [False] * 5]
noAdventure = "[[1, 27691],[1946, 2022],[1, 6477],[false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
jsonTestFilter = "[[1,27691],[1946,1999],[1,6477],[false,true,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
jsonTestFilter2 = "[[1, 1000],[2005, 2011],[1, 33],[false,true,false,false],[true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
jsonTestFilter3 = "[[1,27691],[1946,1999],[1,6477],[false,false,false,false],[false,true,false,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
if callFromNode:
    uId = int(sys.argv[1])
    filtersIn = sys.argv[2]
    filtersIn = json.loads(filtersIn)
    try:
        print(recommend(uId, filtersIn))
        #print(recommend(uId, filtersIn, loadMangaFromLocal=True, useLocalRatings=True))
    except Exception as e:
        print(e)
    sys.stdout.flush()
else:
    start_time = time.time()
    #print(recommend(17441, json.loads(noAdventure)))  # me
    print(recommend(1, includeAll, showResultsFromEachMethod=False, loadMangaFromLocal=True, useLocalRatings=True))
    print("total run time:", time.time() - start_time)
