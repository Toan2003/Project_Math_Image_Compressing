import numpy as np
import matplotlib.pyplot as plt
import PIL as P
import random
import math

def Input(path):
    picture = P.Image.open(path)
    picture = np.array(picture)
    return picture

def Output(picture):
    plt.imshow(np.array(picture))
    plt.show()

def kmeansInitCentroid(init_centroids, k_clusters, array):
    if init_centroids == 'random':
        centers = []
        for i in range(k_clusters):
            centers.append([random.randint(0,255) for i in range(3)])
        return centers
    else:
        return [np.array(array[random.randint(0,len(array))],dtype=int) for i in range(k_clusters)]

def labelElements(array, centroids):
    distanceToCentroids = 500
    index = -1
    labels = np.zeros(len(array),dtype=int)
    for i in range(len(array)):
        for j in range(len(centroids)):
            temp = kmeansCaculateDistance(array[i],centroids[j])
            if distanceToCentroids > temp: 
                distanceToCentroids = temp
                index = j
        labels[i] = index
        distanceToCentroids = 500
        index = -1
    return labels

def updateCentroids(labels, array, k_cluster):
    centroids = np.zeros((k_cluster,3),dtype=int)
    num = np.zeros(k_cluster,dtype=int)
    for i in range(len(labels)):
        centroids[labels[i]] = np.add(array[i],centroids[labels[i]])
        num[labels[i]] += 1
    # print(num)
    # print(centroids)
    # input()
    for i in range(len(centroids)):
        if num[i] != 0:
            centroids[i] = centroids[i] / num[i]
    # print(centroids)
    return centroids

def kmeansCaculateDistance(X,Centroid):
    return math.sqrt( pow(X[0]-Centroid[0],2) + pow(X[1]-Centroid[1],2) + pow(X[2]-Centroid[2],2))

def isTheSameCentroids(center1, center2):
    arrayResult = []
    result = True
    for i in range(len(center1)):
        arrayResult.append(kmeansCaculateDistance(center1[i],center2[i]))
        if arrayResult[i] > 1:
            result = False
            return result
    return result

def returnLabels(labels, centroids):
    result = []
    for i in labels:
        result.extend(centroids[i])
    return result

def kmeans(img_1d, k_clusters, max_iter, init_centroids='random'):
    array = np.reshape(img_1d,(-1,3))
    labels = []
    centers = kmeansInitCentroid(init_centroids,k_clusters,array)
    print('initial centroids: ')
    print(centers)
    iter = 1
    while True:
        print('lần lặp: ',iter)
        if iter == max_iter: break
        labels = labelElements(array,centers)
        tempCenters = updateCentroids(labels,array, k_clusters)
        if isTheSameCentroids(centers, tempCenters):
            break
        else:
            centers = tempCenters
        iter += 1
    return centers, np.reshape(returnLabels(labels,centers), (len(img_1d),len(img_1d[0]),3))

def Interact():
    picture = input('Nhập path đến hình ảnh của bạn/hoặc tên ảnh: ')
    clusters = input('Nhập k clusters mong muốn (số lượng centroids): ')
    type = input('Loại nén: nhập random với chế độ random, inpixel với chế độ inpixel (mặc định): ')
    fileSave = input('Nhập tên file output (example: anhdep): ')
    typeOutput = input('Nhập định dạng file muốn lưu: 1 cho .png (mặc định), 2 cho .pdf: ')
    if typeOutput == 2:
        fileSave = fileSave + '.pdf'
    else:
        fileSave = fileSave + '.png'
    return picture, clusters, type, fileSave

def saveFile(fileSave, array):
    plt.imsave(fileSave, np.array(array,dtype= 'uint8'))

if __name__ == '__main__':
    picture, cluster, type, fileSave = Interact()
    array = Input(picture)
    
    # picture, cluster, type, fileSave = 'test.jpg',7,'inpixel','anhdep.pdf'
    # array = Input(picture)

    centers , labels = kmeans(array, int(cluster),100,type)
    print("centroids after kmeans: ")
    print(centers)
    saveFile(fileSave,labels)
    Output(labels)

    