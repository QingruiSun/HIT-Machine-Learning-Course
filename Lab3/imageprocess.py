import numpy as np
import cv2
from pcastudy import centralization , get_projection_matrix

def calculate_snr(rawdata,newdata):
    suma = 0
    sumb = 0
    for i in range(50):
        for j in range(50):
            suma = suma + rawdata[i][j]**2
            sumb = sumb + (rawdata[i][j] - newdata[i][j])**2
    snr_value = 10 * np.log10(suma/sumb)
    return  snr_value


image = []
imagenumber = 10
for i in range(1,imagenumber+1):
    imagename = str(i) + '.png'
    image.append(imagename)
imageobject = []
for i in range(imagenumber):
    imageobject.append(cv2.imread(image[i]))
    imageobject[i] = cv2.cvtColor(imageobject[i],cv2.COLOR_BGR2GRAY)
    imageobject[i] = cv2.resize(imageobject[i],(50,50))
dataset = []
#将10个50*50的图像表示成10*2500的juzhen,每个图像是一个2500维的数据
for i in range(imagenumber):
    tempdataset = []
    for j in range(50):
        for k in range(50):
            tempdataset.append(imageobject[i][j][k])
    dataset.append(tempdataset)
dataset,meanvalue = centralization(np.array(dataset))
w = get_projection_matrix(np.transpose(dataset),5)
w = np.array(w)
#对降维后的数据进行重构
newdataset = np.array(dataset[0]).dot(np.transpose(w))
newdataset = newdataset.dot(w)
imagedata = []
for i in range(50):
    tempdata = []
    for j in range(50):
        tempdata.append(newdataset[i*50+j] + meanvalue[i*50 + j])
    imagedata.append(tempdata)
snr_value = calculate_snr(imageobject[0],imagedata)
print("snrvalue",snr_value)
cv2.namedWindow("ChenDuling")
cv2.resizeWindow("ChenDuling",500,500)
for i in range(0,50):
    for j in range(0,50):
        imageobject[0][i][j] = imagedata[i][j]
cv2.imshow("ChenDuling",imageobject[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
for i in range(0,10):
    grayimage.append(cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY))
    grayimage[i] = cv2.resize(grayimage[i], (50,50), interpolation=cv2.INTER_CUBIC)
for i in range(0,10):
    temp = []
    for j in range(0,50):
        for k in range(0,50):
            temp.append(grayimage[i][j][k])
    digitalimage.append(np.array(temp))
digitalimage = np.array(digitalimage)
cen_digitalimage = centralization(digitalimage)
project_matrix = get_projection_matrix(np.transpose(cen_digitalimage),100)
new_digitalimage = project_matrix.dot(np.transpose(digitalimage))
print(new_digitalimage.shape)
new_digitalimage = np.transpose(new_digitalimage)
tempimage = new_digitalimage[0]
newtemp = cv2.resize(tempimage, (50,50), interpolation=cv2.INTER_CUBIC)
print(newtemp.shape)
cv2.namedWindow("ChenDuling")

cv2.resizeWindow("ChenDuling",500,500)
cv2.imshow("ChenDuling",newtemp)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




"""
matrix_w = get_projection_matrix(digitalimage1,100)
matrix_w = np.array(matrix_w)
newimage1 = matrix_w.dot(digitalimage1)
cv2.imshow("newimage1",newimage1)
cv2.waitKey(0)
cv2.destroyAllWindows()""
"""