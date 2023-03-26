# 동영상 파일을 이미지로 저장
# 저장되는 이미지에 스켈레톤을 넣는 작업 해야함!!!!!!!!!!

import cv2
import os
from time import sleep
import matplotlib.pyplot as plt
import numpy as np


def movToImg(fileName, video):
    print(fileName)
    savePath = os.getcwd() + '/data/병동/2/RGB_Skeleton/image/'
    os.mkdir(savePath + fileName)
    cnt = 0
    while True:
        success, image = video.read()
        if not(success):
            break
        
        frame = int(video.get(1))
        if frame % 1 == 0:
            start = (100,180)
            end = (530, 900)
            output = np.zeros((end[0]-start[0], end[1]-start[1], 3), np.uint8)

            for y in range(output.shape[1]):
                for x in range(output.shape[0]):
                    xp, yp = x + start[0], y+start[1]
                    output[x,y] = image[xp,yp]
            
            title = savePath + fileName + "/%d.jpg" % (frame)
            cv2.imwrite(title, output)
            print(title + " 저장")
            
            cnt += 1
    

def main():
    filePath = os.getcwd() + '/data/병동/2/RGB_Skeleton/'
    # fileList = glob.glob(filePath)
    
    files = os.listdir(os.getcwd() + '/data/병동/2/RGB_Skeleton')
    # print(files)
    for item in files:
        if item == 'image' or item == '.DS_Store' or item == '.DS':
            continue
        
        video = cv2.VideoCapture(filePath + item)

        # 파일이 존재하지 않을 경우
        if not cv2.VideoCapture.isOpened:
            print("Could not Open:", item)
            exit(0)
        else:
            print(filePath + item)
        
        fileName = item.split('_')
        movToImg(fileName[0], video)
        sleep(0.5)
        
        
main()
