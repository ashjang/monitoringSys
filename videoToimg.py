# 동영상 파일을 이미지로 저장
# 저장되는 이미지에 스켈레톤을 넣는 작업 해야함!!!!!!!!!!

import cv2
import os
from time import sleep


def movToImg(fileName, video):
    print(fileName)
    savePath = os.getcwd() + '/data/병동/1/Thermal/image/'
    os.mkdir(savePath + fileName)
    cnt = 0
    while True:
        success, image = video.read()
        if not(success):
            break
        
        frame = int(video.get(1))
        if frame % 1 == 0:
            title = savePath + fileName + "/%d.jpg" % (frame)
            cv2.imwrite(title, image)
            print(title + " 저장")
            
            cnt += 1
    

def main():
    filePath = os.getcwd() + '/data/병동/1/Thermal/'
    # fileList = glob.glob(filePath)
    
    files = os.listdir(os.getcwd() + '/data/병동/1/Thermal')
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