import os
import natsort


# 폴더 내 파일들 불러오고 날짜별로(이름순) 정렬
filePath = os.getcwd() + '/test/mmWave'
files = os.listdir(os.getcwd() + '/test/mmWave')
files = natsort.natsorted(files)
# print(files)


# 필요에따라 변경
s = 1
c = 1
p = 1

# 고정 변수
r = 1
a = 1

i = 0
while i != (len(files)):
    # print(files[i])
    if r == 7 and a != 7:
        r = 1
        a += 1
        continue
    if r == 9 and a == 7:
        r = 1
        a += 1
        continue
    # 변경할 이름
    newName = "S{0:03d}C{1:03d}P{2:03d}R{3:03d}A{4:03d}.npy".format(s,c,p,r,a)
    # print(newName)
    
    # 변경하기
    oldFile = os.path.join(filePath, files[i])
    newFile = os.path.join(filePath, newName)
    os.rename(oldFile, newFile)
    
    # 다음 파일
    r += 1
    i += 1
    # print()


    
