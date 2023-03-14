# monitoringSys #
프라이버시를 고려한 환자 및 1인 가구의 위험 실내 모니터링 시스템 개발

# 착수보고서 #
[2022후기_착수보고서_08_D_프라이버시를 고려한 실내 위험 모니터링 시스템.pdf](https://github.com/ashjang/monitoringSys/files/10928674/2022._._08_D_.pdf)


#
# < 회의록 > #
## 03.09 ##
1. 공개 데이터셋 (NTU RGB+D 60) https://rose1.ntu.edu.sg/dataset/actionRecognition/
2. HAR recognition 모델 https://paperswithcode.com/dataset/ntu-rgb-d
3. OpenCV 동영상 저장 방법 
4. 공개 데이터셋 구성  확인
5. 파일명 분류 
6. Skeleton input shape 맞추기


## 3.10 회의록 ##
### 시나리오 ###
************ Side view 침상 밖 시나리오 ************  

1. 침대에 걸터 앉아 있다가 낙상 6회 실시
2. 걷다가 낙상 6회
3. 두통 (오른손 왼손 양손) 각 2회
4. 가슴 통증 (오른손 왼손 양손) 각 2회
5. 복통 (오른손 왼손 양손) 각 2회
6. 허리 통증 (오른손 왼손 양손) 각 2회
7. 기침 (오른손으로 입을 막고, 왼손으로 입을 막고, 양손, 손 없이) 각 2회
8. 구토 6회
9. 부채질 6회
10. 비틀거림 6회
11. 절뚝거림 6회
12. 일상행동 자세 - 그냥 걸어다니기(4~6초) 6회
13. 일상행동 자세 - 앉기 (2초정도 걷다가 침대에 앉기) 6회
14. 일상행동 자세 - 앉아있다 일어나기 (침대에 앉아있다가 일어나기) 6회
15. 일상행동 자세 - 기지개 켜기 6회
16. 일상행동 자세 - 물마시기 6회
17. 일상행동 자세 - 휴대폰하기 6회

 =>>>> NTU RGB+D 60 Dataset 데이터 유형과 비슷한 환경으로 조성
 why? 많은 정보가 있음으로 빠르게 학습을 진행해볼 수 있음
 
 SsssCcccPpppRrrrAaaa_카메라종류.avi 혹은 skeleton(numpy)
 ex) S018C001P045R002A097_ir.avi
 18번째의
 
### 파일 형식 ###
-> S001 C001 P001 R001 A001

SsssPpppRrrrAaaa

S : Setup number (상황별 넘버 1: 리빙랩 / 2 : 연구실 / 3: 세미나실)

C : Camera number (camera 1번)

P : Performer id (사람 ID)

R : replication number (반복 횟수)

A : Action lable (동작 레이블 1~?)


Skeleton data

원래 [x,y,z]
-> shape (3, frame, 25, 사람수 2)

(3 {x, y, z}, max_frame, num_joint, 2 {n_subjects})

-> 3D Skeleton numpy 데이터 (25 body joints, 3 - x,y,z, frame 수만큼)



### ---------------------데이터 영상 수집시 주의점--------------------- ###
1. Skeleton data가 확실하게 나오게끔 수집
2. mmWave Point cloud가 잘 나오게끔 동작 크게크게
3. 3~8s의 길이 최대한 데이터
4. Thermal 이미지 안짤리게

### ---------------------개선 필요한점 230309--------------------- ###
1. IR 동영상 저장
2. 파일형식 대로 저장 - 여러분
3. Skeleton input shape 맞추기 - 여러분
4. mmWave x,y,z, 백터만 나오게끔




# H/W kinect 사용법 및 설치법
https://learn.microsoft.com/ko-kr/azure/kinect-dk/ 

Kinect microsoft 공홈 사용법 

https://github.com/ibaiGorordo/pyKinectAzure

C로 구성된 sdk를 python으로 구동하기 위해서 위의 git을 참조하여 구동

-> 공홈은 참조만 하고 위의 모듈 다운로드 및 첫 구동해보기

error 뜨는 것들은 모듈 하나씩 설치해보기 


