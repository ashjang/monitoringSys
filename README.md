# 프라이버시를 고려한 환자 및 1 인 가구의실내 위험 모니터링 시스템 개발

## 연구 목표
- 1인 가구와 환자를 대상으로 한 위험 감지, 행동 분류 모델
- RGB 카메라보다 프라이버시 침해도가 낮은 센서를 사용하는 시스템
- 신체에 부착하지 않는 원격 센서를 사용하는 시스템

<br />

## ✔ 연구 배경
 - 통계청 자료에 의하면, 1인 가구의 증가가 급속히 진행
 - 두통, 구토 등의 전조증상을 보이는 뇌출혈과 기침, 가슴 통증, 비틀거림의 증상을 보이는 심장마비와 같이, 질병이 발생하기 전에 나타나틑 다양한 (전조) 증상
 - 기존의 낙상 감지는 RGB 영상/동영상 또는 웨어러블 기기 사용하는 경우가 많음
 - RGB 영상의 프라이버시 침해 문제
 - 웨어러블 기기 => 주기적으로 충전이 필요하고 이로 인한 장시간 기기 미착용 문제

<br />

## ✔ 사용한 센서
<img src="./images/%EC%84%BC%EC%84%9C%20%ED%8A%B9%EC%A7%95.JPG">

 - Azure Kinect
    - Depth, IR, RGB 카메라 내장
    - 3D Body Tracking (Depth 이미지 이용하여 신체 추적및 실시간 3D Skeleton 추출)

<br />

## ✔ 선정된 건강 이상 / 비이상 행동
<img src="./images/%EC%84%A0%EC%A0%95%EC%9E%90%EC%84%B8.JPG">

## ✔ 데이터 수집 시나리오
1. 침대에 걸터 앉아 있다가 낙상 - 6회
2. 걷다가 낙상 - 6회
3. 두통 - 오른손/왼손/양손 각 2회 (총 6회)
4. 가슴 통증 - 오른손/왼손/양손 각 2회 (총 6회)
5. 복통 - 오른손/왼손/양손 각 2회 (총 6회)
6. 허리 통증 - 오른손/왼손/양손 각 2회 (총 6회)
7. 기침 - 오른손/왼손/양손 각 2회 (총 6회)
8. 구토 - 6회
9. 부채질 - 6회
10. 비틀거림 - 6회
11. 절뚝거림 - 6회
12. 걷기 - 6회
13. 걷다가 앉기 - 6회
14. 앉아있다 일어나기 - 6회
15. 기지개 켜기 - 6회
16. 물마시기 - 6회
17. 휴대폰하기 - 6회
(1~11번은 의학적 이상 행동, 12~17번은 일상(비이상) 행동)

<br />

## ✔ 전처리 과정
<img src="./images/%EC%A0%84%EC%B2%98%EB%A6%AC%EA%B3%BC%EC%A0%95.JPG">

<br />

## ✔ 시스템 개요도
<img src="./images/%EC%8B%9C%EC%8A%A4%ED%85%9C%20%EA%B0%9C%EC%9A%94%EB%8F%84.JPG">

 - 보편적인 Feature를 미리 학습한 모델을 사용하는 전이 학습 방식 사용
    - ResNet-18
    - R(2+1)D-18
 - 이종 데이터를 입력으로 받아 분류하는 Fusion 개념 적용

 <br />

 ## ✔ 참고 문헌
 [1] A. M. De Boissiere and R. Noumeir, “Infrared and 3D Skeleton Feature
Fusion for RGB-D Action Recognition”, Journal of IEEE Access, Vol. 8, pp.
168297-168308, Sep. 2020.

[2] “NTU RGB+D“ Dataset, ROSE LAB, Available:
https://rose1.ntu.edu.sg/dataset/actionRecognition/

[3] FUSION-human-action-recognition, Available:
https://github.com/adeboissiere/FUSION-human-action-recognition

[4] HDF5, Available: https://docs.h5py.org/en/stable/

[5] Azure Kinect DK, Available:
https://azure.microsoft.com/en-us/products/kinect-dk


