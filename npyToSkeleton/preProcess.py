import os
import numpy as np

'''
    전처리 코드
'''

# missing data에 대해 전처리 (프레임 전체가 0.0이면 바로 이전과 이후의 평균값으로 채움)
def interpolate_skeleton_data(data):
    interpolated_data = np.copy(data)
    
    # 각 관절마다
    for j in range(data.shape[1]):
        # x,y,z 좌표
        x = data[:, j, 0]
        y = data[:, j, 1]
        z = data[:, j, 2]
        
        # 값이 0.0인 것
        zero_indices = np.where((x == 0.0) & (y == 0.0) & (z == 0.0))[0]
        
        # 값이 0.0이 아닌 것
        nonzero_indices = np.where((x != 0.0) | (y != 0.0) | (z != 0.0))[0]
        
        # linear interpolation(선형 보간법)
        for idx in zero_indices:
            # 한 프레임 전체가 0이면, 이전 프레임에서 끌고 옴
            if len(nonzero_indices) == 0:
                interpolated_data[idx, j] = interpolated_data[idx - 1, j]
            else:
                # 가장 가까운 0의 값 찾음
                closest_indices = np.abs(nonzero_indices - idx)
                closest_indices = closest_indices.argsort()[:2]
                
                closest_values = interpolated_data[nonzero_indices[closest_indices], j]
                
                # linear interpolation
                interpolated_value = np.mean(closest_values, axis=0)
                interpolated_data[idx, j] = interpolated_value
    
    return interpolated_data


# 정규화 (평균: 0, 표준편차: 1)
def normalize_skeleton_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

# 데이터 부드럽게 함 (이동 평균 이용)
def smooth_skeleton_data(data, window_size=5):
    smoothed_data = np.zeros_like(data)
    num_joints = data.shape[1]
    
    for i in range(num_joints):
        for j in range(3):  # x,y,z
            smoothed_data[:, i, j] = np.convolve(data[:, i, j], np.ones(window_size), mode='same') / window_size
    
    return smoothed_data

# 전처리
def preprocess_skeleton_data(skeleton_data):
    handle_missing_data = interpolate_skeleton_data(skeleton_data)
    # normalize_data = normalize_skeleton_data(handle_missing_data)
    # smooth_data = smooth_skeleton_data(normalize_data)
    # return smooth_data
    return handle_missing_data

# 만약 numpy에 날짜가 포함되어 있다면
def convert_to_skeleton(frames):
    skeleton_frames = []
    for frame in frames:
        skeleton = np.array(frame[1:])  # Exclude the timestamp
        # print(skeleton)
        frameArr = []
        for joint in skeleton:
            jointToNp = np.array(joint)
            frameArr.append(jointToNp)
        skeleton_frames.append(np.array(frameArr))

    return np.array(skeleton_frames)

if __name__ == "__main__":
    skeleton_folder = "./Skeleton/"         # 수정해야함
    preProcess_folder = "./preProcessed/"   # 수정해야함

    dictOfTypeIdx = {0:1, 1:2, 2:4, 3:27, 4:6,
                    5:7, 6:8, 7:9, 8:13, 9:14, 10:15,
                    11:16, 12:19, 13:20, 14:21, 15:22,
                    16:23, 17:24, 18:25, 19:26, 20:3,
                    21:10, 22:11, 23:17, 24:18}

    for root, dirs, files in os.walk(skeleton_folder):
        for file in files:
            if file.endswith('.npy'):
                data = np.load(skeleton_folder + file, allow_pickle=True)
                try:
                    preprocessed_data = preprocess_skeleton_data(data.astype(float))
                except ValueError:
                    data = convert_to_skeleton(data)
                    preprocessed_data = preprocess_skeleton_data(data.astype(float))
                    
                listOfData = preprocessed_data.tolist()
                
                listOfResult = [0 for _ in range(25)]
                
                with open(preProcess_folder + file.split(".")[0] + ".skeleton", 'w') as f:
                    f.write(str(len(listOfData)) + "\n\n")
                    # f.write(str(preprocessed_data.shape) + "\n\n")      # shape
                    
                    for i in range(len(listOfData)):
                        idx = 0
                        # frame: 0부터 시작 ++, 사람수: 무조건 1로 둠

                        for j in range(len(listOfData[i])):
                            listOfResult[idx] = listOfData[i][j]
                            idx += 1
                                    
                        for j in range(25):
                            temp = str(listOfResult[j])[1:-1]
                            temp = temp.replace(',', '')
                            f.write(temp + "\n")
                            
                        listOfResult = [0 for _ in range(25)]
                        f.write("\n")
                        
                    f.close()
    print("데이터 변환 완료하였습니다.")


