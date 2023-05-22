import os
import numpy as np
import pickle


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': 1, 'bodyInfo': []}
            for m in range(frame_info['numBody']):
                f.readline()
                body_info = {'jointInfo': []}
                for v in range(25):
                    joint_info_key = [
                        'x', 'y', 'z'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence



def preProcessing(file):
    seq_info = read_skeleton(file)
   
    max_body = 2
    num_joint = 25
    frame_info = seq_info['frameInfo']
    # print(len(seq_info['frameInfo']))

    # # data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
    # # print(data.shape)
    copy = seq_info

    idx = -1
    # data = [] * len(seq_info['frameInfo'])
    data = []
   
    for n, f in enumerate(copy['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            print(b)
            print('\n')
            tmp = b['jointInfo']
            if(n<4):
                data.append(tmp)
                idx +=1
            else:
                for j, v in enumerate(b['jointInfo']):
                    # print(v)
                    # print('\n')
                    if(j==0):
                        if(v['x'] != 0 and v['y'] != 0 and v['z'] != 0):
                            data.append(tmp)
                            idx +=1
                        else:
                            if(idx > -1):
                                prev = data[idx]
                                data.append(prev)
                                idx+=1
        # print(n)
        # print('\n')

    # print("\n\n\n")
    # print(data)


    # 전처리한 스켈레톤 저장
    processed_folder = './preProcessed'
    file_name = file.split('.npy')[0] + ".skeleton"
    # print(file_name)
    # with open(processed_folder/file_name, 'w' ...)

    with open('./S001C001P001R001A002_result.skeleton', 'w', encoding='utf-8') as f:
        f.write(str(len(seq_info['frameInfo'])) + "\n\n")
        for n, k in enumerate(data):
            for m, b in enumerate(k):
                # print(b)
                f.write(str(b['x']) + " ")
                f.write(str(b['y']) + " ") 
                f.write(str(b['z']) + " ")
                f.write('\n')
            f.write('\n')
            



folder_path = './Skeleton'
file_list = os.listdir(folder_path)

# for i in file_list:
#     print(i)

    # for file in file_list: 
preProcessing(file = './S001C001P001R001A002_test.skeleton')
    # preProcessing(file)