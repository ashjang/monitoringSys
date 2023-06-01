r"""
Contains helper functions to extract skeleton data from the NTU RGB+D dataset.

Three functions are provided.

    - *read_skeleton*: Parses entire skeleton file and outputs skeleton data in a dictionary
    - *read_xyz*: Only keeps 3D coordinates from dictionary and returns numpy version.
    - *read_xy_ir*: Only keeps 2D IR coordinates from dictionary and returns numpy version.

"""

import numpy as np


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': 1, 'bodyInfo': []}
            # print(frame_info['numBody'])
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
                    if joint_info != {}:
                        body_info['jointInfo'].append(joint_info)
                    # print(body_info['jointInfo'])
                frame_info['bodyInfo'].append(body_info)
                # print(body_info)
                # print(frame_info['bodyInfo'])
            skeleton_sequence['frameInfo'].append(frame_info)
    # print(skeleton_sequence)
    return skeleton_sequence


def read_xyz(file, max_body=1, num_joint=25):
    r"""Creates a numpy array containing the 3D skeleton data for a given skeleton file of the NTU RGB+D dataset.
    This code is slightly modified and is courtesy of the awesome ST-GCN repository by yysijie
    (https://github.com/yysijie/st-gcn/)
    Inputs:
        - **file** (str): Complete path to the skeleton file.
        - **max_body** (int): Maximum number of subjects (2 for NTU RGB+D)
        - **numb_joints** (int): Maximum number of joints (25 for Kinect v2)
    Outputs:
        **data (np array)**: Numpy array containing skeleton
        of shape `(3 {x, y, z}, max_frame, num_joint, 2 {n_subjects})`
    """
    seq_info = read_skeleton(file)
    # print(seq_info)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
    # print(data)
    # print(seq_info['frameInfo'])
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    # print(v['x'], v['y'], v['z'])
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    data = np.around(data, decimals=3)
    # print(data)

    return data



def read_xy_ir(file, max_body=2, num_joint=25):
    r"""Creates a numpy array containing the 2D skeleton data projected on the IR frames
    for a given skeleton file of the NTU RGB+D dataset.
    This code is slightly modified and is courtesy of the awesome ST-GCN repository by yysijie
    (https://github.com/yysijie/st-gcn/)

    Inputs:
        - **file** (str): Complete path to the skeleton file.
        - **max_body** (int): Maximum number of subjects (2 for NTU RGB+D)
        - **numb_joints** (int): Maximum number of joints (25 for Kinect v2)

    Outputs:
        **data (np array)**: Numpy array containing skeleton
        of shape `(2 {x, y}, max_frame, num_joint, 2 {n_subjects})`

    """
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)

    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['depthX'], v['depthY']]
                else:
                    pass
    return data


np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
fileName = './S001C001P001R001A015.skeleton'
data = read_xyz(fileName)

print(data)
# print(data['frameInfo'][0])