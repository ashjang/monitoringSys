import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import mpl_toolkits.mplot3d as plt3d
import cv2

def animateJointCoordinates(joint_coordinates, connexion_tuples):

    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = plt.axes(projection='3d')
    
    plt.ion()
    fig.show()
    fig.canvas.draw()
    
    x = 0
    y = 2
    z = 1

    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    for t in range(joint_coordinates.shape[2]):
    # for t in range(1):
        ax.clear()
        
        # Camera coordinate system
        axis_length = 0.2
        
        ax.scatter([0], [0], [0], color="red")
        ax.scatter([axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], marker="v", color="red")
        
        x_axis = plt3d.art3d.Line3D([0, axis_length], [0, 0], [0, 0])
        x_axis.set_color("red")
        y_axis = plt3d.art3d.Line3D([0, 0], [0, axis_length], [0, 0])
        y_axis.set_color("red")
        z_axis = plt3d.art3d.Line3D([0, 0], [0, 0], [0, axis_length])
        z_axis.set_color("red")
        ax.add_line(x_axis)
        ax.add_line(y_axis)
        ax.add_line(z_axis)
        
        # New coordinate system
        x_spine_mid = joint_coordinates[1, x, 0]
        y_spine_mid = joint_coordinates[1, y, 0]
        z_spine_mid = joint_coordinates[1, z, 0]
        
        ax.scatter(x_spine_mid, y_spine_mid, z_spine_mid, color="green")
        ax.scatter([x_spine_mid + axis_length, x_spine_mid, x_spine_mid], 
                   [y_spine_mid, y_spine_mid + axis_length, y_spine_mid], 
                   [z_spine_mid, z_spine_mid, z_spine_mid + axis_length], marker="v", color="green")
        
        x_axis = plt3d.art3d.Line3D([x_spine_mid, x_spine_mid + axis_length], [y_spine_mid, y_spine_mid], [z_spine_mid, z_spine_mid])
        x_axis.set_color("green")
        y_axis = plt3d.art3d.Line3D([x_spine_mid, x_spine_mid], [y_spine_mid, y_spine_mid + axis_length], [z_spine_mid, z_spine_mid])
        y_axis.set_color("green")
        z_axis = plt3d.art3d.Line3D([x_spine_mid, x_spine_mid], [y_spine_mid, y_spine_mid], [z_spine_mid, z_spine_mid + axis_length])
        z_axis.set_color("green")
        ax.add_line(x_axis)
        ax.add_line(y_axis)
        ax.add_line(z_axis)
        
        # Translation vector
        trans_vec = plt3d.art3d.Line3D([0, x_spine_mid], [0, y_spine_mid], [0, z_spine_mid], linestyle='--', color="black")
        ax.add_line(trans_vec)
        
        
        # Subject coordinates
        # ax.set_xlim3d(min(np.amin(joint_coordinates[:, x, :]),-axis_length), max(np.amax(joint_coordinates[:, x, :]), axis_length))
        # ax.set_ylim3d(min(np.amin(joint_coordinates[:, y, :]),-axis_length), max(np.amax(joint_coordinates[:, y, :]), axis_length))
        # ax.set_zlim3d(min(np.amin(joint_coordinates[:, z, :]),-axis_length), max(np.amax(joint_coordinates[:, z, :]), axis_length))
        
        ax.set_xlim3d(min(np.amin(joint_coordinates[:, x, :]),-axis_length)-500, max(np.amax(joint_coordinates[:, x, :]), axis_length))
        ax.set_ylim3d(min(np.amin(joint_coordinates[:, y, :]),-axis_length)-500, max(np.amax(joint_coordinates[:, y, :]), axis_length))
        ax.set_zlim3d(min(np.amin(joint_coordinates[:, z, :]),-axis_length)-500, max(np.amax(joint_coordinates[:, z, :]), axis_length))
        
        for i in range(joint_coordinates.shape[0]):
            x_coord = joint_coordinates[i, x, t]
            y_coord = joint_coordinates[i, y, t]
            z_coord = joint_coordinates[i, z, t]

            ax.scatter(x_coord, y_coord, z_coord, color="blue")
            # ax.text(x_coord, y_coord, z_coord, f'Point {i + 1}')  # 라벨을 점에 표시합니다.
            
        # ax.scatter(joint_coordinates[:, x, t], joint_coordinates[:, y, t], joint_coordinates[:, z, t], color="blue")
        
        
        for i in range(connexion_tuples.shape[0]):
        
            j1 = connexion_tuples[i, 0]
            j2 = connexion_tuples[i, 1]
            
            
            joint_line = plt3d.art3d.Line3D([joint_coordinates[j1, x, t], joint_coordinates[j2, x, t]], 
                                            [joint_coordinates[j1, y, t], joint_coordinates[j2, y, t]], 
                                            [joint_coordinates[j1, z, t], joint_coordinates[j2, z, t]], linestyle=':')
            
            ax.add_line(joint_line)
        
        ax.view_init(10, 50)
        
        ax.invert_zaxis()
        
        plt.ion()
        fig.show()
        fig.canvas.draw()
        
        plt.pause(0.001)
        # plt.savefig("./img/" + str(t) + ".png")
        plt.draw()


sample_name = "S001C001P001R006A001"  

def read_skeleton(file):
    with open(file, 'r') as f:
        shape = eval(f.readline())
        skeleton_sequence = {'numFrame': int(shape[1]), 'frameInfo': []}
    
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': 1, 'bodyInfo': []}
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info['numJoint'] = 25
                body_info['jointInfo'] = []
                f.readline()
                for v in range(25):
                    joint_info_key = [
                        'x', 'y', 'z'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                    # if joint_info != {}:
                    #     body_info['jointInfo'].append(joint_info)
                    # print(body_info['jointInfo'])
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
        
    return skeleton_sequence


def read_xyz(file, max_body=1, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    # print(v)
                    # print("=====================")
                    data[:,n, j, m] = [v['x'], v['y'], v['z']]
                    # print(data)
                    # print("=====================")
                else:
                    pass
    data = np.around(data, decimals=3)
    
    return data


skeleton = read_xyz('./newSkeleton/' + sample_name + ".skeleton")


connexion_tuples = np.array([
    [0,1],
    [1,20],
    [2,3],
    [2,4],
    [4,5],
    [5,6],
    [6,7],
    [2,8],
    [8,9],
    [9,10],
    [10,11],
    [0,12],
    [12,13],
    [13,14],
    [14,15],
    [0,16],
    [16,17],
    [17,18],
    [18,19],
    [20,2],
    [7,21],
    [7,22],
    [11,23],
    [11,24]
])
print(skeleton.transpose(3, 2, 0, 1)[0].shape)

animateJointCoordinates(skeleton.transpose(3, 2, 0, 1)[0], connexion_tuples)