import cv2
import sys

sys.path.append('.')
sys.path.append('..')
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QCoreApplication
import pykinect_azure as pykinect
import qimage2ndarray
from datetime import datetime

RGB_IMAGE_PATH      = 'C:\\ti\\mmwave_industrial_toolbox_4_12_1\\tools\\Visualizer\\datas\\RGB\\'
IR_IMAGE_PATH       = 'C:\\ti\\mmwave_industrial_toolbox_4_12_1\\tools\\Visualizer\\datas\\IR\\'
DEPTH_IMAGE_PATH    = 'C:\\ti\\mmwave_industrial_toolbox_4_12_1\\tools\\Visualizer\\datas\\Depth\\'
THERMAL_IMAGE_PATH  = 'C:\\ti\\mmwave_industrial_toolbox_4_12_1\\tools\\Visualizer\\datas\\Thermal\\'
RGB_SKELETON_PATH  = 'C:\\ti\\mmwave_industrial_toolbox_4_12_1\\tools\\Visualizer\\datas\\RGB_Skeleton\\'
SKELETON_DATA_PATH  = 'C:\\ti\\mmwave_industrial_toolbox_4_12_1\\tools\\Visualizer\\datas\\Skeleton\\'

def cv2_to_qimage(cv_img):
    height, width, bytesPerComponent = cv_img.shape
    bgra = np.zeros([height, width, 4], dtype=np.uint8)
    bgra[:, :, 0:3] = cv_img
    return QtGui.QImage(bgra.data, width, height, QtGui.QImage.Format_RGB32)

# ------------------ Azure Kinect Setting ------------------ #
# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries(track_body=True)

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
# ------------------ Azure Kinect Setting ------------------ #

class ShowVideo(QtCore.QObject):
	VideoSignal_RGB     = QtCore.pyqtSignal(QtGui.QImage)
	VideoSignal_Depth   = QtCore.pyqtSignal(QtGui.QImage)
	VideoSignal_IR      = QtCore.pyqtSignal(QtGui.QImage)
	VideoSignal_THERMAL = QtCore.pyqtSignal(QtGui.QImage)


	def __init__(self, parent=None):
		super(ShowVideo, self).__init__(parent)
		self.device = pykinect.start_device(config=device_config)   # Start device
		self.bodyTracker = pykinect.start_body_tracker()            # Start body tracker
		self.thermal_vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		self.thermal_vid.set(cv2.CAP_PROP_CONVERT_RGB,0)
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		self.list_save = []
		self.record_flag = False
		self.quit_flag = False
		self.image_count = 0
		self.break_flag = 1
		iswrite = True
    
	@QtCore.pyqtSlot()
	def startVideo(self):
		# video save 
		run_video = True
		rgb_out = cv2.VideoWriter(RGB_IMAGE_PATH + 'RGB.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 15, (1920, 1080), True)
		depth_out = cv2.VideoWriter(DEPTH_IMAGE_PATH + 'Depth.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 10, (512, 512), True)
		ir_out = cv2.VideoWriter(IR_IMAGE_PATH + 'IR.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 10, (512, 512), True)
		thermal_out = cv2.VideoWriter(THERMAL_IMAGE_PATH + 'Thermal.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 10, (512, 512), True)
		rgb_skeleton_out = cv2.VideoWriter(RGB_SKELETON_PATH + 'RGB_Skeleton.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1032, 580), True)
        
		while run_video:
			self.datetime = str(datetime.now())
			self.capture = self.device.update()
			self.body_frame = self.bodyTracker.update()
			self.point_bodies = self.body_frame.get_body()
			ret, RGB_image      = self.capture.get_color_image()
			ret, Depth_image    = self.capture.get_colored_depth_image()
			ret, IR_image       = self.capture.get_ir_image()
			ret, Thermal_image  = self.thermal_vid.read()

			if self.record_flag == True : 
				rgb_out.write(RGB_image)
			elif self.record_flag == False:
				self.image_count = 0
                
            # Draw the skeletons into the color image
			RGB_skeleton = self.body_frame.draw_bodies(RGB_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
			RGB_skeleton = cv2.resize(RGB_skeleton, dsize=(1032, 580), interpolation=cv2.INTER_LINEAR)

            # Thermal processing
			Thermal_image = cv2.resize(Thermal_image, dsize=(512,512), interpolation=cv2.INTER_LINEAR)
			Thermal_image = cv2.normalize(Thermal_image, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
			np.right_shift(Thermal_image, 8, Thermal_image)
			Thermal_image = Thermal_image.astype(np.uint8)
			Thermal_image = cv2.cvtColor(Thermal_image, cv2.COLOR_GRAY2BGR)
			Thermal_image = cv2.applyColorMap(Thermal_image, cv2.COLORMAP_INFERNO)
            
			if self.record_flag == True : 
				self.save_npy(self.datetime)
				depth_out.write(Depth_image)
				ir_out.write(IR_image)
				thermal_out.write(Thermal_image)
				rgb_skeleton_out.write(RGB_skeleton)
				iswrite = True

			if self.quit_flag == True:
				if iswrite == True :
        			############ 수정
					# np.save(SKELETON_DATA_PATH + str(datetime.now().strftime('%Y-%m-%d %H%M%S')) + '.npy', np.array(self.list_save))
					# """
     				self.npToSkeleton()
					skeleton = self.read_xyz(SKELETON_DATA_PATH + "np.skeleton")
					f = open(SKELETON_DATA_PATH + formatted_name + '.skeleton', 'w')
					f.write(skeleton.shape, "\n")
					f.write(skeleton[0,0,0,:] + "\n")
     				f.write("===== 3D skeleton =====\n")
					f.write(skeleton[0, 0, :, 0], "\n\n")
					f.write(skeleton[0, :, 0, 0], "\n\n")
					f.write(skeleton[:, 0, 0, 0], "\n\n")
					f.write(skeleton[:, 0, :, 0], "\n\n")
					f.write(skeleton.transpose(3, 2, 0, 1)[0].shape, "\n")
					f.close()
					# """
					
					break

            #     self.image_count = self.image_count + 1                
            #     self.image_count = 0

                
			RGB_skeleton = cv2_to_qimage(RGB_skeleton)
			Depth_image  = qimage2ndarray.array2qimage(Depth_image, normalize=False)
			IR_image     = qimage2ndarray.array2qimage(IR_image, normalize=False)
			Thermal_image= qimage2ndarray.array2qimage(Thermal_image, normalize=False)

			qt_image_RGB = QtGui.QImage(RGB_skeleton)
			qt_image_Depth = QtGui.QImage(Depth_image)
			qt_image_IR = QtGui.QImage(IR_image)
			qt_image_Thermal = QtGui.QImage(Thermal_image)

			self.VideoSignal_RGB.emit(qt_image_RGB)
			self.VideoSignal_Depth.emit(qt_image_Depth)
			self.VideoSignal_IR.emit(qt_image_IR)
			self.VideoSignal_THERMAL.emit(qt_image_Thermal)

			loop = QtCore.QEventLoop()
			QtCore.QTimer.singleShot(25, loop.quit) #25 ms
			loop.exec_()


	def record_button(self):
		if self.record_flag == False : 
			self.record_flag = True
			push_button1.setText('Finish')

		elif self.record_flag == True : 
			self.record_flag = False
			push_button1.setText('Record')
			# np.save(SKELETON_DATA_PATH + str(datetime.now().strftime('%Y-%m-%d %H%M%S')) + '.npy', np.array(self.list_save))
			self.image_count = 0
			# self.list_save = []
            
	def quit_button(self):
		if self.quit_flag == False :
			self.quit_flag = True
		elif self.quit_flag == True :
			# np.save(SKELETON_DATA_PATH + str(datetime.now().strftime('%Y-%m-%d %H%M%S')) + '.npy', np.array(self.list_save))
			self.image_count = 0
			self.list_save = []
			self.quit_flag = False

   
			# quit_button.clicked.connect(QCoreApplication.instance().quit)
			# quit_button.clicked.connect(QCoreApplication.instance().quit)


	def save_npy(self, datetime=str):
		# Skeleton data save
		if True : 
			# 골반
			pelvis_pos_x = float(self.point_bodies.joints[0].position.x)
			pelvis_pos_y = float(self.point_bodies.joints[0].position.y)
			pelvis_pos_z = float(self.point_bodies.joints[0].position.z)
            
            # 배꼽
			navel_pos_x = float(self.point_bodies.joints[1].position.x)
			navel_pos_y = float(self.point_bodies.joints[1].position.y)
			navel_pos_z = float(self.point_bodies.joints[1].position.z)

			# 가슴
			chest_pos_x = float(self.point_bodies.joints[2].position.x)
			chest_pos_y = float(self.point_bodies.joints[2].position.y)
			chest_pos_z = float(self.point_bodies.joints[2].position.z)
            # # 3 neck
			neck_pos_x = float(self.point_bodies.joints[3].position.x)
			neck_pos_y = float(self.point_bodies.joints[3].position.y)
			neck_pos_z = float(self.point_bodies.joints[3].position.z)
            # # 4 left_clavicle (쇄골)
			# left_clavicle_pos_x = float(self.point_bodies.joints[4].position.x)
			# left_clavicle_pos_y = float(self.point_bodies.joints[4].position.y)
			# left_clavicle_pos_z = float(self.point_bodies.joints[4].position.z)
            # # 5 left_shoulder (어깨)
			left_shoulder_pos_x = float(self.point_bodies.joints[5].position.x)
			left_shoulder_pos_y = float(self.point_bodies.joints[5].position.y)
			left_shoulder_pos_z = float(self.point_bodies.joints[5].position.z)
            # 6 left_elbow (팔꿈치)
			left_elbow_pos_x = float(self.point_bodies.joints[6].position.x)
			left_elbow_pos_y = float(self.point_bodies.joints[6].position.y)
			left_elbow_pos_z = float(self.point_bodies.joints[6].position.z)
            # 7 left_wrist (손목)
			left_wrist_pos_x = float(self.point_bodies.joints[7].position.x)
			left_wrist_pos_y = float(self.point_bodies.joints[7].position.y)
			left_wrist_pos_z = float(self.point_bodies.joints[7].position.z)		
            # 8 left_hand (손)
			left_hand_pos_x = float(self.point_bodies.joints[8].position.x)
			left_hand_pos_y = float(self.point_bodies.joints[8].position.y)
			left_hand_pos_z = float(self.point_bodies.joints[8].position.z)		
            # 9 left_handtip
			left_handtip_pos_x = float(self.point_bodies.joints[9].position.x)
			left_handtip_pos_y = float(self.point_bodies.joints[9].position.y)
			left_handtip_pos_z = float(self.point_bodies.joints[9].position.z)		
            # 10 left_thumb (엄지)
			left_thumb_pos_x = float(self.point_bodies.joints[10].position.x)
			left_thumb_pos_y = float(self.point_bodies.joints[10].position.y)
			left_thumb_pos_z = float(self.point_bodies.joints[10].position.z)		
            # 11 right_clavicle (쇄골)
			# right_clavicle_pos_x = float(self.point_bodies.joints[11].position.x)
			# right_clavicle_pos_y = float(self.point_bodies.joints[11].position.y)
			# right_clavicle_pos_z = float(self.point_bodies.joints[11].position.z)		
            # 12 right_shoulder (어깨)
			right_shoulder_pos_x = float(self.point_bodies.joints[12].position.x)
			right_shoulder_pos_y = float(self.point_bodies.joints[12].position.y)
			right_shoulder_pos_z = float(self.point_bodies.joints[12].position.z)		
            # 13 right_elbow (팔꿈치)
			right_elbow_pos_x = float(self.point_bodies.joints[13].position.x)
			right_elbow_pos_y = float(self.point_bodies.joints[13].position.y)
			right_elbow_pos_z = float(self.point_bodies.joints[13].position.z)		
            # 14 right_wrist (손목)
			right_wrist_pos_x = float(self.point_bodies.joints[14].position.x)
			right_wrist_pos_y = float(self.point_bodies.joints[14].position.y)
			right_wrist_pos_z = float(self.point_bodies.joints[14].position.z)		
            # 15 right_hand (손)
			right_hand_pos_x = float(self.point_bodies.joints[15].position.x)
			right_hand_pos_y = float(self.point_bodies.joints[15].position.y)
			right_hand_pos_z = float(self.point_bodies.joints[15].position.z)		
            # 16 right_handtip
			right_handtip_pos_x = float(self.point_bodies.joints[16].position.x)
			right_handtip_pos_y = float(self.point_bodies.joints[16].position.y)
			right_handtip_pos_z = float(self.point_bodies.joints[16].position.z)
            # 17 right_thumb (엄지)
			right_thumb_pos_x = float(self.point_bodies.joints[17].position.x)
			right_thumb_pos_y = float(self.point_bodies.joints[17].position.y)
			right_thumb_pos_z = float(self.point_bodies.joints[17].position.z)				
            # 18 left_hip (엉덩이)
			left_hip_pos_x = float(self.point_bodies.joints[18].position.x)
			left_hip_pos_y = float(self.point_bodies.joints[18].position.y)
			left_hip_pos_z = float(self.point_bodies.joints[18].position.z)				
            # 19 left_knee (무릎)
			left_knee_pos_x = float(self.point_bodies.joints[19].position.x)
			left_knee_pos_y = float(self.point_bodies.joints[19].position.y)
			left_knee_pos_z = float(self.point_bodies.joints[19].position.z)				
            # 20 left_ankle (발목)
			left_ankle_pos_x = float(self.point_bodies.joints[20].position.x)
			left_ankle_pos_y = float(self.point_bodies.joints[20].position.y)
			left_ankle_pos_z = float(self.point_bodies.joints[20].position.z)				
            # 21 left_foot (발)
			left_foot_pos_x = float(self.point_bodies.joints[21].position.x)
			left_foot_pos_y = float(self.point_bodies.joints[21].position.y)
			left_foot_pos_z = float(self.point_bodies.joints[21].position.z)				
            # 22 right_hip (엉덩이)
			right_hip_pos_x = float(self.point_bodies.joints[22].position.x)
			right_hip_pos_y = float(self.point_bodies.joints[22].position.y)
			right_hip_pos_z = float(self.point_bodies.joints[22].position.z)				
            # 23 right_knee (무릎)
			right_knee_pos_x = float(self.point_bodies.joints[23].position.x)
			right_knee_pos_y = float(self.point_bodies.joints[23].position.y)
			right_knee_pos_z = float(self.point_bodies.joints[23].position.z)				
            # 24 right_ankle (발목)
			right_ankle_pos_x = float(self.point_bodies.joints[24].position.x)
			right_ankle_pos_y = float(self.point_bodies.joints[24].position.y)
			right_ankle_pos_z = float(self.point_bodies.joints[24].position.z)				
            # 25 right_foot (발)
			right_foot_pos_x = float(self.point_bodies.joints[25].position.x)
			right_foot_pos_y = float(self.point_bodies.joints[25].position.y)
			right_foot_pos_z = float(self.point_bodies.joints[25].position.z)				
            # 26 head (머리)
			head_pos_x = float(self.point_bodies.joints[26].position.x)
			head_pos_y = float(self.point_bodies.joints[26].position.y)
			head_pos_z = float(self.point_bodies.joints[26].position.z)				
            # 27 nose (코)
			# nose_pos_x = float(self.point_bodies.joints[27].position.x)
			# nose_pos_y = float(self.point_bodies.joints[27].position.y)
			# nose_pos_z = float(self.point_bodies.joints[27].position.z)				
            # 28 left_eye (눈)
			# left_eye_pos_x = float(self.point_bodies.joints[28].position.x)
			# left_eye_pos_y = float(self.point_bodies.joints[28].position.y)
			# left_eye_pos_z = float(self.point_bodies.joints[28].position.z)				
            # 29 left_ear (귀)
			# left_ear_pos_x = float(self.point_bodies.joints[29].position.x)
			# left_ear_pos_y = float(self.point_bodies.joints[29].position.y)
			# left_ear_pos_z = float(self.point_bodies.joints[29].position.z)				
            # 30 right_eye (눈)
			# right_eye_pos_x = float(self.point_bodies.joints[30].position.x)
			# right_eye_pos_y = float(self.point_bodies.joints[30].position.y)
			# right_eye_pos_z = float(self.point_bodies.joints[30].position.z)				
            # 31 right_ear (귀)
			# right_ear_pos_x = float(self.point_bodies.joints[31].position.x)
			# right_ear_pos_y = float(self.point_bodies.joints[31].position.y)
			# right_ear_pos_z = float(self.point_bodies.joints[31].position.z)				
            
            # arr1 = np.dstack()
            
			arr = np.array([datetime,
			[pelvis_pos_x, pelvis_pos_y, pelvis_pos_z],
			[navel_pos_x, navel_pos_y, navel_pos_z], 
			[neck_pos_x, neck_pos_y, neck_pos_z], 
			[head_pos_x, head_pos_y, head_pos_z],
			[left_shoulder_pos_x, left_shoulder_pos_y, left_shoulder_pos_z],
			[left_elbow_pos_x, left_elbow_pos_y, left_elbow_pos_z],
			[left_wrist_pos_x, left_wrist_pos_y, left_wrist_pos_z],
			[left_hand_pos_x, left_hand_pos_y, left_hand_pos_z],
			[right_shoulder_pos_x, right_shoulder_pos_y, right_shoulder_pos_z],
			[right_elbow_pos_x, right_elbow_pos_y, right_elbow_pos_z],
			[right_wrist_pos_x, right_wrist_pos_y, right_wrist_pos_z],
			[right_hand_pos_x, right_hand_pos_y, right_hand_pos_z],
    		[left_hip_pos_x, left_hip_pos_y, left_hip_pos_z],
			[left_knee_pos_x, left_knee_pos_y, left_knee_pos_z],
			[left_ankle_pos_x, left_ankle_pos_y, left_ankle_pos_z],
			[left_foot_pos_x, left_foot_pos_y, left_foot_pos_z],
			[right_hip_pos_x, right_hip_pos_y, right_hip_pos_z],
			[right_knee_pos_x, right_knee_pos_y, right_knee_pos_z],
			[right_ankle_pos_x, right_ankle_pos_y, right_ankle_pos_z],
			[right_foot_pos_x, right_foot_pos_y, right_foot_pos_z],
			[chest_pos_x, chest_pos_y, chest_pos_z], 
			[left_handtip_pos_x, left_handtip_pos_y, left_handtip_pos_z],
			[left_thumb_pos_x, left_thumb_pos_y, left_thumb_pos_z],			
			[right_handtip_pos_x, right_handtip_pos_y, right_handtip_pos_z],
			[right_thumb_pos_x, right_thumb_pos_y, right_thumb_pos_z],	 
			],dtype=object)
			print(datetime)
			self.list_save.append(arr)
   
	# 수정
	def npToSkeleton(self):
		listOfData = self.list_save.tolist()
		listOfResult = [0 for i in range(25)]
	
		f = open(SKELETON_DATA_PATH + "np.skeleton", 'w')
		f.write(str(len(listOfData)) + "\n\n")
		for i in range(len(listOfData)):
			idx = 0
			for j in range(1, len(listOfData[i])):
				listOfResult.append(self.list_save[i][j])
				idx = idx + 1
    
			for j in range(25):
				temp = str(listOfResult[j])[1:-1]
				temp = temp.replace(',','')
				f.write(temp + "\n")
			listOfResult = [0 for i in range(25)]
			f.write("\n")
		f.close()
  
	def readSkeleton(self, file):
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

	def read_xyz(self, file, max_body=2, num_joint=25):
		seq_info = self.readSkeleton(file)
		data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
		for n, f in enumerate(seq_info['frameInfo']):
			for m, b in enumerate(f['bodyInfo']):
				for j, v in enumerate(b['jointInfo']):
					if m < max_body and j < num_joint:
						data[:, n, j, m] = [v['x'], v['y'], v['z']]
					else:
						pass
		data = np.around(data, decimals=3)
		return data
	


class ImageViewer(QtWidgets.QWidget):
	def __init__(self, parent=None):
		super(ImageViewer, self).__init__(parent)
		self.init_image = cv2.imread('init_Image.jpg')
		self.init_image = cv2.cvtColor(self.init_image, cv2.COLOR_BGR2RGB)
		self.h, self.w, self.c = self.init_image.shape
		self.image = QtGui.QImage(self.init_image.data, self.w, self.h, self.w*self.c, 		QtGui.QImage.Format_RGB888)
		self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawImage(0, 0, self.image)
		self.image = QtGui.QImage()

	def initUI(self):
		self.setWindowTitle('Test')

	@QtCore.pyqtSlot(QtGui.QImage)
	def setImage(self, image):
		if image.isNull():
			print("Viewer Dropped frame!")

		self.image = image
		if image.size() != self.size():
			self.setFixedSize(image.size())
		self.update()


if __name__ == '__main__':

	app = QtWidgets.QApplication(sys.argv)

	thread = QtCore.QThread()
	thread.start()
	vid = ShowVideo()
	vid.moveToThread(thread)

	# 창을 띄워 각 카메라에 맞는 이미지뷰를 띄우고 녹화버튼 생성하는 세팅
	image_viewer_RGB    = ImageViewer()
	image_viewer_Depth  = ImageViewer()
	image_viewer_IR     = ImageViewer()
	image_viewer_THERMAL= ImageViewer()

	vid.VideoSignal_RGB.connect(image_viewer_RGB.setImage)
	vid.VideoSignal_Depth.connect(image_viewer_Depth.setImage)
	vid.VideoSignal_IR.connect(image_viewer_IR.setImage)
	vid.VideoSignal_THERMAL.connect(image_viewer_THERMAL.setImage)

	push_button1 = QtWidgets.QPushButton('Record')
	push_button1.clicked.connect(vid.record_button)
	quit_button = QtWidgets.QPushButton('Record end')
	quit_button.clicked.connect(vid.quit_button)
    
	vertical_layout = QtWidgets.QVBoxLayout()
	horizontal_layout = QtWidgets.QHBoxLayout()

	vertical_layout.addWidget(image_viewer_RGB)
	horizontal_layout.addWidget(image_viewer_Depth)
	horizontal_layout.addWidget(image_viewer_IR)
	horizontal_layout.addWidget(image_viewer_THERMAL)

	vertical_layout.addLayout(horizontal_layout)
	vertical_layout.addWidget(push_button1)
	vertical_layout.addWidget(quit_button)


    # gridlay = QtWidgets.QGridLayout()
	layout_widget = QtWidgets.QWidget()
	layout_widget.setLayout(vertical_layout)

	main_window = QtWidgets.QMainWindow()
	main_window.setCentralWidget(layout_widget)
	main_window.show()
	vid.startVideo()
    
	app.exec_()
	sys.exit(app.exec_())