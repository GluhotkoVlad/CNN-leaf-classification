import sys
import numpy as np
import os
import glob
import cv2   
import tensorflow as tf
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataLoader():
	def CountFiles(somefolderpath):
		return len([name for name in os.listdir(somefolderpath) if os.path.isfile(os.path.join(somefolderpath, name))])

	def LoadAllData(allimagescount, datasetpath, rastersize, maxarraysize_in_GB):
		array_approximatesize = allimagescount * rastersize * rastersize * 3 * 1
		if (array_approximatesize < maxarraysize_in_GB * 1024 * 1024 * 1024):
			i = 0       			 # - ID растра
			j = 0		 			 # - ID класса растения

			nparray_rasters = np.zeros((allimagescount, rastersize, rastersize, 3), dtype=np.uint8)
			nparray_labels = np.zeros((allimagescount), dtype = np.uint16)
			subfolders_list = list([f.path for f in os.scandir(datasetpath) if f.is_dir()])

			# Вспомогательные переменные, чтобы отслеживать прогресс загрузки.
			temp_uploading_onepercent = int(allimagescount / 100)
			temp_uploading_step = 5

			for subf in subfolders_list:
				subf = os.path.join(subf,'*g')
				subf_rasters = glob.glob(subf)

				for r in subf_rasters:
					# Подготовка растра:
					# 1. Загрузка изображения из папки:
					image_loaded = cv2.imread(r)

					# 2. Обработка изображения - приведение к необходимому размеру с таким сохранением отношения сторон, насколько это возможно					
					image_resized = resize(image_loaded, (rastersize, rastersize))
					
					# 3. Перевод обработанного изображения в растр (массив NxNx3, где третяя размерность хранит RGB-координаты пикселя)
					# Для того, чтобы точки, выделенные другим оттенком цвета (напр. граничные) не обнулились после занесения в массив типа uint8,
					# используется функция sign, превращающая малые значения RGB в белый цвет, напр: [0.1, 0.4, 0.3] -> [1, 1, 1]  
					# raster = np.asarray(image_resized, dtype=np.float32)
					raster = np.sign(np.asarray(image_resized, dtype=np.float64)).astype('uint8')

					# Полученный растр необходимо загрузить в массив растров, каждая новая папка subf в цикле - новое значение j.
					nparray_rasters[i] = raster 
					nparray_labels[i] = j

					if (i % (temp_uploading_onepercent*temp_uploading_step) == 0):
						print("Обработано растров: " + str(i) + "  (" + str(int(100*i/allimagescount)) + "%)")

					i += 1		        
				j += 1
		else:
			print("Ошибка: массив будет иметь размер более, чем: " + str(maxarraysize_in_GB) + "Гб")
		return nparray_rasters, nparray_labels

	def LoadSingleImageAsRaster(someimagepath, rastersize):
		image_loaded = cv2.imread(someimagepath)
		image_resized = resize(image_loaded, (rastersize, rastersize))
		return np.asarray(image_resized, dtype=np.float64)

	def LoadArray(arrayname):
		return np.load(arrayname)

	def LoadModel(modelname):
		return tf.keras.models.load_model(modelname)

	def CountAllImages(folderpath):
		count = 0
		subfolders_list = list([f.path for f in os.scandir(folderpath) if f.is_dir()])

		for subf in subfolders_list:
			count += len([name for name in os.listdir(subf) if os.path.isfile(os.path.join(subf, name))])
		return count

	def GetAllClassesNamesList(somefolderpath):
		# Названия классов определяются названиями папок в библиотеке LeafData 
		subfolders_path_list = list([f.path for f in os.scandir(somefolderpath) if f.is_dir()])
		classesnames_list = []

		for subf_path in subfolders_path_list:
			l = len(subf_path)
			for i in range(0,l-1):
				if subf_path[l-1-i] == "\\" and i >= 1:					
					classesnames_list.append(subf_path[l-i : l]) 
					break

		return classesnames_list
		
class DataSaver():
	def SaveArray(arrayname, numpyarray):
		np.save(arrayname,numpyarray)

	def SaveModel(modelname, tensorflowmodel):
		tensorflowmodel.save(modelname)

class MainWindow(QtWidgets.QMainWindow):
	def __init__(self,tensorflowmodel,rastersize,predictionstopN,classesnames,somedatasetpath,window_width,window_height,image_width,image_height,txtlabel_leftrightmargin,txtlabel_height):
		# Загрузка параметров окна:
		super(MainWindow, self).__init__()
		uic.loadUi('mainwindow_design.ui', self)

		# Настройка окна:
		self.setFixedSize(window_width, window_height)
		self.imagewidth = image_width
		self.imageheight = image_height
		self.frombottomtxtboxmargin = window_height-min([self.findChild(QtWidgets.QPushButton, 'goLeft').y(),self.findChild(QtWidgets.QPushButton, 'goRight').y()])+5 # расположение панели кнопок определяется файлом интерфейса .ui, поэтому я учитываю расположение кнопок для последующей верстки 
		
		# Загрузка модели и переменных, связанных с моделью:
		self.tfmodel = tensorflowmodel
		self.rsize = rastersize
		self.predtopN = predictionstopN
		self.clnames = classesnames

		# Добавление методов-обработчиков на кнопки:
		self.button_goLeft = self.findChild(QtWidgets.QPushButton, 'goLeft') 
		self.button_goRight = self.findChild(QtWidgets.QPushButton, 'goRight')
		self.button_goLeft.clicked.connect(self.ButtonClick_goLeft)
		self.button_goRight.clicked.connect(self.ButtonClick_goRight)

		# Занесение списка путей изображений из ValidationSet в поле класса, подсчет изображений:
		self.allimagespath = glob.glob(os.path.join(somedatasetpath,'*g'))
		self.allimagesamount = len(self.allimagespath)

		# Подготовка окна для отображения текста:
		self.txtlabel = QtWidgets.QTextEdit(self)
		self.txtlabelleftrightmargin = txtlabel_leftrightmargin
		self.txtlabelheight = txtlabel_height
		self.txtlabel.setGeometry(txtlabel_leftrightmargin, window_height-txtlabel_height-self.frombottomtxtboxmargin, window_width-txtlabel_leftrightmargin*2, txtlabel_height)
		
		# Подготовка окна для отображения изображения:
		self.imglabel = QtWidgets.QLabel(self)
		self.imglabel.setGeometry(int((window_width-image_width)/2), int((window_height-image_height-txtlabel_height-self.frombottomtxtboxmargin)/2), image_width, image_height)

		# Сначала показывается первое изображение в выбранной папке:
		self.currentimagecounter = 0;
		self.ImageLoad(self.allimagespath[0], image_width, image_height)
		self.TextLoad(tensorflowmodel,classesnames,rastersize,predictionstopN)  
		self.show()

	def ImageLoad(self, imgpath, imgwidth, imgheight):
		self.imglabel.clear()
		self.imglabel.setPixmap((QPixmap(imgpath)).scaled(imgwidth, imgheight))
		self.show()

	def TextLoad(self, TFmodel, clnames, rsize, n):
		currentraster = np.zeros((1,rsize,rsize,3), dtype=np.float64)
		currentraster[0] = DataLoader.LoadSingleImageAsRaster(self.allimagespath[self.currentimagecounter], rsize)
		topNlabels, topNresults = WorkWithTensorFlowModel.MakePrediction(TFmodel,clnames,currentraster,n)
		
		predictiontext = "Нейросеть думает, что это:" + '\n' 
		for i in range(0,n):
			predictiontext = predictiontext + topNlabels[i] + "  ----  " + str(100*topNresults[i]) + "%" + '\n'
		self.txtlabel.setText(predictiontext)

	# Обработчики нажатия на кнопки:
	def ButtonClick_goLeft(self):
		if self.currentimagecounter == 0:
			self.currentimagecounter = self.allimagesamount - 1 		# - к последнему изображению:
		else:
			self.currentimagecounter = self.currentimagecounter - 1

		self.ImageLoad(self.allimagespath[self.currentimagecounter],self.imagewidth,self.imageheight)
		self.TextLoad(self.tfmodel,self.clnames,self.rsize,self.predtopN)

	def ButtonClick_goRight(self):
		if self.currentimagecounter == self.allimagesamount - 1:
			self.currentimagecounter = 0								# - к первому изображению.
		else:
			self.currentimagecounter = self.currentimagecounter + 1

		self.ImageLoad(self.allimagespath[self.currentimagecounter],self.imagewidth,self.imageheight)
		self.TextLoad(self.tfmodel,self.clnames,self.rsize,self.predtopN)

class WorkWithTensorFlowModel():
	def MakePrediction(model, labelsnames, raster, topN):
		predictions = (model.predict(raster))[0]
		labelsnames_copy = labelsnames.copy()
		n = predictions.shape[0]

		if n < topN:
			print("Нельзя выделить " + str(topN) + " классов, так как это больше, чем общее число классов: " + str(allclasses_count))
		else:
			for i in range(0, n):
				for j in range(i+1, n):
					if predictions[i] < predictions[j]:
						predictions[i], predictions[j] = predictions[j], predictions[i]
						labelsnames_copy[i], labelsnames_copy[j] = labelsnames_copy[j], labelsnames_copy[i]

		return labelsnames_copy[:topN], predictions[:topN]


def main(traindatapath,validationdatapath,rastersarray_save_name,labelssarray_save_name,model_save_name,rastersize,maxarraysizeGB,testsize,predictions_topN,mode1="new",mode2="new"):
	# Подсчет общего количества фотографий в библиотеке:
	count = DataLoader.CountAllImages(traindatapath)
	print("Обнаружено фотографий: " + str(count))

	# Загрузка всех данных - растров, меток и список названий всех классов:
	if mode1 == "new":
		# Путь первый - извлечение и обработка исходных данных, преобразование их в массив, сохранение массивов:
		rasters, labels = DataLoader.LoadAllData(count, traindatapath, rastersize, maxarraysizeGB)
		DataSaver.SaveArray(rastersarray_save_name, rasters)
		DataSaver.SaveArray(labelssarray_save_name, labels)

	elif mode1 == "load":
		# Путь второй - загрузка уже заранее обработанных данных из 
		rasters, labels = DataLoader.LoadArray(rastersarray_save_name), DataLoader.LoadArray(labelssarray_save_name)

	# Загрузка всех названий классов:
	labelsnames = DataLoader.GetAllClassesNamesList(traindatapath)
	amountofspecies = len(labelsnames)

	# Разбиение выборки: 
	rasters_train, rasters_test, labels_train, labels_test = train_test_split(rasters, labels, test_size=testsize, random_state=42)
	print("Выборка разделена в отношении: ")
	print(" - Train: " + str(rasters_train.shape[0]))
	print(" - Test: "+ str(rasters_test.shape[0]))

	# Модель:
	if mode2 == "load":
		# Путь первый: загрузка готовой модели:
		model = DataLoader.LoadModel(model_save_name)

	elif mode2 == "new":
		# Путь второй: создание новой модели:
		# Программирование архитектуры модели при помощи пакетов Tensorflow/Keras:
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(32,(3,3),padding='same', input_shape=(rastersize,rastersize,3), activation='relu'))
		model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
		model.add(tf.keras.layers.Dropout(0.25))
		model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))
		model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
		model.add(tf.keras.layers.Dropout(0.25))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(512,activation='relu'))
		model.add(tf.keras.layers.Dropout(0.5))
		model.add(tf.keras.layers.Dense(amountofspecies,activation='softmax'))

		
		# Настройка оптимизатора SGD к модели:
		sgd = tf.keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

		# Обучение модели:
		model.fit(rasters_train, to_categorical(labels_train), batch_size=50, nb_epoch=20, validation_split=0.0, shuffle=False)
	
		# Сохранение обученной модели в папку с названием model_save_name:
		DataSaver.SaveModel(model_save_name,model)
	
	# Проверка модели на тестовых данных:
	scores = model.evaluate(rasters_test, to_categorical(labels_test), verbose=0)
	print("Точность на тестовых данных: "+str(100*scores[1]))
	
	# Запуск окна:
	app = QtWidgets.QApplication(sys.argv)    																			 # Создание приложения
	window = MainWindow(model,rastersize,predictions_topN,labelsnames,validationdatapath,440, 560, 375, 375, 32, 120)    # Создание окна
	window.show()																										 # Отображение окна
	app.exec_() 

if __name__ == '__main__':
	# Входные параметры:
	traindata_path = "C:\\Users\\VG\\Desktop\\LeafProject\\LeafDataSet\\Segmented\\field"
	validationdata_path = "C:\\Users\\VG\\Desktop\\LeafProject\\Program\\ValidationSet"
	rastersarray_save_name = 'rasters.npy'
	labelsarray_save_name = 'labels.npy'
	model_save_name = '40percents'
	raster_size = 32
	maxarraysize_in_Gigabytes = 6
	test_size = 0.2
	predictions_topN = 5
	arraymode = "load"
	modelmode = "load"

	main(traindata_path, validationdata_path, rastersarray_save_name, labelsarray_save_name, model_save_name, raster_size, maxarraysize_in_Gigabytes, test_size, predictions_topN, mode1=arraymode, mode2=modelmode)