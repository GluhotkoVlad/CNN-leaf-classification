import sys
import numpy as np
import os
import glob
import cv2   
import tensorflow as tf
from PyQt5 import QtWidgets, uic
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

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
					# 1. Загрузка изображения из папки и получение его размеров:

					image_loaded = cv2.imread(r)
					width = image_loaded.shape[0]
					height = image_loaded.shape[1]

					# 2. Обработка изображения - приведение к необходимому размеру с таким сохранением отношения сторон, насколько это возможно
					if width == max([width, height]):
						image_resized = resize(image_loaded, (rastersize, int(rastersize*(height/width))))
					else:
						image_resized = resize(image_loaded, (int(rastersize*(width/height)), rastersize))  

					# 3. Перевод обработанного изображения в растр (массив NxNx3, где третяя размерность хранит RGB-координаты пикселя)
					# Для того, чтобы точки, выделенные другим оттенком цвета (напр. граничные) не обнулились после занесения в массив типа uint8,
					# используется функция sign, превращающая малые значения RGB в белый цвет, напр: [0.1, 0.4, 0.3] -> [1, 1, 1]  
					raster = np.sign(np.asarray(image_resized, dtype=np.float64)).astype('uint8')

					# Полученный растр необходимо загрузить в массив растров. Поскольку размер растра может отличаться
					# от размера отведенного под него места (но не превышать его), то необходимо приравнивать поэлементно,
					# а оставшиеся пиксели, что не были присвоены, так и останутся нулевыми.
					nparray_rasters[i,:raster.shape[0],:raster.shape[1]] = raster 
					nparray_labels[i] = j

					if (i % (temp_uploading_onepercent*temp_uploading_step) == 0):
						print("Загружено фотографий: " + str(i) + "  (" + str(int(100*i/allimagescount)) + "%)")

					i += 1		        
				j += 1
		else:
			print("Ошибка: массив будет иметь размер более, чем: " + str(maxarraysize_in_GB) + "Гб")
		return nparray_rasters, nparray_labels

	def LoadSingleImage(someimagepath):
		return cv2.imread(someimagepath)

	def LoadArray(arrayname):
		return np.load(arrayname)

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

class MainWindow(QtWidgets.QMainWindow):
	def __init__(self):
		# Загрузка параметров окна:
		super(MainWindow, self).__init__()
		uic.loadUi('mainwindow_design.ui', self)

		# Настройка окна:
		self.setFixedSize(443, 560)

		# Добавление методов-обработчиков на кнопки
		self.button_goLeft = self.findChild(QtWidgets.QPushButton, 'goLeft') 
		self.button_goRight = self.findChild(QtWidgets.QPushButton, 'goRight')
		self.button_goLeft.clicked.connect(self.ButtonClick_goLeft)
		self.button_goRight.clicked.connect(self.ButtonClick_goRight)  
		self.show()

	# Обработчики нажатия на кнопки:
	def ButtonClick_goLeft(self):
		# ...
		print("Go left")

	def ButtonClick_goRight(self):
		# ...
		print("Go right")

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

		return predictions[:topN], labelsnames_copy[:topN]


def main(datapath, rastersarray_save_name, labelssarray_save_name, rastersize, maxarraysizeGB, testsize, validationsize, predictions_topN):
	# Подсчет общего количества растров в библиотеке:
	count = DataLoader.CountAllImages(datapath)
	print("Обнаружено растров: " + str(count))

	# Загрузка всех данных - растров, меток и список названий всех классов:
	# Путь первый - извлечение и обработка исходных данных, преобразование их в массив, сохранение массивов:
	rasters, labels = DataLoader.LoadAllData(count, datapath, rastersize, maxarraysizeGB)
	DataSaver.SaveArray(rastersarray_save_name, rasters)
	DataSaver.SaveArray(labelssarray_save_name, labels)

	# Путь второй - загрузка уже заранее обработанных данных из 
	# rasters, labels = DataLoader.LoadArray(rastersarray_save_name), DataLoader.LoadArray(labelssarray_save_name)

	# Загрузка всех названий классов:
	labelsnames = DataLoader.GetAllClassesNamesList(datapath)
	amountofspecies = len(labelsnames)

	# Разбиение выборки: 
	rasters_train, rasters_test, labels_train, labels_test = train_test_split(rasters, labels, test_size=testsize+validationsize, random_state=42)
	rasters_test, rasters_validation, labels_test, labels_validation = train_test_split(rasters_test, labels_test, test_size=validationsize/(testsize+validationsize), random_state=42)
	print("Выборка разделена в отношении: ")
	print(" - Train: " + str(rasters_train.shape[0]))
	print(" - Test: "+ str(rasters_test.shape[0]))
	print(" - Validation: " +str(rasters_validation.shape[0]))

	# Создание архитектуры нейросети при помощи tensorflow2:
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(16, (5,5), activation='relu', input_shape=(rastersize, rastersize,3)),
		tf.keras.layers.MaxPooling2D(3,3),
		tf.keras.layers.Conv2D(32, (4,4), activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dense(amountofspecies, activation='sigmoid')
	])

	model.compile(optimizer='adam',	loss='sparse_categorical_crossentropy',	metrics=['accuracy'])
	model.fit(rasters_train, labels_train, batch_size=50, epochs=10, validation_data=(rasters_test,labels_test))
	# 40% точность

if __name__ == '__main__':
	# Константы:
	data_path = "C:\\Users\\VG\\Desktop\\LeafProject\\LeafDataSet\\Segmented\\field"
	rastersarray_save_name = 'rasters.npy'
	labelsarray_save_name = 'labels.npy'
	raster_size = 128
	maxarraysize_in_Gigabytes = 1
	test_size = 0.2
	validation_size = 0.05
	predictions_topN = 5

	main(data_path, rastersarray_save_name, labelsarray_save_name, raster_size, maxarraysize_in_Gigabytes, test_size, validation_size, predictions_topN)