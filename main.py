import sys  					# Для передачи argv в QApplication
import numpy as np
import os
import glob                     # Для загрузки всех картинок из папки
from PyQt5 import QtWidgets, uic
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split


class DataLoader():
    def CountFiles(somefolderpath):
        return len([name for name in os.listdir(somefolderpath) if os.path.isfile(os.path.join(somefolderpath, name))])

    def LoadImageAsRaster(someimagepath, width, height):                
        raster = io.imread(someimagepath)
        if raster.shape[0] > raster.shape[1]:
            return raster[:width,:height]
        else:
            return (raster.transpose())[:width,:height]

# На будущее (класс - окно программы):
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


def main():
    # Инициализация постоянных:
    data_path = "C:\\Users\\VG\\Desktop\\LeafProject\\LeafDataSet\\Segmented\\field"
    raster_width = 700
    raster_height = 522
    testsize = 0.2
    validationsize = 0.05 
    
    # Подсчет общего количества растров в библиотеке:
    count = 0
    subfolders_list = list([f.path for f in os.scandir(data_path) if f.is_dir()])    
    for subf in subfolders_list:
        count += DataLoader.CountFiles(subf);
    print("Обнаружено растров: " + str(count))

    # Выделение памяти под растры и метки:
    rasters = np.zeros((count,raster_width,raster_height), dtype=np.ubyte)
    labels = np.zeros((count), dtype = np.intc)


    # Загрузка растров и меток:
    print("Начинаем загрузку растров")
    i = 0                   # - ID растра
    j = 0                   # - ID класса растения

    for subf in subfolders_list:
        subf = os.path.join(subf,'*g')
        subf_rasters = glob.glob(subf)

        for r in subf_rasters:
            rasters[i] = DataLoader.LoadImageAsRaster(r,raster_width,raster_height)
            labels[i] = j
            i += 1

        j += 1
    print("Растры загружены.")

    # Разбиение выборки: 
    rasters_train, rasters_test, labels_train, labels_test = train_test_split(rasters, labels, test_size=testsize+validationsize, random_state=42)
    rasters_test, rasters_validation, labels_test, labels_validation = train_test_split(rasters_test, labels_test, test_size=validationsize/(testsize+validationsize), random_state=42)
    print("Выборка разделена в отношении: ")
    print(" - Train: " + str(rasters_train.shape[0]))
    print(" - Test: "+ str(rasters_test.shape[0]))
    print(" - Validation: " +str(rasters_validation.shape[0]))

    # На будущее:
    # ----
    #app = QtWidgets.QApplication(sys.argv)    # Создание приложения
    #window = MainWindow()  				   # Создание окна
    #window.show()   						   # Отображение окна
    #app.exec_()    						   # Запуск приложения
    # ----

   


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()