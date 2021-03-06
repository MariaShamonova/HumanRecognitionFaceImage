import cv2 as cv
import os
from design import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore

# img_rgb = cv.imread('images/photo_1_0.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#
# template = cv.imread('images/template_1.png', 0)
# w, h = template.shape[::-1]
#
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCORR', 'cv.TM_SQDIFF']
#
# for meth in methods:
#     img = img_gray.copy()
#     image_rgb = img_rgb.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#     # If the method is TM_SQDIFF, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(image_rgb, top_left, bottom_right, 255, 2)
#     cv.imwrite(f'/images/{meth}.png', image_rgb)

class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()  # Это здесь нужно для доступа к переменным, методам
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.addFunctions()
        self.imageUrl = '3'
        self.templateUrl = '5'

    def addFunctions(self):
        self.uploadImage.clicked.connect(self.uploadImg)
        self.uploadTemplate.clicked.connect(self.uploadTemp)
        self.resultButton.clicked.connect(self.getResult)

    def uploadImg(self):
        fname = self.browserFiles()

        pixmap = QtGui.QPixmap(fname).scaled(176, 179, QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)

        self.imageUrl = fname
        print(self.imageUrl)

    def uploadTemp(self):
        fname = self.browserFiles()

        pixmap = QtGui.QPixmap(fname).scaled(176, 179, QtCore.Qt.KeepAspectRatio)
        self.temp.setPixmap(pixmap)
        print(self.temp)
        self.templateUrl = fname

    def browserFiles(self):
        dirname = os.getcwd()
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image',
                                                      dirname + '/images')
        return fname[0]

    def getResult(self):

        img_rgb = cv.imread(self.imageUrl)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

        template = cv.imread(self.templateUrl, 0)
        w, h = template.shape[::-1]

        methods = ['TM_CCOEFF', 'TM_CCORR', 'TM_SQDIFF']

        for meth in methods:
            img = img_gray.copy()
            image_rgb = img_rgb.copy()

            method = eval('cv.' + meth)

            # Apply template Matching
            res = cv.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(image_rgb, top_left, bottom_right, 255, 5)
            cv.imwrite(f'results/{meth}.png', image_rgb)

            pixmap = QtGui.QPixmap(f'results/{meth}.png').scaled(215, 200, QtCore.Qt.KeepAspectRatio)
            func = getattr(self, meth)
            func.setPixmap(pixmap)






def main():
    import sys
    app = QtWidgets.QApplication(sys.argv) # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()