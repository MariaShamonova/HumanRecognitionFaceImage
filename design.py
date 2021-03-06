# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiTM.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.uploadTemplate = QtWidgets.QPushButton(self.centralwidget)
        self.uploadTemplate.setGeometry(QtCore.QRect(430, 10, 99, 32))
        self.uploadTemplate.setObjectName("uploadTemplate")
        self.uploadImage = QtWidgets.QPushButton(self.centralwidget)
        self.uploadImage.setGeometry(QtCore.QRect(270, 10, 79, 32))
        self.uploadImage.setObjectName("uploadImage")
        self.resultButton = QtWidgets.QPushButton(self.centralwidget)
        self.resultButton.setGeometry(QtCore.QRect(340, 250, 99, 32))
        self.resultButton.setStyleSheet("background-color: rgb(0, 118, 0);\n"
"color: rgb(255, 255, 255);")
        self.resultButton.setObjectName("resultButton")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(220, 60, 361, 181))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setVerticalSpacing(20)
        self.gridLayout.setObjectName("gridLayout")
        self.temp = QtWidgets.QLabel(self.widget)
        self.temp.setAlignment(QtCore.Qt.AlignCenter)
        self.temp.setIndent(0)
        self.temp.setObjectName("temp")
        self.gridLayout.addWidget(self.temp, 0, 1, 1, 1)
        self.image = QtWidgets.QLabel(self.widget)
        self.image.setMaximumSize(QtCore.QSize(16777215, 180))
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setIndent(0)
        self.image.setObjectName("image")
        self.gridLayout.addWidget(self.image, 0, 0, 1, 1)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(30, 300, 731, 237))
        self.widget1.setObjectName("widget1")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget1)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.TM_CCOEFF_label = QtWidgets.QLabel(self.widget1)
        self.TM_CCOEFF_label.setMinimumSize(QtCore.QSize(0, 25))
        self.TM_CCOEFF_label.setAlignment(QtCore.Qt.AlignCenter)
        self.TM_CCOEFF_label.setObjectName("TM_CCOEFF_label")
        self.gridLayout_2.addWidget(self.TM_CCOEFF_label, 0, 0, 1, 1)
        self.TM_CCOEFF = QtWidgets.QLabel(self.widget1)
        self.TM_CCOEFF.setMinimumSize(QtCore.QSize(0, 200))
        self.TM_CCOEFF.setAlignment(QtCore.Qt.AlignCenter)
        self.TM_CCOEFF.setObjectName("TM_CCOEFF")
        self.gridLayout_2.addWidget(self.TM_CCOEFF, 1, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.TM_CCORR_label = QtWidgets.QLabel(self.widget1)
        self.TM_CCORR_label.setMinimumSize(QtCore.QSize(0, 25))
        self.TM_CCORR_label.setAlignment(QtCore.Qt.AlignCenter)
        self.TM_CCORR_label.setObjectName("TM_CCORR_label")
        self.gridLayout_3.addWidget(self.TM_CCORR_label, 0, 0, 1, 1)
        self.TM_CCORR = QtWidgets.QLabel(self.widget1)
        self.TM_CCORR.setMinimumSize(QtCore.QSize(0, 200))
        self.TM_CCORR.setAlignment(QtCore.Qt.AlignCenter)
        self.TM_CCORR.setObjectName("TM_CCORR")
        self.gridLayout_3.addWidget(self.TM_CCORR, 1, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_3, 0, 1, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.TM_SQDIFF_label = QtWidgets.QLabel(self.widget1)
        self.TM_SQDIFF_label.setMinimumSize(QtCore.QSize(0, 25))
        self.TM_SQDIFF_label.setAlignment(QtCore.Qt.AlignCenter)
        self.TM_SQDIFF_label.setObjectName("TM_SQDIFF_label")
        self.gridLayout_4.addWidget(self.TM_SQDIFF_label, 0, 0, 1, 1)
        self.TM_SQDIFF = QtWidgets.QLabel(self.widget1)
        self.TM_SQDIFF.setMinimumSize(QtCore.QSize(0, 200))
        self.TM_SQDIFF.setAlignment(QtCore.Qt.AlignCenter)
        self.TM_SQDIFF.setObjectName("TM_SQDIFF")
        self.gridLayout_4.addWidget(self.TM_SQDIFF, 1, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.uploadTemplate.setText(_translate("MainWindow", "Template"))
        self.uploadImage.setText(_translate("MainWindow", "Image"))
        self.resultButton.setText(_translate("MainWindow", "Result"))
        self.temp.setText(_translate("MainWindow", "Template"))
        self.image.setText(_translate("MainWindow", "Image"))
        self.TM_CCOEFF_label.setText(_translate("MainWindow", "CCOEFF"))
        self.TM_CCOEFF.setText(_translate("MainWindow", "TM_CCOEFF"))
        self.TM_CCORR_label.setText(_translate("MainWindow", "CCORR"))
        self.TM_CCORR.setText(_translate("MainWindow", "TM_CCORR"))
        self.TM_SQDIFF_label.setText(_translate("MainWindow", "SQDIFF"))
        self.TM_SQDIFF.setText(_translate("MainWindow", "TextLabel"))
