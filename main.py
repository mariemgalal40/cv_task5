from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2 as cv
import numpy as np
from Recognition import FaceRecongnition


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.rgb2xyz_matrix = np.matrix('0.412453 0.357580 0.180423; ''0.212671 0.715160 0.072169; ''0.019334 0.119193 0.950227' )
        self.xyz2rgb_matrix = np.matrix('3.2404790 -1.537150 -0.498535; ''-0.969256 1.8759910 0.041556; ''0.0556480 -0.204043 1.057311')
        self.xw = 0.95
        self.yw = 100
        self.zw = 1.09
        self.uw = (4 * self.xw) / (self.xw + 15 * self.yw + 3 * self.zw)
        self.vw = (9 * self.yw) / (self.xw + 15 * self.yw + 3 * self.zw)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(910, 700)
        MainWindow.setStyleSheet("background-color:rgb(219, 219, 206);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, -20, 241, 721))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.widget.setFont(font)
        self.widget.setStyleSheet("backgroud-color: rgb(34, 34, 34);")
        self.widget.setObjectName("widget")
        self.browse = QtWidgets.QPushButton(self.widget)
        self.browse.setGeometry(QtCore.QRect(10, 110, 231, 41))
        self.browse.setStyleSheet("font: italic 20pt \"Times New Roman\";\n" "color:rgb(255, 255, 255);\n" "border-radius:20px;\n" "background-color:rgb(255, 170, 127);")
        self.browse.setObjectName("browse")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(10, 170, 231, 61))
        self.label.setStyleSheet("font: italic 10pt \"Times New Roman\";\n" "color:rgb(16, 15, 15);\n")
        self.label.setObjectName("label")
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(250, -10, 661, 691))
        self.widget_2.setStyleSheet("background-color:rgb(85, 85, 85);")
        self.widget_2.setObjectName("widget_2")
        self.label_5 = QtWidgets.QLabel(self.widget_2)
        self.label_5.setGeometry(QtCore.QRect(0, 10, 661, 311))
        self.label_5.setStyleSheet("background-color:rgb(85, 85, 85);\n""\n""")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.widget_2)
        self.label_6.setGeometry(QtCore.QRect(0, 360, 661, 311))
        self.label_6.setStyleSheet("background-color: rgb(85, 85, 85);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.widget_2)
        self.label_7.setGeometry(QtCore.QRect(0, 330, 661, 50))
        self.label_7.setStyleSheet("font: italic 15pt \"Times New Roman\";\n" "color:rgb(255, 255, 255);\n")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 910, 26))
        self.menubar.setObjectName("menubar")
        self.menuselect_operation = QtWidgets.QMenu(self.menubar)
        self.menuselect_operation.setObjectName("menuselect_operation")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionface_Detection = QtWidgets.QAction(MainWindow)
        self.actionface_Detection.setObjectName("actionface_Detection")
        self.actionface_Recognition = QtWidgets.QAction(MainWindow)
        self.actionface_Recognition.setObjectName("actionface_Recognition")
        self.actionROC = QtWidgets.QAction(MainWindow)
        self.actionface_Recognition.setObjectName("actionROC")
        self.menuselect_operation.addAction(self.actionface_Detection)
        self.actionface_Detection.triggered.connect(self.face_detection)
        self.menuselect_operation.addAction(self.actionface_Recognition)
        self.actionface_Recognition.triggered.connect(self.face_recognition)
        self.menuselect_operation.addAction(self.actionROC)
        self.actionROC.triggered.connect(self.ROC)
        self.menubar.addAction(self.menuselect_operation.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.browse.setText(_translate("MainWindow", "Browse image"))
        self.browse.clicked.connect(self.browse_image)
        self.menuselect_operation.setTitle(_translate("MainWindow", "select operation"))
        self.actionface_Detection.setText(_translate("MainWindow", "face_Detection"))
        self.actionface_Recognition.setText(_translate("MainWindow", "face_Recognition"))
        self.actionROC.setText(_translate("MainWindow", "ROC"))

    def browse_image(self):
        image1 = QFileDialog.getOpenFileName(None, 'OpenFile', '')
        self.imagePath = image1[0]
        pixmap = QPixmap(self.imagePath)
        self.label_5.setPixmap(pixmap.scaled(450, 300))

    def face_detection(self):
        # Get user supplied values
        cascPath = "haarcascade_frontalface_default.xml"
        # Create the haar cascade
        faceCascade = cv.CascadeClassifier(cascPath)
        # Read the image
        image = cv.imread(self.imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
        print("Found {0} faces!".format(len(faces)))
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imwrite('saved.png',image)
        pixmap = QPixmap('saved.png')
        self.label_6.setPixmap(pixmap.scaled(450, 300))

    def face_recognition(self):
        test = FaceRecongnition("train")
        result = test.fit(self.imagePath)
        print(result)
        if result[0] == "unknown Face":
            self.label_7.setText("unknown Face")
            print("unknown Face")
        else:
            print("ffffffffff")
            path = "train/" + result[0]
            pixmap = QPixmap(path)
            self.label_6.setPixmap(pixmap.scaled(450, 300))

    def ROC(self):
        test = FaceRecongnition("train")
        test.Roc()
        pixmap = QPixmap("roc.png")
        self.label_5.setPixmap(pixmap.scaled(450, 300))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())