import sys
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, \
    QFileDialog, QLabel
from PyQt5.QtWidgets import QDesktopWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QKeySequence


class MainWindow(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.resize(400, 400)
        # move the window to the center
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(frame_geometry.topLeft())
        self.statusBar().show()
        # add menu bar
        self.addMenu()
        self.pixmap = self.filename = None
        self.layout()
        self.show()

    def addMenu(self):
        menuBar = self.menuBar()
        # '&' for underscore
        fileMenu = menuBar.addMenu('&File')
        openAction = QAction('&Open', fileMenu)
        openAction.setShortcut(QKeySequence('Ctrl+O'))
        openAction.setStatusTip('open an image file')
        openAction.triggered.connect(self.openFile)
        fileMenu.addAction(openAction)


    def openFile(self):
        self.filename, _ = QFileDialog.getOpenFileName(filter="Images (*.png *.jpg)")
        self.pixmap = QPixmap(self.filename)
        self.label = QLabel(self)
        self.label.setPixmap(self.pixmap)
        self.label.setScaledContents(True)
        self.setCentralWidget(self.label)




if __name__ == '__main__':
    app = QApplication([])
    ex = MainWindow()
    sys.exit(app.exec_())
