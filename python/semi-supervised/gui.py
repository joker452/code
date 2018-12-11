import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, \
    QFileDialog, QLabel, QInputDialog
from PyQt5.QtWidgets import QDesktopWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QKeySequence, QIcon, QCloseEvent,\
    QImage
from PyQt5.QtCore import QFile, QIODevice, QBuffer,QByteArray, QSaveFile


# TODO: zoom in, zoom out
# TODO: savedialog from the qt website
# TODO: better GUI, size policy
# TODO: Qstylesheet, layout management on the qt website
# TODO: make tidy code

class MainWindow(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.resize(400, 400)
        # move the window to the center
        self.movetoCenter()
        self.statusBar().show()
        # add menu bar
        self.addMenu()
        # add toolbar
        self.addToolbar()
        self.pixmap = self.filename = self.image = self.file = None
        # for save prompt
        self.modified = False
        self.layout()
        self.show()

    def addMenu(self):
        menuBar = self.menuBar()
        # '&' for underscore
        fileMenu = menuBar.addMenu('&File')
        openAction = QAction('&Open', fileMenu)
        openAction.setShortcut(QKeySequence('Ctrl+O'))
        openAction.setStatusTip('open an image file')
        saveAction = QAction('&Save', fileMenu)
        saveAction.setShortcut(QKeySequence('Ctrl+S'))
        saveAction.setStatusTip('open an image file')
        saveasAction = QAction('Save As', fileMenu)
        saveAction.setShortcut(QKeySequence('Ctrl+A'))
        openAction.triggered.connect(self.openFile)
        saveasAction.triggered.connect(self.saveasFile)

        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(saveasAction)

    def addToolbar(self):
        self.toolBar = self.addToolBar('Search')
        self.qbsAction = QAction(QIcon('./QBS.png'), 'qbs', self.toolBar)
        self.qbsAction.setStatusTip("query by string")
        self.qbsAction.setEnabled(False)
        self.qbeAction = QAction(QIcon('./QBE.png'), 'qbe', self.toolBar)
        self.qbeAction.setStatusTip("query by example")
        self.qbeAction.setEnabled(False)
        self.qbsAction.triggered.connect(self.search)
        self.toolBar.addAction(self.qbsAction)
        self.toolBar.addAction(self.qbeAction)

    def openFile(self):
        if self.checkSave():
            self.label = QLabel(self)
            self.label.setScaledContents(True)
            self.filename, _ = QFileDialog.getOpenFileName(caption="Open an image",
                                                           filter="JPEG (*.jpeg *.jpg);;PNG (*.png)")
            if self.filename:
                self.file = QFile(self.filename)
                if not self.file.open(QIODevice.ReadOnly):
                    return False
                self.imageData = self.file.readAll()
                if self.imageData.isEmpty():
                    return False
                self.file.close()

                self.image = QImage()
                self.buffer = QBuffer(self.imageData)
                self.buffer.open(QIODevice.ReadOnly)
                self.image.load(self.buffer, "jpg")
                self.buffer.close()
                self.setCentralWidget(self.label)

                self.pixmap = QPixmap.fromImage(self.image)
                self.frameGeometry().size()

                self.resize(self.pixmap.size())
                self.label.setPixmap(self.pixmap)
                self.qbsAction.setEnabled(True)
                self.qbeAction.setEnabled(True)
                self.movetoCenter()

    def search(self):
        text, ok = QInputDialog.getText(self, 'Input', "Enter")
        print("Label size after", self.label.size())

    def movetoCenter(self):
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(frame_geometry.topLeft())

    def saveFile(self, fileFormat='.png'):
        if self.pixmap.save(self.filename, fileFormat):
            self.modified = False

    def saveasFile(self):
        savefile, filter = QFileDialog.getSaveFileName(caption="Save as", directory='./no_title.jpeg',
                                               filter="JPEG (*.jpeg, *.jpg);;PNG (*.png);;BMP (*.bmp)")
        print(self.image)
        if self.image:
            # print(savefile)
            # savefile = savefile.lower()
            # if filter.startswith('JPEG'):
            #     if not savefile.endswith('jpeg') and not savefile.endswith('jpg'):
            #         savefile += '.jpeg'
            # elif filter.startswith('PNG'):
            #     if not savefile.endswith('png'):
            #         savefile += '.png'
            # else:
            #     if not savefile.endswith('bmp'):
            #         savefile += '.bmp'
            # self.image.save(savefile, quality=100)
            self.imageData = QByteArray()
            self.buffer = QBuffer(self.imageData)
            self.buffer.open(QIODevice.WriteOnly)
            self.image.save(self.buffer)
            print(self.buffer.size())
            file = QSaveFile(savefile)
            if not file.open(QIODevice.WriteOnly):
                print("No")
                return False
            print(self.imageData.size())
            file.write(self.imageData)


    def checkSave(self):
        if self.modified:
            reply = QMessageBox.warning(self, "Test",
                                        "There are unsaved changes.\n" +
                                        "Do you want to save your changes?",
                                        QMessageBox.Save | QMessageBox.Discard |
                                        QMessageBox.Cancel)
            if reply == QMessageBox.Save:
                return self.saveFile()
            elif reply == QMessageBox.Cancel:
                return False
        return True

    def closeEvent(self, e: QCloseEvent):
        if self.checkSave():
            e.accept()
        else:
            e.ignore()


if __name__ == '__main__':
    app = QApplication([])
    ex = MainWindow()
    sys.exit(app.exec_())
