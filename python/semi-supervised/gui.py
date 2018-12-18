import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, \
    QFileDialog, QLabel, QInputDialog
from PyQt5.QtWidgets import QDesktopWidget, QMessageBox, QGraphicsView, \
    QGraphicsScene
from PyQt5.QtGui import QPixmap, QKeySequence, QIcon, QCloseEvent, QTransform
import PyQt5.QtCore as QtCore


# TODO: zoom in, zoom out
# TODO: savedialog from the qt website
# TODO: better GUI, size policy
# TODO: Qstylesheet, layout management on the qt website
# TODO: load and save image in same bytes
# TODO: make tidy code

class MainWindow(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.resize(400, 400)
        # add menu bar
        self.addMenu()
        # add toolbar
        self.addToolbar()
        self.view = self.scene = None
        self.zoom = 1
        self.border = 20
        self.imagefiles = []
        self.pixmap = self.filename = self.image = self.file = None
        self.resolution = QDesktopWidget().availableGeometry()
        self.geometry_w = self.resolution.width()
        self.geometry_h = self.resolution.height()
        # for save prompt
        self.modified = False
        self.layout()
        self.statusBar().show()
        # move the window to the center
        self.movetoCenter()

    def addMenu(self):
        menuBar = self.menuBar()
        # '&' for underscore
        fileMenu = menuBar.addMenu('&File')
        openAction = QAction('&Open', fileMenu)
        openAction.setShortcut(QKeySequence('Ctrl+O'))
        openAction.setStatusTip('open an image')
        saveAction = QAction('&Save', fileMenu)
        saveAction.setShortcut(QKeySequence('Ctrl+S'))
        saveAction.setStatusTip('save an image')
        saveasAction = QAction('Save As', fileMenu)
        saveasAction.setShortcut(QKeySequence('Ctrl+A'))
        saveasAction.setStatusTip('save an image in the given format')
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
        self.zoominAction = QAction(QIcon('./zoom-in.png'), 'zoom-in', self.toolBar)
        self.zoominAction.setStatusTip('zoom in')
        self.zoominAction.setEnabled(False)
        self.zoominAction.triggered.connect(self.zoomIn)
        self.zoomoutAction = QAction(QIcon('./zoom-out.png'), 'zoom-out', self.toolBar)
        self.zoomoutAction.setStatusTip('zoom out')
        self.zoomoutAction.triggered.connect(self.zoomOut)
        self.zoomoutAction.setEnabled(False)
        self.toolBar.addAction(self.qbsAction)
        self.toolBar.addAction(self.qbeAction)
        self.toolBar.addAction(self.zoominAction)
        self.toolBar.addAction(self.zoomoutAction)

    def openFile(self):
        if self.checkSave():
            self.label = QLabel(self)
            self.label.setScaledContents(True)
            self.filename, _ = QFileDialog.getOpenFileName(caption="Open an image",
                                                           filter="JPEG (*.jpeg *.jpg);;PNG (*.png)")
            if self.filename:
                self.path, name = os.path.split(self.filename)
                suffix = name.split('.')[1].lower()
                self.imagefiles = [file for file in os.listdir(self.path) if file.lower().endswith(suffix)]
                self.imagefiles.sort()
                self.index = self.imagefiles.index(name)
                self.setCentralWidget(self.label)
                self.pixmap = QPixmap(self.filename)
                self.scene = QGraphicsScene(self)
                self.scene.addPixmap(self.pixmap)
                self.view = QGraphicsView(self.scene, self)

                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                print(self.view.transform().isIdentity())
                self.reset()
                self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                self.setCentralWidget(self.view)
                self.qbsAction.setEnabled(True)
                self.qbeAction.setEnabled(True)
                self.zoominAction.setEnabled(True)
                self.zoomoutAction.setEnabled(True)
                self.movetoCenter()

    def reset(self):
        ratio_w = self.pixmap.width() / (self.geometry_w - self.border)
        ratio_h = self.pixmap.height() / (self.geometry_h - self.border)
        print(ratio_w, ratio_h, self.geometry_w, self.geometry_h, self.pixmap.size())
        if ratio_w > 1:
            if ratio_h > 1:
                ratio = max(ratio_w, ratio_h)
                self.view.setTransform(QTransform().scale(1 / ratio, 1 / ratio))
                self.resize(self.pixmap.width() // ratio, self.pixmap.height() // ratio)
            else:
                self.view.setTransform(QTransform().scale(1 / ratio_w, 1 / ratio_w))
                self.resize(self.pixmap.width() // ratio_w, self.pixmap.height() // ratio_w)
        elif ratio_h > 1:
            self.view.setTransform(QTransform().scale(1 / ratio_h, 1 / ratio_h))
            self.resize(self.pixmap.width() // ratio_h, self.pixmap.height() // ratio_h)
        else:
            self.view.resetTransform()
            #self.view.resize(self.pixmap.size())
            self.resize(self.pixmap.size())
        self.view.verticalScrollBar().setValue(0)
        self.view.horizontalScrollBar().setValue(0)

    def search(self):
        text, ok = QInputDialog.getText(self, 'Input', "Enter")

    def movetoCenter(self):
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(self.resolution.center())
        self.move(frame_geometry.topLeft())
        self.show()

    def saveFile(self, fileFormat='.png'):
        if self.pixmap.save(self.filename, fileFormat):
            self.modified = False

    def saveasFile(self):
        savefile, filter = QFileDialog.getSaveFileName(caption="Save as", directory='./no_title.jpeg',
                                                       filter="JPEG (*.jpeg, *.jpg);;PNG (*.png);;BMP (*.bmp)")
        if self.pixmap:
            savefile = savefile.lower()
            if filter.startswith('JPEG'):
                if not savefile.endswith('jpeg') and not savefile.endswith('jpg'):
                    savefile += '.jpeg'
            elif filter.startswith('PNG'):
                if not savefile.endswith('png'):
                    savefile += '.png'
            else:
                if not savefile.endswith('bmp'):
                    savefile += '.bmp'
            self.pixmap.save(savefile, quality=100)

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

    def zoomIn(self):
        self.zoom *= 1.1
        self.view.setTransform(QTransform().scale(self.zoom, self.zoom))

    def zoomOut(self):
        self.zoom /= 1.1
        self.view.setTransform(QTransform().scale(self.zoom, self.zoom))

    def keyPressEvent(self, e):
        if self.imagefiles:
            if e.key() == QtCore.Qt.Key_W:
                self.zoomIn()
            elif e.key() == QtCore.Qt.Key_S:
                self.zoomOut()
            elif e.key() == QtCore.Qt.Key_A:
                self.browseDir(-1)
            elif e.key() == QtCore.Qt.Key_D:
                self.browseDir(1)

    def mouseDoubleClickEvent(self, e):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, e: QCloseEvent):
        if self.checkSave():
            e.accept()
        else:
            e.ignore()

    def browseDir(self, direction: int):
        length = len(self.imagefiles)
        if length > 1:
            self.index = (self.index + direction + length) % length
            self.pixmap = QPixmap(os.path.join(self.path, self.imagefiles[self.index]))
            self.scene.clear()
            self.scene.addPixmap(self.pixmap)
            self.reset()
            self.movetoCenter()



if __name__ == '__main__':
    app = QApplication([])
    ex = MainWindow()
    sys.exit(app.exec_())
