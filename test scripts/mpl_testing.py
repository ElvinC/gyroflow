from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import sys

from PySide2 import QtCore, QtWidgets, QtGui



class Launcher(QtWidgets.QWidget):
    """Main launcher with options to open different utilities
    """
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Testing")


        self.calibrator_button = QtWidgets.QPushButton("Camera Calibrator")
        self.calibrator_button.setMinimumSize(300,50)
        self.calibrator_button.setStyleSheet("font-size: 14px;")
        self.calibrator_button.setToolTip("Use this to generate camera calibration files")


        self.layout = QtWidgets.QVBoxLayout()

        self.layout.addWidget(self.calibrator_button)
        
        

        self.setLayout(self.layout)

        self.calibrator_button.clicked.connect(self.open_calib_util)


    def open_calib_util(self):
        """Open the camera calibration utility in a new window
        """

        plt.figure()

        plt.plot(np.random.random(10))

        plt.show()

        plt.figure()

        plt.plot(np.random.random(10))

        plt.show()

        #plt.figure()
        n = 2
        fig, axes = plt.subplots(3, 2, sharey=True)
        fig.set_size_inches(4 * n, 6)
        for j in range(n):
            axes[0][j].set(title=f"Syncpoint")
            for i, r in enumerate(['x', 'y', 'z']):
                axes[i][j].plot(np.random.random(10))
        plt.show()





if __name__ == "__main__":
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))

    app = QtWidgets.QApplication([])

    widget = Launcher() # Launcher()
    widget.resize(500, 500)


    widget.show()

    sys.exit(app.exec_())