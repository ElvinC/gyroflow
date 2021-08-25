import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Qt
from cv2 import detail_HomographyBasedEstimator
import numpy as np

class MultiSyncList(QtWidgets.QListWidget):
    def __init__(self, parent=None, stab=None):
        super().__init__(parent)
        self.stab = stab # link with stabilization utility

        self.setAlternatingRowColors(True)

        self.item_data = []

        #self.add_row(5)
        #self.add_row(10)

        self.itemDoubleClicked.connect(self.item_click_action)
        

    def add_row(self, time=0, frames=40,delay=0,cost=100):

        self.item_data.append({
            "time": time,
            "frames": frames,
            "delay": delay,
            "cost": cost
        })

        itm = QtWidgets.QListWidgetItem() 
        #Create widget
        widget = QtWidgets.QWidget()
        timeText =  QtWidgets.QLabel(f"Time: {time:.2f} s\n Frames: {frames}")
        delayText = QtWidgets.QLabel(f"Offset\n {delay:.4f} s")
        costText = QtWidgets.QLabel(f"Error\n {cost:.2f}")

        for txt in [timeText, delayText, costText]:
            txt.setAlignment(QtCore.Qt.AlignCenter)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)

        deleteButton =  QtWidgets.QPushButton()
        deleteButton.setMaximumWidth(40)
        deleteButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogCancelButton))
        deleteButton.clicked.connect(self.item_delete_action(itm))

        editButton = QtWidgets.QPushButton("Edit")
        editButton.setMaximumWidth(40)
        editButton.clicked.connect(self.item_edit_action(itm))


        widgetLayout = QtWidgets.QHBoxLayout()
        widgetLayout.addWidget(timeText)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        widgetLayout.addWidget(line)
        widgetLayout.addWidget(delayText)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        widgetLayout.addWidget(line)

        widgetLayout.addWidget(costText)


        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        widgetLayout.addWidget(line)

        widgetLayout.addWidget(editButton)
        widgetLayout.addWidget(deleteButton)
        widgetLayout.addStretch()

        #widgetLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        widget.setLayout(widgetLayout)  
        itm.setSizeHint(widget.sizeHint()) 

        self.addItem(itm)
        self.setItemWidget(itm, widget)

    def item_delete_action(self, row=None):
        # Delete button
        _row = row
        def _btn_action():
            idx = self.row(_row)
            self.takeItem(idx)
            del self.item_data[idx]
            if self.has_valid_stab():
                self.stab.multi_sync_delete_slice(idx)

        return _btn_action

    def item_edit_action(self, row=None):
        _row = row
        def _btn_action():
            idx = self.row(_row)
            self.edit_offset_handler(idx)
            

        
        return _btn_action

    def edit_offset_handler(self, idx):
        val, success = QtWidgets.QInputDialog.getDouble(self, "Edit sync", "New offset: ", self.item_data[idx]["delay"], decimals=5)
        if success and self.has_valid_stab():
            self.stab.multi_sync_change_offset(idx, val)
            self.update_from_stab_data()

    def item_click_action(self, itm=None):
        self.edit_offset_handler(self.row(itm))

    def update_from_stab_data(self):
        # first, delete everything
        self.clear()

        if type(self.stab) == type(None):
            #print("Stabilizer not initialized")
            return False

        for i in range(len(self.stab.transform_times)):
            time, frames = self.stab.sync_inputs[i]
            delay = self.stab.sync_delays[i]
            cost = self.stab.sync_costs[i]
            self.add_row(time/self.stab.fps, frames, delay, cost)

    def recompute_changed(self):
        print(self.stab)

    def update_stab_instance(self, stab=None):
        self.stab = stab
        self.update_from_stab_data()

    def has_valid_stab(self):
        return type(self.stab) != type(None)

    def UI_reset(self):
        self.stab = None
        self.clear()

class MultiSyncUI(QtWidgets.QWidget):
    def __init__(self, stab=None):
        super().__init__()

        self.stab = stab

        self.sync_list = MultiSyncList(self, self.stab)

        self.main_layout = QtWidgets.QVBoxLayout()

        #self.add_sync_subwidget = QtWidgets.QWidget()
        #self.add_sync_subwidget_layout = QtWidgets.QHBoxLayout()
        #self.add_sync_subwidget.setLayout(self.add_sync_subwidget_layout)


        #self.add_sync_button = QtWidgets.QPushButton("Add sync at time:")
        #self.add_sync_input = QtWidgets.QDoubleSpinBox()
        #self.add_sync_subwidget_layout.addWidget(self.add_sync_button)
        #self.add_sync_subwidget_layout.addWidget(self.add_sync_input)
        #self.main_layout.addWidget(self.add_sync_subwidget)

        #self.add_sync_button.clicked.connect(self.add_sync_button_action)

        self.main_layout.addWidget(self.sync_list)

        #self.recompute_button = QtWidgets.QPushButton("Recompute sync")
        #self.main_layout.addWidget(self.recompute_button)
        #self.recompute_button.clicked.connect(self.recompute_action)

        self.setLayout(self.main_layout)

    def has_valid_stab(self):
        return type(self.stab) != type(None)

    def recompute_action(self):
        if self.has_valid_stab():
            self.stab.multi_sync_compute()

    def add_sync_button_action(self):
        if self.has_valid_stab():
            synctime = self.add_sync_input.value()
            self.stab.multi_sync_add_slice(synctime)

    def update_stab_instance(self, stab=None):
        self.stab = stab
        self.sync_list.update_stab_instance(stab)
        

    def update_from_stab_data(self):
        # first, delete everything
        self.sync_list.update_from_stab_data()

    def UI_reset(self):
        self.sync_list.UI_reset()
        self.stab = None

            

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        fakestab = fakeStabilizer()
        self.multiSyncUI = MultiSyncUI(fakestab)
        self.setCentralWidget(self.multiSyncUI)

        fakestab.full_auto_sync()

        self.multiSyncUI.update_from_stab_data()

        



class fakeStabilizer:
    def __init__(self):
        pass

    def multi_sync_init(self):
        self.transform_times = []
        self.transforms = []
        self.sync_inputs = []
        self.sync_vtimes = []
        self.sync_delays = []
        self.sync_costs = []


    def multi_sync_add_slice(self, slice_frame_start, slicelength = 50, debug_plots = True):
        v1 = (slice_frame_start + slicelength/2) / 30
        d1, cost1, times1, transforms1 = np.random.random(), np.random.random() * 100, np.array([1,2,3]), np.array([3,4,5])

        self.sync_inputs.append((slice_frame_start, slicelength))
        self.transform_times.append(times1)
        self.transforms.append(transforms1)
        self.sync_vtimes.append(v1)
        self.sync_delays.append(d1)
        self.sync_costs.append(cost1)

        return cost1
    
    def multi_sync_delete_slice(self, idx):
        
        if len(self.transform_times) > idx:
            del self.transform_times[idx]
            del self.transforms[idx]
            del self.sync_inputs[idx]
            del self.sync_vtimes[idx]
            del self.sync_delays[idx]
            del self.sync_costs[idx]

            return True

        return False

    def multi_sync_change_offset(self, idx, newoffset=0):
        if len(self.transform_times) > idx:
            self.sync_delays[idx] = newoffset
            return True

        return False

    def multi_sync_compute(self, max_cost = 5, max_fitting_error = 0.02, piecewise_correction = False, debug_plots = True):

        assert len(self.transform_times) == len(self.transforms) == len(self.sync_vtimes) == len(self.sync_delays) == len(self.sync_costs)
        print(len(self.transform_times))
        return True


    def full_auto_sync(self, max_fitting_error = 0.02, debug_plots=True):

        self.multi_sync_init()
        self.fps = 30
        self.num_frames = 30 * 60

        max_sync_cost_tot = 10 # > 10 is nogo

        syncpoints = [] # save where to analyze. list of [frameindex, num_analysis_frames]
        num_frames_analyze = 30
        
        max_sync_cost = max_sync_cost_tot / 30 * num_frames_analyze
        num_frames_offset = int(num_frames_analyze / 2)
        end_delay = 3 # seconds buffer zone
        end_frames = end_delay * self.fps # buffer zone

        num_frames = self.num_frames
        vid_length = num_frames / self.fps

        inter_delay = 13 # second between syncs
        inter_delay_frames = int(inter_delay * self.fps)

        min_slices = 4

        max_slices = 9



        if vid_length < 4: # only one sync
            syncpoints.append([5, max(60, int(num_frames-5-self.fps)) ])

            num_syncs = 1

        elif vid_length < 10: # two points
            first_index = 30
            last_index = num_frames - 30 - num_frames_analyze
            syncpoints.append([first_index, num_frames_analyze])
            syncpoints.append([last_index, num_frames_analyze])

            num_syncs = 2

        else:
            # Analysis starts at first frame, so take this into account
            # Add also motion analysis from logs here
            first_index = end_frames - num_frames_offset
            last_index = num_frames - end_frames - num_frames_offset

            num_syncs = max(min(round((last_index - first_index)/inter_delay_frames), max_slices), min_slices)
            inter_frames_actual = (last_index - first_index) / num_syncs

            for i in range(num_syncs):
                syncpoints.append([round(first_index + i * inter_frames_actual), num_frames_analyze])

        # Analyze these slices

        print(f"Analyzing {num_syncs} slices")

        for frame_index, n_frames in syncpoints:
            self.multi_sync_add_slice(frame_index, n_frames, False)

        success = self.multi_sync_compute(max_fitting_error = max_fitting_error, debug_plots=debug_plots)

        if not success:
            success = self.multi_sync_compute(max_fitting_error = max_fitting_error * 2, debug_plots=debug_plots) # larger bound

        if success:
            print("Auto sync complete")
            return True
        else:
            print("Auto sync failed to converge. Sorry about that")
            return False

if __name__ == "__main__":

    app=QtWidgets.QApplication(sys.argv)
    window=MainWindow()
    window.show()
    app.exec_()
