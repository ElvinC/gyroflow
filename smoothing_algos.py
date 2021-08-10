import numpy as np
import quaternion as quat
import matplotlib.pyplot as plt
import sys, inspect
from PySide2 import QtCore, QtWidgets, QtGui

class SmoothingAlgo:
    def __init__(self, name="nothing"):
        self.name = name

        # Options exposed to the user
        # Each dict value is a list is a dict with [value, min, max, sort_index]
        self.user_options = {}

        self.num_data_points = 0
        self.gyro_sample_rate = 1

        self.ui_widget = None
        self.ui_widget_layout = None
        self.ui_input_widgets = {}
        
    def get_ui_widget(self):
        if self.ui_widget:
            return self.ui_widget

        self.ui_widget = QtWidgets.QWidget()
        self.ui_widget_layout = QtWidgets.QVBoxLayout()
        self.ui_widget.setLayout(self.ui_widget_layout)
        self.ui_widget_layout.setAlignment(QtCore.Qt.AlignTop)

        options = self.get_user_option_all()
        for option in options:
            input_type = option["input_type"]
            optionname = option["name"]
            value = option["value"]
            if input_type == "slider":
                ui_input = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ui_widget)
                steps = 100
                conv_func = self.get_slider_conv_func(option["min"], option["max"], steps, option["slider_expo"])

                initial = 20
                ui_input.setMinimum(0)
                ui_input.setMaximum(steps)
                ui_input.setValue(initial)

                ui_input.setSingleStep(1)
                ui_input.setTickInterval(1)

                initial_string = option["ui_label"].format(conv_func(initial))
                ui_label = QtWidgets.QLabel(initial_string)
                
                ui_input.valueChanged.connect(self.widget_input_update(optionname))


            elif input_type == "int":
                ui_input = QtWidgets.QSpinBox(self.ui_widget)

                conv_func = lambda val: val

                ui_input.setMinimum(option["min"])
                ui_input.setMaximum(option["max"])
                ui_input.setValue(value)

                initial_string = option["ui_label"].format(conv_func(initial))
                ui_label = QtWidgets.QLabel(initial_string)
                
                ui_input.valueChanged.connect(self.widget_input_update(optionname))


            elif input_type == "float":
                ui_input = QtWidgets.QDoubleSpinBox(self.ui_widget)

                conv_func = lambda val: val

                ui_input.setMinimum(option["min"])
                ui_input.setMaximum(option["max"])
                ui_input.setValue(value)

                initial_string = option["ui_label"].format(conv_func(initial))
                ui_label = QtWidgets.QLabel(initial_string)
                
                ui_input.valueChanged.connect(self.widget_input_update(optionname))

            ui_input.setToolTip(option["explanation"])

            self.ui_input_widgets[optionname] = [ui_input, ui_label, conv_func]

            self.ui_widget_layout.addWidget(ui_label)
            self.ui_widget_layout.addWidget(ui_input)

        return self.ui_widget

    def get_slider_conv_func(self, minval, maxval, steps, expo):
        return lambda val: (val/steps)**expo * (maxval - minval)+minval




    def widget_input_update(self, optionname = ""):
        #print("Update option")
        _self = self
        def innerfunc():
            val = _self.ui_input_widgets[optionname][0].value()
            label = _self.ui_input_widgets[optionname][1]
            conv_func = _self.ui_input_widgets[optionname][2]

            conv_val = conv_func(val)

            new_text = _self.get_user_option(optionname)["ui_label"].format(conv_val)
            label.setText(new_text)

            _self.set_user_option(optionname, conv_val)
            #print(self.user_options[optionname])

        return innerfunc

    def preview_widget(self):
        QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        app = QtWidgets.QApplication([])
        widget = self.get_ui_widget()



        widget.resize(500, 500)

        widget.show()
        
        sys.exit(app.exec_())

    def set_user_option(self, optionname, value):
        if optionname in self.user_options:
            self.user_options[optionname]["value"] = value
        else:
            print(f"{optionname} is not a valid option")

        self.update_after_user_option()

    def update_after_user_option(self):
        # to be overloaded
        pass

    def add_user_option(self, optionname, value, minval, maxval, ui_label = "Option {0}", explanation="", input_type="slider", input_expo = 1, sort_index = None):
        if sort_index == None:
            sort_index = len(self.user_options)
        self.user_options[optionname] = {
            "name": optionname, # also main ID
            "ui_label": ui_label, # Label template
            "value": value, # Default value
            "min": minval, # set to -1 to disable limit
            "max": maxval,
            "explanation": explanation, # Tooltip
            "input_type": input_type, # "slider" or "float" or "int"
            "slider_expo": input_expo,
            "sort_index": sort_index
        }

    def get_user_option(self, optionname):
        if optionname in self.user_options:
            return self.user_options[optionname]

    def get_user_option_value(self, optionname):
        return self.get_user_option(optionname)["value"]

    def get_user_option_all(self):
        """Get all exposed options as a list

        Returns:
            list: sorted list with elements of dict containing {name, value, min, max, sort_index}
        """

        retval = [self.user_options[key] for key in self.user_options]
        retval.sort(key = lambda itm: itm["sort_index"])
        return retval

    def get_smooth_orientations(self, times, orientation_quats):
        # Some stuff before processing
        self.num_data_points = times.shape[0]
        self.gyro_sample_rate = self.num_data_points / (times[-1] - times[0])

        return self.smooth_orientations_internal(times, orientation_quats)

    def smooth_orientations_internal(self, times, orientation_quats):
        # To be overloaded
        return times, orientation_quats




class PlainSlerp(SmoothingAlgo):
    """Default symmetrical quaternion slerp without limits
    """
    def __init__(self):
        super().__init__("Plain quaternion slerp")

        self.add_user_option("smoothness", 0.2, 0, 30, ui_label = "Smoothness (time constant: {0:.3f} s):",
                             explanation="Smoothness time constant in seconds", input_expo = 3, input_type="slider")
        

    def smooth_orientations_internal(self, times, orientation_list):
        # To be overloaded


        # https://en.wikipedia.org/wiki/Exponential_smoothing
        # the smooth value corresponds to the time constant

        alpha = 1
        smooth = self.get_user_option_value("smoothness")
        if smooth > 0:
            alpha = 1 - np.exp(-(1 / self.gyro_sample_rate) /smooth)

        smoothed_orientation = np.zeros(orientation_list.shape)

        value = orientation_list[0,:]


        for i in range(self.num_data_points):
            value = quat.single_slerp(value, orientation_list[i,:],alpha)
            smoothed_orientation[i] = value

        # reverse pass
        smoothed_orientation2 = np.zeros(orientation_list.shape)

        value2 = smoothed_orientation[-1,:]

        for i in range(self.num_data_points-1, -1, -1):
            value2 = quat.single_slerp(value2, smoothed_orientation[i,:],alpha)
            smoothed_orientation2[i] = value2

        # Test rotation lock (doesn't work)
        #if test:
        #    from scipy.spatial.transform import Rotation
        #    for i in range(self.num_data_points):
        #        quat = smoothed_orientation2[i,:]
        #        eul = Rotation([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        #        new_quat = Rotation.from_euler('xyz', [eul[0], eul[1], np.pi]).as_quat()
        #        smoothed_orientation2[i,:] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        return times, smoothed_orientation2

class LimitedSlerp(SmoothingAlgo):
    """Default symmetrical quaternion slerp with limits
    """
    def __init__(self):
        super().__init__("Quaternion slerp with experimental angle limit")

        self.add_user_option("smoothness", 0.2, 0, 30, ui_label = "Smoothness (time constant: {0:.3f} s):",
                             explanation="Smoothness time constant in seconds", input_expo = 3, input_type="slider")

        self.add_user_option("rotlimit", 15, 0, 180, ui_label = "Rotation limit (degrees):",
                             explanation="Maximum angular rotation for virtual camera", input_expo = 1, input_type="int")

        #self.add_user_option("limitslope", 10, 0, 180, ui_label = "Limit slope. time constant per radian:",
        #                     explanation="Maximum angular rotation for virtual camera", input_expo = 1, input_type="int")

    def smooth_orientations_internal(self, times, orientation_list):
        # To be overloaded


        # https://en.wikipedia.org/wiki/Exponential_smoothing
        # the smooth value corresponds to the time constant

        alpha = 1
        smooth = self.get_user_option_value("smoothness")
        print(f"Smoothing orientation with smoothness={smooth}")
        smooth2 = min(smooth * 0.1, 0.1)  # When outside zone 
        alpha2 = 1 - np.exp(-(1 / self.gyro_sample_rate) /smooth2)
        
        if smooth > 0:
            alpha = 1 - np.exp(-(1 / self.gyro_sample_rate) /smooth)
        

        smoothed_orientation = np.zeros(orientation_list.shape)

        value = orientation_list[0,:]

        rotlimit = self.get_user_option_value("rotlimit") * np.pi / 180

        begin_curve = rotlimit * 0.6

        # Forward pass
        for i in range(self.num_data_points):
            temp_value = quat.single_slerp(value, orientation_list[i,:],alpha)
            anglebetween = abs(quat.angle_between(temp_value, orientation_list[i,:]))
            if begin_curve < anglebetween <= rotlimit:
                smoothinterp = smooth + (anglebetween - begin_curve) * (smooth2 - smooth) / (rotlimit - begin_curve)
                
                alphainterp = 1 - np.exp(-(1 / self.gyro_sample_rate) /smoothinterp)
                temp_value = quat.single_slerp(value, orientation_list[i,:],alphainterp)
            
            elif anglebetween > rotlimit: # new smoothed orientation over angle limit
                temp_value = quat.single_slerp(value, orientation_list[i,:],alpha2)

            value = temp_value
            smoothed_orientation[i] = value

        # reverse pass
        smoothed_orientation2 = np.zeros(orientation_list.shape)

        value2 = smoothed_orientation[-1,:]

        for i in range(self.num_data_points-1, -1, -1):
            temp_value2 = quat.single_slerp(value2, smoothed_orientation[i,:],alpha)
            anglebetween = abs(quat.angle_between(temp_value2, orientation_list[i,:]))
            #print(anglebetween, rotlimit)
            if begin_curve < anglebetween <= rotlimit:
                smoothinterp = smooth + (anglebetween - begin_curve) * (smooth2 - smooth) / (rotlimit - begin_curve)
                alphainterp = 1 - np.exp(-(1 / self.gyro_sample_rate) /smoothinterp)
                temp_value2 = quat.single_slerp(value2, smoothed_orientation[i,:],alphainterp)
            
            elif anglebetween > rotlimit: # new smoothed orientation over angle limit
                temp_value2 = quat.single_slerp(value2, smoothed_orientation[i,:],alpha2)
            
            value2 = temp_value2
            smoothed_orientation2[i] = value2

        return times, smoothed_orientation2


#class HorizonLock(SmoothingAlgo):
#    """Keep horizon level
#    """
#    def __init__(self):
#        super().__init__("Lock horizon (TODO)")

#        self.add_user_option("smoothness", 0.2, 0, 30, ui_label = "Smoothness (time constant: {0:.3f} s):",
#                             explanation="Smoothness time constant in seconds", input_expo = 3, input_type="slider")
#
#
#    def smooth_orientations_internal(self, times, orientation_list):
#        pass

smooth_algo_classes = []

for n, obj in inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__):
    if inspect.isclass(obj):
        smooth_algo_classes.append(obj)

smooth_algo_names = [alg().name for alg in smooth_algo_classes]

def get_stab_algo_names():
    """List of available control algorithms in plaintext
    """
    return smooth_algo_names

def get_all_stab_algo_instances():
    return [alg() for alg in smooth_algo_classes]

def get_stab_algo_by_name(name="nothing"):
    """Get an instance of a smoothing algorithm class from name
    """
    if name in smooth_algo_names:
        return smooth_algo_classes[smooth_algo_names.index(name)]()
    else:
        return None


if __name__ == "__main__":
    testalgo = LimitedSlerp()
    testalgo.set_user_option("smoothness", 5)
    np.random.seed(22323)
    testquats = np.random.random((100, 4))
    for i in range(testquats.shape[0]):
        testquats[i,:] /= np.linalg.norm(testquats[i,:])

    testimes = np.arange(0,10,0.1)
    times, quats = testalgo.get_smooth_orientations(testimes, testquats)
    #print(testquats)
    #print(quats)
    plt.plot(testquats[:,0])
    plt.plot(quats[:,0])
    plt.show()