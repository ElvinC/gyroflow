#from matplotlib import pyplot as plt
import math
import numpy as np
import cv2

class NonlinearStretch:
    """Class for nonlinear stretching of images using cv2 maps
    """
    def __init__(self, in_size = (12,9), out_size = (16,9), safe_area = 0, expo = 2):

        self.in_size = in_size
        self.out_size = out_size
        self.safe_area = safe_area
        self.expo = expo

        self.map1 = np.zeros((out_size[1], out_size[0])) # x coords
        self.map2 = np.zeros((out_size[1], out_size[0])) # y coords

    def set_safe_area(self, safe_area):
        """Set untouched safe area

        Args:
            safe_area (float): Safe area (0-1)
        """
        self.safe_area = min(safe_area, 0.999)

    def set_in_size(self, in_size):
        """Set image input size

        Args:
            in_size (int, int): (width, height)
        """
        self.in_size = in_size

    def set_expo(self, expo = 2):
        """Set nonlinear stretch expo

        Args:
            expo (float): Default value of 2 works fine
        """
        # expo<1: Gets real weird
        # expo=1: Linear stretch
        # expo=2: Similar to superview
        # expo=3: More non-linear 
        # expo>3: Gets wonky 
        self.expo = expo

    def set_out_size(self, out_size):
        """Set image output size

        Args:
            in_size (int, int): (width, height)
        """
        self.out_size = out_size
        self.map1 = np.zeros((out_size[1], out_size[0]))
        self.map2 = np.zeros((out_size[1], out_size[0]))

    def recompute_maps(self):
        """Recompute image maps for the nonlinear stretch operation
           required after any changes to parameters of image sizes
        """
        vertical_scale = self.out_size[1] / self.in_size[1] # Image scaling to match height

        # width of side pillar with no stretching (source image scale)
        pillar_width = (self.out_size[0]/vertical_scale - self.in_size[0]) / 2

        self.map1 = np.tile(np.arange(self.out_size[0]), (self.out_size[1],1))
        # center source image. 0.5 to fix mapping offset
        self.map1 = self.map1 / vertical_scale - pillar_width - 0.5

        # stretch offset computation
        # inspired by https://github.com/banelle/derperview

        # create array of x values normalized to -1 < x < 1
        normalized_xcoords = np.tile(np.arange(self.out_size[0]), (self.out_size[1],1))
        normalized_xcoords = (normalized_xcoords / self.out_size[0] - 0.5) * 2

        val_sign = np.zeros((self.out_size[1], self.out_size[0]))
        val_sign[normalized_xcoords<0] = -1
        val_sign[normalized_xcoords>=0] = 1

        offset_map = ((abs(normalized_xcoords) - self.safe_area) / (1 - self.safe_area))**self.expo

        # reset safe area offset to 0
        offset_map[abs(normalized_xcoords) < self.safe_area] = 0

        # correct sign and scale
        offset_map = np.multiply(offset_map, val_sign) * pillar_width

        self.map1 = self.map1 - offset_map

        # y map for scaling only
        # identity map
        self.map2 = np.tile(np.vstack(np.arange(self.out_size[1])), (1, self.out_size[0]))
        # scale and fix offset
        self.map2 = self.map2 / vertical_scale - 0.5

        # convert to datatype supported by opencv
        self.map1 = self.map1.astype('float32')
        self.map2 = self.map2.astype('float32')


    def compute_remap_val(self, tx, target_width, src_width, safe_area = 0.0, expo = 5):
        x = (float(tx)/ target_width - 0.5) * 2

        blanking = (target_width - src_width) / 2

        sx = tx - blanking # shift source pixels by left blanking

        offset = 0

        if abs(x) >= safe_area:
            offset = ((abs(x) - safe_area) /(1- safe_area))**expo * (-1 if x < 0 else 1) * blanking

        final_px = sx - offset

        return final_px


    def apply_stretch(self, img, show_protected = False):
        """Apply nonlinear stretch to cv2 image

        Args:
            img (np.ndarray): cv2 image
            show_protected (bool, optional): Show safe area. Defaults to False.

        Returns:
            np.ndarray: cv2 image
        """
        out_img = cv2.remap(img, self.map1.astype('float32'), self.map2.astype('float32'), cv2.INTER_CUBIC )

        if show_protected:
            midpoint = out_img.shape[1] / 2
            safe_dist = self.safe_area * out_img.shape[1] / 2
            line1 = int(midpoint + safe_dist)
            line2 = int(midpoint - safe_dist)
            cv2.line(out_img,(line1, 0),(line1,out_img.shape[0]),(255,255,0),2)
            cv2.line(out_img,(line2, 0),(line2,out_img.shape[0]),(255,255,0),2)
        return out_img

    def stretch_save_video(self, inpath, outpath = "stretched.mp4"):
        """Load, stretch, and save video

        Args:
            inpath (string): Input file path
            outpath (str, optional): Output file path. Defaults to "stretched.mp4".
        """
        cap = cv2.VideoCapture(inpath)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.set_in_size((width, height))
        self.recompute_maps()
        print(self.out_size)

        out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, self.out_size)

        # use framecount to prevent weird premature termination bug
        frame_num = 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            frame_num += 1
            if ret:
                superview = self.apply_stretch(frame)
                out.write(superview)

                cv2.imshow('frame',superview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif frame_num > num_frames:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
            




if __name__ == "__main__":

    nonlin = NonlinearStretch(out_size=(1280,720))
    nonlin.set_safe_area(0.4)
    nonlin.set_expo(1)


    nonlin.stretch_save_video("PICT0053.AVI", "outfile.mp4")

    # stretch testframe (4:3) to 16:8 using nonlinear stretch
    #input_img = cv2.imread("testframe.png", cv2.IMREAD_COLOR)
    # nonlin = NonlinearStretch(out_size=(1280,720))
    # nonlin.set_in_size((input_img.shape[1], input_img.shape[0]))
    # nonlin.set_safe_area(0.06)

    # for i in range(0, 40):
    #     print("Heyo")
    #     nonlin.set_expo(i/10)
    #     nonlin.recompute_maps()
        
    #     out_img = nonlin.apply_stretch(input_img, True)
    #     print("Heyo2")

    #     cv2.imshow("img", out_img)
    #     cv2.waitKey(4)
    # cv2.destroyAllWindows()
