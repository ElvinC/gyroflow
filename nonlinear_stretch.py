#from matplotlib import pyplot as plt
import numpy as np
import math
import cv2

class NonlinearStretch:
    def __init__(self, in_size = (12,9), out_size = (16,9), safe_area = 0, expo = 2):
        self.in_size = in_size
        self.out_size = out_size
        self.safe_area = safe_area
        self.expo = expo

        self.map1 = np.zeros((out_size[1], out_size[0])) # x coords
        self.map2 = np.zeros((out_size[1], out_size[0])) # y coords

    def set_safe_area(self, safe_area):
        self.safe_area = safe_area

    def set_in_size(self, in_size):
        self.in_size = in_size

    def set_expo(self, expo):
        # expo<1: Gets real weird
        # expo=1: Linear stretch
        # expo=2: Similar to superview
        # expo=3: More non-linear 
        # expo>3: Gets wonky 
        self.expo = expo

    def set_out_size(self, out_size):
        self.out_size = out_size
        self.map1 = np.zeros((out_size[1], out_size[0]))
        self.map2 = np.zeros((out_size[1], out_size[0]))

    def recompute_maps(self):
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


    def compute_remap_val(tx, target_width, src_width, safe_area = 0.0, expo = 5):
        x = (float(tx)/ target_width - 0.5) * 2

        blanking = (target_width - src_width) / 2

        sx = tx - blanking # shift source pixels by left blanking

        offset = 0

        if abs(x) >= safe_area:
            offset = ((abs(x) - safe_area) /(1- safe_area))**expo * (-1 if x < 0 else 1) * blanking

        final_px = sx - offset

        return final_px


    def apply_stretch(self, img, show_protected = False):
        out_img = cv2.remap(img, self.map1.astype('float32'), self.map2.astype('float32'), cv2.INTER_CUBIC )

        if show_protected:
            midpoint = out_img.shape[1] / 2
            safe_dist = self.safe_area * out_img.shape[1] / 2
            line1 = int(midpoint + safe_dist)
            line2 = int(midpoint - safe_dist)
            cv2.line(out_img,(line1, 0),(line1,out_img.shape[0]),(255,255,0),2)
            cv2.line(out_img,(line2, 0),(line2,out_img.shape[0]),(255,255,0),2)
        return out_img


if __name__ == "__main__":


    cap = cv2.VideoCapture("hero5.mp4")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1280,720))
    nonlin = NonlinearStretch(out_size=(1280,720))
    nonlin.set_in_size((width, height))
    nonlin.set_safe_area(0.06)
    nonlin.recompute_maps()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            superview = nonlin.apply_stretch(frame)
            out.write(superview)

            cv2.imshow('frame',superview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #else:
            #print("wait")
            #break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
