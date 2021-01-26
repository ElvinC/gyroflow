#https://stackoverflow.com/questions/45811421/python-create-image-with-new-camera-position
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


f = 500
rotXval = 90
rotYval = 90
rotZval = 90
distXval = 500
distYval = 500
distZval = 500

def onFchange(val):
    global f
    f = val
def onRotXChange(val):
    global rotXval
    rotXval = val
def onRotYChange(val):
    global rotYval
    rotYval = val
def onRotZChange(val):
    global rotZval
    rotZval = val
def onDistXChange(val):
    global distXval
    distXval = val
def onDistYChange(val):
    global distYval
    distYval = val
def onDistZChange(val):
    global distZval
    distZval = val


from scipy.spatial.transform import Rotation
if __name__ == '__main__':

    #Read input image, and create output image
    src = cv2.imread('goldstone.jpg')
    src = cv2.resize(src,(640,480))
    dst = np.zeros_like(src)
    h, w = src.shape[:2]

    #Create user interface with trackbars that will allow to modify the parameters of the transformation
    wndname1 = "Source:"
    wndname2 = "WarpPerspective: "
    cv2.namedWindow(wndname1, 1)
    cv2.namedWindow(wndname2, 1)
    cv2.createTrackbar("f", wndname2, f, 1000, onFchange)
    cv2.createTrackbar("Rotation X", wndname2, rotXval, 180, onRotXChange)
    cv2.createTrackbar("Rotation Y", wndname2, rotYval, 180, onRotYChange)
    cv2.createTrackbar("Rotation Z", wndname2, rotZval, 180, onRotZChange)
    cv2.createTrackbar("Distance X", wndname2, distXval, 1000, onDistXChange)
    cv2.createTrackbar("Distance Y", wndname2, distYval, 1000, onDistYChange)
    cv2.createTrackbar("Distance Z", wndname2, distZval, 1000, onDistZChange)

    #Show original image
    cv2.imshow(wndname1, src)

    k = -1
    while k != 27:

        




        if f <= 0: f = 1
        rotX = (rotXval - 90)*np.pi/180
        rotY = (rotYval - 90)*np.pi/180
        rotZ = (rotZval - 90)*np.pi/180
        distX = distXval - 500
        distY = distYval - 500
        distZ = distZval - 500

        combined_rotation = np.eye(4)
        #combined_rotation[0:3,0:3] = Rotation.from_euler('xyz', [eul[0], eul[1], -eul[2]], degrees=False).as_matrix()
        combined_rotation[0:3,0:3] = Rotation.from_euler('xyz', [rotX, rotY, rotZ], degrees=False).as_matrix() #Rotation([-quart[1],-quart[2],quart[3],-quart[0]]).as_matrix()
        #eul = Rotation(quart).as_euler('xyz')[0]

        # Camera intrinsic matrix
        K = np.array([[f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0,   1, 0]])

        # K inverse
        Kinv = np.zeros((4,3))
        Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*f
        Kinv[-1,:] = [0, 0, 1]

        # Rotation matrices around the X,Y,Z axis
        RX = np.array([[1,           0,            0, 0],
                    [0,np.cos(rotX),-np.sin(rotX), 0],
                    [0,np.sin(rotX),np.cos(rotX) , 0],
                    [0,           0,            0, 1]])

        RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
                    [            0, 1,            0, 0],
                    [ -np.sin(rotY), 0, np.cos(rotY), 0],
                    [            0, 0,            0, 1]])

        RZ = np.array([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                    [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                    [            0,            0, 1, 0],
                    [            0,            0, 0, 1]])

        # Composed rotation matrix with (RX,RY,RZ)
        R = np.eye(4,4)
        R = combined_rotation

        
        
        

        # Translation matrix
        T = np.array([[1,0,0,distX],
                    [0,1,0,distY],
                    [0,0,1,distZ],
                    [0,0,0,1]])

        # Overall homography matrix
        H = np.linalg.multi_dot([K, R, T, Kinv])

        #print(Kinv)


        # Apply matrix transformation
        cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0)

        # Show the image
        cv2.imshow(wndname2, dst)
        k = cv2.waitKey(1)