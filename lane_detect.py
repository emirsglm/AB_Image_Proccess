import numpy as np
import cv2
import matplotlib.pyplot as plt 


path = "/Users/emirysaglam/Desktop/codes/ImageProccesing/codes/u_net/single_class/roadline_data/train_images/image_224(143).png"

def canny(img):
    lane_img = np.copy(img)
    lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
    #lane_img = cv2.GaussianBlur(lane_img, (3,3),0)
    #              lower tresh, upper tresh  ratio 1/2 or 1/3
    lane_img = cv2.Canny(lane_img,300,600)
    return lane_img

#roi (224,224) --> base:(40,170) , height:(104,100)
def roi(img):
    height = img.shape[0]
    
    #define triangular roi  (left corner , right corner , height)
    polygons = np.array([
        [(40,height),(170,height),(104,100)]
        ])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    masked = cv2.bitwise_and(img,mask)

    return masked

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        
        #ave_slope_intercept zaten bunu yapıyo
        #for line in lines:
        #    x1,y1,x2,y2 = line.reshape(4)
        #    cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),4)
        
        for x1,y1,x2,y2  in lines:
            cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),4)

    return line_image

def make_coord(img,line_param):
    slope , intercept = line_param
    y1 = img.shape[0]
    y2 = int(y1*(4/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1,y1,x2,y2])

def ave_slope_intercept(img,lines):
    left_fit = []
    right_fit =[]

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        param = np.polyfit((x1,x2),(y1,y2),1)
        slope = param[0]
        intercept = param[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_ave = np.average(left_fit,axis=0)
    right_fit_ave = np.average(right_fit,axis=0)

    left_line = make_coord(img,left_fit_ave)
    right_line = make_coord(img,right_fit_ave)

    return np.array([left_line,right_line])



img = cv2.imread(path)

canny_img = canny(img)

cropped = roi(canny_img)

lines = cv2.HoughLinesP(cropped, 2, np.pi/10, 10, np.array([]), minLineLength=10, maxLineGap=2)
#print(lines)

averaged_lines = ave_slope_intercept(img,lines)
print(averaged_lines)

lines_img = display_lines(img,averaged_lines)
final_out = cv2.addWeighted(img, 0.8, lines_img, 1,1)


def steer(img,lines):

    lc_point = int((lines[0][0] + lines[0][2])/2) 
    rc_point = int((lines[1][0] + lines[1][2])/2)
    cc_point = int((lc_point+ rc_point)/2)
    frame_center = int(img.shape[1]/2)

    if cc_point > frame_center + 10:
        print("steer left")
    
    elif frame_center - 10 > cc_point:
        print("steer right")

    else:
        print("straight")

    #burayı düzelt

cv2.line(final_out, (int(img.shape[1]/2),0),(int(img.shape[1]/2),244),(0,255,0),1)

steer(final_out,averaged_lines)
cv2.imshow("final",final_out)
cv2.imshow("canny",canny_img)
cv2.waitKey(0)