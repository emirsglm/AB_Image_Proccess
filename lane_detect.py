import numpy as np
import cv2
import matplotlib.pyplot as plt 


path = "/Users/emirysaglam/Documents/GitHub/AB_Image_Proccess/calib/1.png"

def canny(img,min_tresh,max_tresh):
    
    lane_img = np.copy(img)
    lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
    #lane_img = cv2.GaussianBlur(lane_img, (3,3),0)
    #              lower tresh, upper tresh  ratio 1/2 or 1/3
    lane_img = cv2.Canny(lane_img,min_tresh,max_tresh)
    return lane_img



#roi (224,224) --> base:(40,170) , height:(104,100)
#left = 40
#right = 170
#up = ((104,100))
#tresh = 10


def roi(img,left,right,up):
    height = img.shape[0]

    #define triangular roi  (left corner , right corner , height)
    polygons = np.array([
        [(left,height),(right,height),up]
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

def steer(img,lines,tresh):

    lc_point = int((lines[0][0] + lines[0][2])/2) 
    rc_point = int((lines[1][0] + lines[1][2])/2)
    cc_point = int((lc_point+ rc_point)/2)
    frame_center = int(img.shape[1]/2)

    if cc_point > frame_center + tresh:
        print("steer left")
    
    elif tresh - 10 > cc_point:
        print("steer right")

    else:
        print("straight")

    #burayı düzelt

    return cc_point


img = cv2.imread(path)
img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_AREA)


# bu uc parametre yolu icine alan ucgenin kose koordinatları
# left ucgenin sol alt kosesinin x eksenindeki yeri
# right ucgenin sag alt kosesinin x eksenindeki yeri
# up ise yüksekliginin sirayla x , y koordinatindaki yerleri
left = 130    
right = 1140
up = (600,280)

# tresh sağ sol outputu icin eşik degeri
tresh = 50

# miin_tresh max_tresh canny fonksiyonu için esik degerleri 1/2 ya da 1/3 oranında olmali
min_tresh = 70
max_tresh = 140


canny_img = canny(img,min_tresh,max_tresh)
cv2.imshow("canny",canny_img)

cropped = roi(canny_img,left,right,up)
#roi ven canny nin sırasını değiştirmeyi dene
cv2.imshow("cropped",cropped)

lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 180, np.array([]), minLineLength=30, maxLineGap=10)

averaged_lines = ave_slope_intercept(img,lines)
print(averaged_lines)

lines_img = display_lines(img,averaged_lines)
final_out = cv2.addWeighted(img, 0.8, lines_img, 1,1)


cv2.line(final_out, (int(img.shape[1]/2),0),(int(img.shape[1]/2),img.shape[0]),(0,255,0),1)

steer(final_out,averaged_lines,tresh)

cv2.imshow("final",final_out)
cv2.waitKey(0)





#cap = cv2.VideoCapture(path)

"""# resizing image for faster runtime
scale_percent = 20
dim = (int(width * scale_percent / 100), int(height * scale_percent / 100))
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
"""
"""
while True:
    ret,frame = cap.read()
    if not ret:
        break
    

    left = 270
    right = 2400
    up = (1278,590)
    tresh = 200
    min_tresh = 150
    max_tresh = 300
    
    #plt.imshow() lef,right ,up parametrelerini ayarla
    #plt.show()

    canny_img = canny(frame)
    cropped = roi(canny_img,left,right,up)
    #roi ven canny nin sırasını değiştirmeyi dene

    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 180, np.array([]), minLineLength=30, maxLineGap=5)

    averaged_lines = ave_slope_intercept(frame,lines)
    steer(frame,averaged_lines,tresh)

    #lines_img = display_lines(img,averaged_lines)
    #cv2.line(final_out, (int(img.shape[1]/2),0),(int(img.shape[1]/2),244),(0,255,0),1)
    #final_out = cv2.addWeighted(img, 0.8, lines_img, 1,1)   
    #cv2.imshow("final", final_out)
    #cv2.waitKey(0)


#cv2.destroyAllWindows()
"""