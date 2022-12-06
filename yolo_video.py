import cv2
import numpy as np
import glob
import random

import time

cap = cv2.VideoCapture("video path")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
prev_frame_time = 0
new_frame_time = 0
a = 0
fps_record=[]
#out = cv2.VideoWriter('/Users/emirysaglam/Desktop/ImageProcessing/yolo/yolo_trained_duba/video_proccesed/golet_3_w_3000.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# hangi nesneleri tanıyosa o nesnelerin isimleri girilmeli
labels = ["Orange","Yellow"]


colors = ["0,140,255","0,255,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1))  # matrisi büyütüyo copy paste gibi

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (416, 234))
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
#                                   optimal  eğtitimdeki
#                                    değer  foto sayısı
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416),swapRB=True, crop=False)


# section 3
    model = cv2.dnn.readNetFromDarknet(".cfg path",".weights path")
    layers = model.getLayerNames()
    output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)

    detection_layers= model.forward(output_layer)

############### nms o1

    ids_list = []
    boxes_list = []
    confidences_list = []

############### end nms o1


    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores= object_detection[5:]
            predicted_id= np.argmax(scores) #max scoreun indexini verir
            confidence= scores[predicted_id]

            if confidence > 0.20:

                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height]) #object_detection[0:4] çok küçük
                (box_center_x,box_center_y,box_width,box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))

                ############## nms 02
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x,start_y, int(box_width), int(box_height)])
                ############## end nms 02

############## nms 03
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5,0.1) #0.5,0.4 ideal ama değiştirlebilir

    cizgi=[]
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label= labels [predicted_id]
        confidence = confidences_list[max_class_id]
    ############## end nms 03

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label2= "{}: {:.2f}".format(label, confidence*100)
        print("predicted obj: {}".format(label2))

        cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),box_color,1)
        cv2.putText(frame,label2,(start_x,start_y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, box_color, 1)
        cizgi.append([int((start_x+end_x)/2),int((start_y+end_y)/2),label])


    print(cizgi)
    if len(cizgi)==2:
        if cizgi[0][2]=="Yellow":
            if cizgi[0][0]>cizgi[1][0]:
                cv2.line(frame, (cizgi[0][0], cizgi[0][1]), (cizgi[1][0], cizgi[1][1]), (0, 255, 0), 2)
                print("orange on the left")
            else:
                cv2.line(frame, (cizgi[0][0], cizgi[0][1]), (cizgi[1][0], cizgi[1][1]), (0, 0, 255), 2)
                print("orange on the right")

        elif cizgi[0][2]=="Orange":
            if cizgi[0][0]>cizgi[1][0]:
                cv2.line(frame, (cizgi[0][0], cizgi[0][1]), (cizgi[1][0], cizgi[1][1]), (0, 0, 255), 2)
                print("orange on the right")

            else:
                cv2.line(frame, (cizgi[0][0], cizgi[0][1]), (cizgi[1][0], cizgi[1][1]), (0, 255, 0), 2)
                print("orange on the left")


    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time


    # putting the FPS count on the frame
    fps_record.append(fps)
    print(" fps is {}".format(fps))
    ave = sum(fps_record) / len(fps_record)
    print("average fps is {}".format(ave))
    cv2.imshow("frame",frame)
    print("#########################")

    #out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


