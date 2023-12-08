import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import imutils

#url = "http://100.102.143.116:8080/shot.jpg"
cap = cv2.VideoCapture(0)
yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg.txt')

classes = []
with open('coco.names.txt', 'r', ) as f:
    classes = f.read().splitlines()


frame_counter =0
output_counter =[]

while True:
    '''
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    '''
    ret, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    height, width = img.shape[:2]
    '''
    #to print img
    i = blob[0].reshape(320, 320, 3)
    plt.imshow(i)
    '''

    yolo.setInput(blob)
    output_layer_names = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layer_names)


    boxes=[]
    confidences=[]
    class_ids=[]


    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id=np.argmax(score)
            confidence=score[class_id]
            #if confidence.size >0:
            if confidence>0.7:
                center_x=int(detection[0]*width)
                center_y = int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    #colors = np.random.uniform(0,255,size=(len(boxes),3))

    c=0
    #if len(indexes)>0:
    for i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i],2))
        #color = colors[i]
        if label == 'person':
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,None, (x,y+20),font,2,(255,255,255),2)
            c+=1
    if frame_counter % 60 ==0:
        output_counter.append(c)

    cv2.putText(img, 'PEOPLE COUNTER: ' + str(c), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame',img)
    frame_counter +=1

    if cv2.waitKey(1) == 13:
        print(output_counter)
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

