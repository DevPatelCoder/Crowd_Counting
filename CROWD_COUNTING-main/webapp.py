import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import requests
# import imutils
import streamlit as st
import tempfile


def detect(cap, stop):
    output_frame = st.image([])
    # url = "http://192.168.122.208:8080/shot.jpg"
    yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg.txt')
    classes = []
    with open('coco.names.txt', 'r', ) as f:
        classes = f.read().splitlines()

    frame_counter = 0
    # output_counter =[]

    while not stop:
        # '''
        # img_resp = requests.get(url)
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # img = cv2.imdecode(img_arr, -1)
        # img = imutils.resize(img, width=1000, height=1800)
        # '''
        ret, img = cap.read()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        height, width = img.shape[:2]
        # '''
        ##to print img
        # i = blob[0].reshape(320, 320, 3)
        # plt.imshow(i)
        # '''

        yolo.setInput(blob)
        output_layer_names = yolo.getUnconnectedOutLayersNames()
        layeroutput = yolo.forward(output_layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layeroutput:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                # if confidence.size >0:
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        # colors = np.random.uniform(0,255,size=(len(boxes),3))

        c = 0
        # if len(indexes)>0:
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            # color = colors[i]
            if label == 'person':
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, None, (x, y + 20), font, 2, (255, 255, 255), 2)
                c += 1
        # if frame_counter % 60 ==0:
        #    output_counter.append(c)

        cv2.putText(img, 'PEOPLE COUNTER: ' + str(c), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow('frame',img)
        output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_frame.image(output_img)
        # frame_counter +=1

        if cv2.waitKey(1) == 13:
            st.write('Stopped')
            break


st.title('CROWD COUNTING')
input_type = st.selectbox('Select Input type: ', ['Select', 'Use Webcam', 'Upload a video'])

if input_type == 'Use Webcam':
    cap = cv2.VideoCapture(0)
    stop = st.button('Stop')
    detect(cap, stop)
    if stop:
        cap.release()
        input_type = 'Select'

elif input_type == 'Upload a video':
    video_file_buffer = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mov'])
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    if video_file_buffer:
        temp_file.write(video_file_buffer.read())
        # input_video = input_video.write(video_file_buffer.read())
        cap = cv2.VideoCapture(temp_file.name)
        stop = st.button('Stop')
        detect(cap, stop)
        if stop:
            cap.release()
            input_type = 'Select'

else:
    st.write('Select any option')
