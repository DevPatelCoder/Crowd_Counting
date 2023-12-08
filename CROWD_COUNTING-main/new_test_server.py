# Start with a basic flask app webpage.
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context
from threading import Thread, Event
import cv2
import numpy as np


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

# random number Generator Thread
thread = Thread()
thread_stop_event = Event()


def randomNumberGenerator():
    cap = cv2.VideoCapture('in.avi')

    yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg.txt')

    with open('coco.names.txt', 'r', ) as f:
        classes = f.read().splitlines()

    while cap.isOpened():

        ret, img = cap.read()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        height, width = img.shape[:2]

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
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        # colors = np.random.uniform(0,255,size=(len(boxes),3))
        number=0
        if len(detectionNMS) > 0:
            for i in detectionNMS.flatten():
                label = str(classes[class_ids[i]])
                if label == 'person':
                    number += 1
                '''
                else:
                    number=0
                '''
                print(number)
                socketio.emit('newnumber', {'number': number}, namespace='/test')
                socketio.sleep(0)


@app.route('/')
def index():
    # only by sending this page first will the client be connected to the socketio instance
    return render_template('new_index.html')


@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    # Start the random number generator thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(randomNumberGenerator)


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')
    socketio.stop()


if __name__ == '__main__':
    socketio.run(app)
