import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture("videos/test2.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.82)



while True:
    success,img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            bounding_box = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bounding_box.xmin * iw), int(bounding_box.ymin * ih),\
                   int(bounding_box.width * iw), int(bounding_box.height * ih)

            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f"det:{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 2)
            # Syntax: cv2.putText(image, text, org(coordinates(x,y)), font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS:{int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(10)