import cv2 as cv
import numpy as np

age_model = cv.dnn.readNetFromCaffe("age.prototxt","dex_chalearn_iccv2015.caffemodel")
gender_model = cv.dnn.readNetFromCaffe("gender.prototxt","gender.caffemodel")
detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
FourCC = cv.VideoWriter_fourcc(*"XVID")
result = cv.VideoWriter("ageandgender2.avi", FourCC, 2.5, (640, 480))

cap = cv.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_detect = detector.detectMultiScale(gray_img, 1.1, 4)

    for (x, y, w, h) in face_detect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)
        face_detected = frame[int(y):int(y + h), int(x):int(x + w)]
        face_detected = cv.resize(face_detected, (224, 224))  # as the input for pretrained model
        face_detected_blob = cv.dnn.blobFromImage(face_detected)  # ready for put in  model

        age_model.setInput(face_detected_blob)
        age_result = age_model.forward()
        gender_model.setInput(face_detected_blob)
        gender_result = gender_model.forward()

        indexes = np.array([i for i in range(0, 101)])
        age = round(np.sum(age_result[0] * indexes))
        if np.argmax(gender_result[0]) ==1:
            gender ="male"
        else:
            gender ="female"

        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(frame, " Age:" + str(age), (x - 35, y + h + 20), font, 1.2, (0, 0, 255), 2)
        cv.putText(frame, " Gender: "+ gender, (x + 45, y + h + 20), font, 1.2, (0, 0, 255), 2)


        cv.imshow("Frame", frame)
        result.write(frame)

    if cv.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
cv.destroyAllWindows()