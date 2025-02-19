import cv2
import numpy as np

def estimate_age(image_path):
    # Model files required: deploy_age.prototxt, age_net.caffemodel, deploy_face.prototxt, res10_300x300_ssd_iter_140000.caffemodel
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    # Load face detector
    face_net = cv2.dnn.readNetFromCaffe('deploy_face.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    # Load age estimation model
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            i = age_preds[0].argmax()
            age = age_list[i]
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, f"Age: {age}", (startX, startY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Age Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    estimate_age("face.jpg")
