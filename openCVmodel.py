import cv2
from fer import FER
import numpy as np

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize the emotion detector
emotion_detector = FER()

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)


def detect_bounding_box_and_emotions(vid):
    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Draw rectangles around detected faces
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the detected face
        face_region = vid[y:y + h, x:x + w]

        # Detect emotion in the face region
        emotion, score = emotion_detector.top_emotion(face_region)

        if emotion is not None:
            # Put text of the detected emotion on the video frame
            cv2.putText(vid, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return faces


while True:
    # Capture frame-by-frame
    result, video_frame = video_capture.read()
    if not result:
        break  # Exit the loop if the frame is not read successfully

    # Detect faces and emotions in the frame
    faces = detect_bounding_box_and_emotions(video_frame)

    # Display the processed frame
    cv2.imshow("Face and Emotion Detection", video_frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
