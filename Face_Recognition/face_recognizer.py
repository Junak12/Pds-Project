import cv2
import numpy as np
import json

if __name__ == "__main__":
    # Initialize the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Try loading the trained model
    try:
        recognizer.read('trainer.yml')
        print("Recognizer model loaded successfully.")
    except cv2.error as e:
        print(f"Error loading recognizer model: {e}")
        exit(1)

    # Load the face cascade classifier
    face_cascade_path = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(face_cascade_path)

    if faceCascade.empty():
        print(f"Error loading face cascade classifier from {face_cascade_path}")
        exit(1)

    # Font for displaying text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize names list
    names = ['None']
    try:
        with open('names.json', 'r') as fs:
            names = json.load(fs)
            names = list(names.values())
        print("Loaded names from 'names.json'.")
    except Exception as e:
        print(f"Error loading 'names.json': {e}")
        exit(1)

    # Initialize the webcam
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error opening video stream.")
        exit(1)

    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    # Set minimum width and height for face detection
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("Starting face recognition... Press 'Esc' to exit.")

    while True:
        ret, img = cam.read()

        if not ret:
            print("Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        # If faces are detected, process each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recognize the face and get the confidence
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence > 51:
                try:
                    name = names[id]
                    confidence = "  {0}%".format(round(confidence))
                except IndexError:
                    name = "Go Train Your Face First"
                    confidence = "N/A"
            else:
                name = "Who are you?"
                confidence = "N/A"

            # Display the name and confidence on the image
            cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Show the processed image
        cv2.imshow('Camera', img)

        # Wait for the 'Esc' key to exit
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    # Release the camera and close all OpenCV windows
    print("\n[INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()
