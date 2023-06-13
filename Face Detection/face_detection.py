import cv2
import numpy as np

# Load the pre-trained cascade classifier for face detection
cascade_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_file)

# Open the video capture
cap = cv2.VideoCapture(0)

# Define the codec for video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create a video writer object to save the processed frames
output_filename = 'output.avi'
output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame with face detections
    cv2.imshow('Face Detection', frame)

    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, video writer, and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()
