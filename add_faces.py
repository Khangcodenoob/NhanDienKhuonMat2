import cv2
import pickle
import numpy as np
import os
import sys

# Get the name from command-line argument
name = sys.argv[1]
print(f"Adding face for {name}")

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Check if the camera is successfully opened
if not video.isOpened():
    print("Error: Could not open video stream.")
    sys.exit()

faces_data = []
i = 0

# Loop to capture face data
while len(faces_data) < 100:  # Continue until we capture 100 images
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture image from camera.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize face to 50x50 pixels

        if len(faces_data) < 100 and i % 10 == 0:  # Skip some frames to avoid redundancy
            faces_data.append(resized_img)

        i += 1

        # Show progress on frame
        cv2.putText(frame, f"Collected: {len(faces_data)}/100", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    # Display frame
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    
    # Break loop if 'q' is pressed or 100 images are captured
    if k == ord('q') or len(faces_data) == 100:
        break

# Release video and close all windows
video.release()
cv2.destroyAllWindows()

# Convert face data to a numpy array
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)  # Flatten each 50x50 face image into a 1D array

# Ensure 'data/' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Save or update 'names.pkl'
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save or update 'faces_data.pkl'
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print(f"Successfully added face data for {name}")
