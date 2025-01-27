import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

ak_img = face_recognition.load_image_file("photos/ak.jpg")
ak_encoding = face_recognition.face_encodings(ak_img)[0]

apj_img= face_recognition.load_image_file("photos/apj.jpg")
apj_encoding = face_recognition.face_encodings(apj_img)[0]

arjun_img = face_recognition.load_image_file("photos/arjun.jpg")
arjun_encoding = face_recognition.face_encodings(arjun_img)[0]

dhoni_img = face_recognition.load_image_file("photos/dhoni.jpg")
dhoni_encoding = face_recognition.face_encodings(dhoni_img)[0]

dulquer_img = face_recognition.load_image_file("photos/dulquer.jpg")
dulquer_encoding = face_recognition.face_encodings(dulquer_img)[0]

elon_img = face_recognition.load_image_file("photos/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_img)[0]

harshad_img = face_recognition.load_image_file("photos/harshad.jpg")
harshad_encoding = face_recognition.face_encodings(harshad_img)[0]

tesla_img = face_recognition.load_image_file("photos/tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_img)[0]

known_face_encodings = [
    ak_encoding, apj_encoding, arjun_encoding, dhoni_encoding,
    dulquer_encoding, elon_encoding, harshad_encoding, tesla_encoding
]
known_face_names = [
    "Ajith Kumar", "Apj Abdul Kalam", "Arjun Mehta", "M S Dhoni",
    "Dulquer Salman", "Elon Musk", "Harshad Mehta", "Tesla"
]

# Copy list for tracking room mates
room_mates = known_face_names.copy()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
s = True

# Create a text file for attendance with the current date
now = datetime.now()
curr_date = now.strftime("%Y-%m-%d")
attendance_file = open(curr_date + '.txt', 'w')

# Add a header to the text file
attendance_file.write("Attendance Record\n")
attendance_file.write("=================\n")

# Start video processing loop
while True:
    # Capture frame from webcam
    _, frame = video_capture.read()
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert to RGB

    # Process every frame for face recognition
    if s:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            # Compare faces and find matches
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            if len(matches) > 0:
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)
            # Record attendance if the person is recognized and not already marked
            if name in known_face_names and name in room_mates:
                room_mates.remove(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                # Write to the text file
                attendance_file.write(f"{name} - {current_time}\n")
                print(f"{name} marked present at {current_time}")

    # Display the video feed with face recognition
    cv2.imshow("Attendance System", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Close the attendance file
attendance_file.close()
