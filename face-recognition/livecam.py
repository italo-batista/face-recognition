import face_recognition
import cv2
import numpy as np

import glob
import os
import sys
import re


DATA_PATH = "../data"
process_this_frame = True

# Create arrays of known face encodings and their names
known_face_encodings = []
users_ids = []


def is_camera_valid(video_capture):
    return video_capture.read() != None


def add_user_to_train(image_path, id):
    print('Loading image: ' + image_path, id)
    image = face_recognition.load_image_file(image_path)
    
    # Return the 128-dimension face encoding for each face in the image.
    face_encoding = face_recognition.face_encodings(image)

    # If has at least one face, add it to train_data and label to users_ids
    if len(face_encoding) > 0:
        known_face_encodings.append(face_encoding[0])
        users_ids.append(id)


def load_faces():
    # TODO: switch between /faces/ or /train_data/ path depending on execution (if demo or 
    # experiment). This will change how get the user id also (this will depend on how the dataset
    # structure). Implement it also.
    for image_path in glob.glob(DATA_PATH + '/faces/*.jpg'):
        img_name = image_path.split('/')[-1]
        user_id = img_name.split('.jpg')[0]
        add_user_to_train(image_path, user_id)


def draw_rectangle(face_locations, names, frame):
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)


def recognize(frame, process_this_frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
  
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s) in train data
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.6)  #TODO tolerance
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = users_ids[best_match_index]

            face_names.append(name)
        
        # Display the results
        draw_rectangle(face_locations, face_names, frame)

    process_this_frame = not process_this_frame


def main():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    
    # Load a sample pictures and learn how to recognize trem.
    load_faces()
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if is_camera_valid(video_capture):
            recognize(frame, process_this_frame)
        else:
            print('An error occurred in frame. Please check the camera')
            break

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    sys.exit(main())