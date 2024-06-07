import os
import time
import face_recognition
import cv2
import numpy as np
import RPi.GPIO as GPIO

TOLERANCE = 0.45
MELODY_GREEN = [(800, 600)]
MELODY_RED = [(400, 200)]
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

GPIO.setup(17,GPIO.OUT) # GREEN
GPIO.setup(27,GPIO.OUT) # RED
GPIO.setup(12, GPIO.OUT) # BUZZER
GPIO.setup(14, GPIO.IN)  # PIR sensor input pin (motion sensor)
GPIO.setup(18, GPIO.OUT) # FOR TRIGGER IN SENSOR
GPIO.setup(24, GPIO.IN) # FOR ECHO IN SENSOR

def distance():
    global prev
    # set Trigger to HIGH
    GPIO.output(18, True)
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(18, False)

    StartTime = time.time()
    StopTime = time.time()

    # save StartTime
    while GPIO.input(24) == 0:
        StartTime = time.time()

    # save time of arrival
    while GPIO.input(24) == 1:
        StopTime = time.time()

    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2#
    print(distance)
    return distance

def melody(melody, pause_duration):
    pwm = GPIO.PWM(12, 440)  # Initialize PWM with a default frequency
    pwm.start(50)  # Start PWM with 50% duty cycle

    try:
        for frequency, duration in melody:
            print(f"Playing Frequency: {frequency} Hz for {duration / 1000.0} seconds")
            pwm.ChangeFrequency(frequency)
            time.sleep(duration / 1000.0)
            pwm.ChangeFrequency(1)  # Set frequency to 1Hz to create a pause without stopping PWM
            time.sleep(pause_duration / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        pass
        pwm.stop()
        # GPIO.cleanup() REMOVED DUE TO THE POTENTIAL EFFECT ON THE LED LAMPS
    pwm.stop()

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

known_face_encodings = []
known_face_names = []

with open('server/users.txt', 'r') as f:
    for line in f:
        name = '_'.join(line.split()[2:])
        image_path = f"server/{name}.jpg"
        if os.path.exists(image_path):
            print('yedi')
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
        else:
            print(image_path)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def run():
    video_capture = cv2.VideoCapture(0)
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if ret:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            if name == "Unknown":
                GPIO.output(27, GPIO.HIGH)
                melody(MELODY_RED, 100)
                #time.sleep(0.3)
                GPIO.output(27, GPIO.LOW)
            else:
                GPIO.output(17, GPIO.HIGH)
                melody(MELODY_GREEN, 100)
                #time.sleep(0.3)
                GPIO.output(17, GPIO.LOW)
            face_names.append(name)
            print(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    # cv2.imshow('Video', frame)
    video_capture.release()
    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    # break

while True:
    if distance() < 50:
        run()
    else:
        continue
# Release handle to the webcam
cv2.destroyAllWindows()
