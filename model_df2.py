import face_recognition
import face_recognition_models
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import os
import cv2
import joblib
import pandas as pd
from datetime import datetime
import csv
import MySQLdb

app = Flask(__name__)
socket = SocketIO(app)
global camera

# MySQL database credentials
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DATABASE = 'dbms'

class CamInput:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.face_recognition_model = face_recognition_models.face_recognition_model_location()

        # Load known faces from trained data
        self.known_face_encodings, self.known_face_names = self.load_trained_model()

        # DataFrame to store recognized names
        self.present_df = pd.DataFrame(columns=['Name'])

        self.face_detection_started = False  

    def load_trained_model(self):
        model_path = 'trained_model.joblib'
        if os.path.exists(model_path):
            trained_data = joblib.load(model_path)
            return trained_data['encodings'], trained_data['names']
        else:
            print("No pre-trained model found. Training the model...")
            return self.train_faces(r"C:\Users\Ayaj Anand\work-space\facepics")
            

    def save_trained_model(self, encodings, names):
        model_path = 'trained_model.joblib'
        trained_data = {'encodings': encodings, 'names': names}
        joblib.dump(trained_data, model_path)

    def train_faces(self, folder_path):
        known_face_encodings = []
        known_face_names = []

        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)

            if os.path.isdir(person_folder):
                image_filenames = [f for f in os.listdir(person_folder) if f.endswith(".jpg") or f.endswith(".png")]

                for image_filename in image_filenames:
                    image_path = os.path.join(person_folder, image_filename)
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=[(0, 0, image.shape[0], image.shape[1])], model=self.face_recognition_model)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

        self.save_trained_model(known_face_encodings, known_face_names)
        return known_face_encodings, known_face_names

    def gen_frames(self):
        while self.camera.isOpened():
            _, frame = self.camera.read()

            if not _:
                break
            else:
                if self.face_detection_started:
                    # Convert the frame from BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Find all face locations and face encodings in the current frame
                    face_locations = face_recognition.face_locations(rgb_frame, model=self.face_recognition_model)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model=self.face_recognition_model)

                    # Loop through each face found in the frame
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Compare with known faces
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"

                        # If a match is found, use the name of the known person
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_names[first_match_index]

                            # Add unique names to the set and DataFrame
                            if name not in self.present_df['Name'].values:
                                #timestamp_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                new_data = {'Name': [name]}
                                new_df = pd.DataFrame(new_data)
                                self.present_df = pd.concat([self.present_df, new_df], ignore_index=True)

                            # Draw a rectangle around the face and display the name
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def close_cam(self):
        self.camera.release()

global cam_obj

@app.route('/')
def home():
    global cam_obj
    cam_obj = CamInput()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(cam_obj.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global cam_obj
    cam_obj.face_detection_started = True
    data = request.get_json()
    global subjName
    subjName = data['subj']
    print(subjName)
    return "Face detection started"

@app.route('/stop', methods=['POST'])
def stop():
    global cam_obj
    # Stop the video capture
    cam_obj.close_cam()

    # Get current date and time
    current_datetime = datetime.now()

    # Extract date and time components
    date_str = current_datetime.strftime('%Y-%m-%d')
    time_str = current_datetime.strftime('%H:%M:%S')

    # Add date and time columns to DataFrame
    cam_obj.present_df['Date'] = date_str
    cam_obj.present_df['Time'] = time_str

    # Save DataFrame to CSV
    csv_filename = f'attendance_{subjName}_{date_str}.csv'
    cam_obj.present_df.to_csv(csv_filename, columns=['Name', 'Date', 'Time'], index=False)

    # Upload CSV to MySQL database
    upload_to_mysql(csv_filename)

    return f"CSV data saved and uploaded to MySQL database."

def upload_to_mysql(csv_file):
    connection = MySQLdb.connect(host=MYSQL_HOST, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db=MYSQL_DATABASE)
    cursor = connection.cursor()
    with open(csv_file, 'r') as file:
        csv_data = csv.reader(file)
        next(csv_data)  # Skip the header
        for row in csv_data:
            cursor.execute(f"INSERT INTO {subjName}_attendance (Name, Date, Time) VALUES (%s, %s, %s)", row)
    connection.commit()
    cursor.close()
    connection.close()

if __name__ == "__main__":
    socket.run(app, allow_unsafe_werkzeug=True, debug=True)
