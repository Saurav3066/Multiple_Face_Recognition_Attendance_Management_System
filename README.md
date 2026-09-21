## Multiple Face Recognition Attendance System using Deep Learning

This repository contains Python code for a Multiple Face Recognition Attendance System using Deep Learning. The system allows for new registration of students, dataset creation, model training, retraining, and attendance marking. It utilizes the Tkinter library for the GUI, OpenCV for image processing, and deep learning models for face recognition.

### Features:
- **New Registration:** Allows adding new students to the system by capturing their images and saving them along with their information.
- **Dataset Creation:** Provides functionality to create a dataset by capturing images of students for training the recognition model.
- **Model Training:** Trains the recognition model using the captured dataset.
- **Retraining:** Allows retraining of the model using existing embeddings.
- **Attendance Marking:** Marks attendance by recognizing faces in real-time using the trained model.

### Instructions for Use:
1. **New Registration:** Click on the "New Registration" button to register new students. Enter the name and USN (University Serial Number) in uppercase, then capture images for dataset creation.
2. **Dataset Creation:** After entering the student's details, click on "Create Dataset" to capture images for training.
3. **Model Training:** Once the dataset is created, click on "Train Model" to train the recognition model using the captured images.
4. **Retraining:** Use "Retrain Model" if you want to retrain the model with updated data.
5. **Attendance Marking:** Click on "Attendance Marking" to start marking attendance. The system will recognize faces in real-time and mark attendance accordingly.

### Dependencies:
- Python 3.x
- Tkinter
- OpenCV
- NumPy
- scikit-learn
- imutils
- Pillow

### How to Run:
1. Ensure all dependencies are installed.
2. Run the Python script containing the provided code.
3. Follow the on-screen instructions to perform various tasks like registration, dataset creation, and attendance marking.

### Additional Notes:
- Ensure proper lighting and camera setup for accurate face recognition.
- Adjust the confidence threshold as needed for optimal recognition performance.
- Make sure to provide correct inputs and follow the instructions carefully for each operation.

For any issues or inquiries, feel free to contact the repository owner SAURAV ANAND on (contact.vikrant3066@gmail.com).

**Disclaimer:** This system is provided as-is without any warranty. Use at your own risk.


Project Structure Diagram:

flowchart TD

subgraph group_interface["User Interface"]
  node_gui["Tkinter GUI"]
end

subgraph group_enrollment["Enrollment"]
  node_registration["Registration Flow"]
  node_dataset_capture["Dataset Capture"]
end

subgraph group_learning["Model Learning"]
  node_training["Model Training"]
  node_retraining["Model Retraining"]
  node_face_detector["Face Detector<br/>[deploy.prototxt]"]
  node_detector_weights["Detector Weights"]
  node_embedder["Face Embedder"]
end

subgraph group_recognition["Recognition"]
  node_attendance["Attendance Marking"]
end

subgraph group_storage["Persistence"]
  node_dataset[("Captured Dataset")]
  node_student_records[("Student Records<br/>[student.csv]")]
  node_embeddings[("Embeddings<br/>[embeddings.pickle]")]
  node_recognizer_artifacts[("Recognizer Artifacts<br/>[recognizer.pickle]")]
  node_attendance_csv[("Attendance CSV")]
end

node_user(("User"))
node_camera(("Webcam"))

node_user -->|"uses"| node_gui
node_gui -->|"opens"| node_registration
node_gui -->|"starts"| node_attendance
node_registration -->|"starts"| node_dataset_capture
node_dataset_capture -->|"captures frames"| node_camera
node_dataset_capture -->|"detects faces"| node_face_detector
node_dataset_capture -->|"saves images"| node_dataset
node_dataset_capture -->|"appends details"| node_student_records
node_gui -->|"starts training"| node_training
node_gui -.->|"starts retraining"| node_retraining
node_training -->|"reads images"| node_dataset
node_training -->|"loads detector"| node_face_detector
node_training -->|"loads weights"| node_detector_weights
node_training -->|"loads model"| node_embedder
node_training -->|"writes vectors"| node_embeddings
node_retraining -.->|"reads vectors"| node_embeddings
node_retraining -.->|"writes artifacts"| node_recognizer_artifacts
node_attendance -->|"captures frames"| node_camera
node_attendance -->|"detects faces"| node_face_detector
node_attendance -->|"creates vectors"| node_embedder
node_attendance -->|"loads predictor"| node_recognizer_artifacts
node_attendance -->|"looks up USN"| node_student_records
node_attendance -->|"writes attendance"| node_attendance_csv

click node_gui "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/attendance_management_system.py"
click node_registration "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/attendance_management_system.py"
click node_dataset_capture "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/attendance_management_system.py"
click node_student_records "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/student.csv"
click node_training "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/attendance_management_system.py"
click node_retraining "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/attendance_management_system.py"
click node_face_detector "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/model/deploy.prototxt"
click node_detector_weights "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/model/res10_300x300_ssd_iter_140000.caffemodel"
click node_embedder "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/openface_nn4.small2.v1.t7"
click node_embeddings "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/output/embeddings.pickle"
click node_recognizer_artifacts "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/output/recognizer.pickle"
click node_attendance "https://github.com/saurav3066/multiple_face_recognition_attendance_management_system/blob/main/attendance_management_system.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_gui,node_user toneBlue
class node_registration,node_dataset_capture toneAmber
class node_training,node_retraining,node_face_detector,node_detector_weights,node_embedder toneMint
class node_attendance toneRose
class node_dataset,node_student_records,node_embeddings,node_recognizer_artifacts,node_attendance_csv,node_camera toneIndigo
