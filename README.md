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
