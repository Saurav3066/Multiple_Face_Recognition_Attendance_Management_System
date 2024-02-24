import tkinter as tk
import imutils
import time
import cv2
import csv
import os
from tkinter import *
from tkinter import messagebox
from imutils import paths
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from collections.abc import Iterable
import datetime
from tkinter import filedialog, messagebox, simpledialog, Tk
from PIL import ImageTk, Image




def new_registration():
    # Create a new window for dataset creation, model training, and retraining
    registration_window = tk.Toplevel(root)
    registration_window.title("New Registration")
    registration_window.geometry("500x400")
    
    # registration_window.attributes("-topmost", True)
    
    screen_width = registration_window.winfo_screenwidth()
    screen_height = registration_window.winfo_screenheight()
    window_width = 500
    window_height = 400
    x = (screen_width - window_width) // 9
    y = (screen_height - window_height) // 9
    
    registration_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    # Load and resize the background image
    image = Image.open("images/background.jpg")
    image = image.resize((500, 400), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(image)

    # Create a label to display the background image
    background_label = tk.Label(registration_window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Create buttons for dataset creation, model training, and retraining
    button_config = {
        "height": 2,
        "width": 20,
        "font": ("Helvetica", 14),
        "bg": "white",  # Button background color (yellow)
        "fg": "black",  # Button text color
        "bd": 5,  # Border width
        "relief": "raised",  # Border style
        "activebackground": "black",  # Background color when button is pressed
        "activeforeground": "white"  # Text color when button is pressed
    }

    dataset_button = tk.Button(registration_window, text="Create Dataset", command=create_dataset_window, **button_config)
    train_button = tk.Button(registration_window, text="Train Model", command=train_model, **button_config)
    retrain_button = tk.Button(registration_window, text="Retrain Model", command=retrain_model, **button_config)
    close_button = tk.Button(registration_window, text="Close", command=registration_window.destroy, **button_config)

    # Set the spacing between the buttons
    button_spacing = 10
    dataset_button.pack(pady=(50, button_spacing), anchor="center")
    train_button.pack(pady=(0, button_spacing), anchor="center")
    retrain_button.pack(pady=(0, button_spacing), anchor="center")
    close_button.pack(anchor="center")

    # Start the main event loop for the registration window
    registration_window.mainloop()




    
def attendance_marking():
    attendance()

# Create the main window
root = tk.Tk()
root.title("MULTIPLE FACE RECOGNITION ATTENDANCE SYSTEM USING DEEP LEARNING")
root.geometry("800x600")
# root.configure(background="white")


# Load and resize the image using PIL
image = Image.open("images/background.jpg")
image = image.resize((1600, 1080), Image.LANCZOS)
# Set the background image
background_image = ImageTk.PhotoImage(image)
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a heading label
heading_label = tk.Label(root, text="MULTIPLE FACE RECOGNITION ATTENDANCE SYSTEM USING DEEP LEARNING", font=("Helvetica", 24, "bold"))
heading_label.pack(pady=20)

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create the buttons with text and images
new_reg_icon = ImageTk.PhotoImage(file="images/user.png")
new_reg_button = tk.Button(button_frame, text="New Registration", command=new_registration, bd=0, borderwidth=4, relief="raised", font=("Helvetica", 20, "bold"), image=new_reg_icon, compound=tk.TOP)
new_reg_button.pack(side="left", padx=20, pady=10)

attendance_icon = ImageTk.PhotoImage(file="images/roll-call.png")
attendance_button = tk.Button(button_frame, text="Attendance Marking", command=attendance_marking, bd=0, borderwidth=4, relief="raised", font=("Helvetica", 20, "bold"), image=attendance_icon, compound=tk.TOP)
attendance_button.pack(side="right", padx=20, pady=10)

# Create a close button with lambda function
close_button = tk.Button(root, text="Close", command=lambda: root.destroy(), font=("Helvetica", 22))
close_button.pack(side="bottom", pady=20)




#############################---DATASET_CREATION_SECTION---#######################################################################
cascade_path = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)
dataset_dir = 'dataset'
# def create_dataset_dir(name, usn):
#     sub_dir = os.path.join(dataset_dir, name)
#     if not os.path.exists(sub_dir):
#         os.makedirs(sub_dir)
#         return sub_dir
#     else:
#         raise Exception("Dataset directory already exists for this student")
def create_dataset_dir(name, usn):
    sub_dir = os.path.join(dataset_dir, name)
    if not os.path.exists(sub_dir):
        if not check_usn_in_csv(usn):
            os.makedirs(sub_dir)
            return sub_dir
        else:
            raise Exception("USN already exists in the student dataset")
    else:
        raise Exception("Dataset directory already exists for this student")

def check_usn_in_csv(usn):
    with open('student.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if len(row) > 1 and row[1] == usn:
                return True
    return False


def capture_dataset(name, usn):
    sub_dir = create_dataset_dir(name, usn)

    print("Starting video stream...")
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)
    total = 0

    while total < 100:
        _, frame = cam.read()
        img = imutils.resize(frame, width=400)
        rects = face_detector.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = os.path.sep.join([sub_dir, "{}.png".format(str(total).zfill(5))])
            cv2.imwrite(p, img)
            total += 1

        cv2.imshow("Create Dataset", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save student details to CSV file
    info = [name, usn]
    with open('student.csv', 'a') as csvFile:
        write = csv.writer(csvFile)
        write.writerow(info)

    messagebox.showinfo("Student Details", f"Student Details saved successfully for USN: {usn}, Name: {name}")
 

def create_dataset_window():
    window = tk.Toplevel()
    window.title("Create Dataset")
    window.geometry("500x300")
    # window.attributes("-topmost", True)

    # Calculate the center position for the window
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window_width = 500
    window_height = 300
    x = (screen_width - window_width) // 9
    y = (screen_height - window_height) // 9
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Load and resize the background image
    image = Image.open("images/background.jpg")
    image = image.resize((500, 300), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(image)

    # Create a label to display the background image
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    frame = tk.Frame(window, bg="#FFFFFF")  # Frame to hold the contents
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    name_label = tk.Label(frame, text="Enter Name (in upper case)", font=("Helvetica", 12), bg="#FFFFFF", fg="#000000")
    name_label.grid(row=0, column=0, pady=10)

    name_entry = tk.Entry(frame, font=("Helvetica", 12))
    name_entry.grid(row=1, column=0, pady=5)

    usn_label = tk.Label(frame, text="Enter USN (in upper case)", font=("Helvetica", 12), bg="#FFFFFF", fg="#000000")
    usn_label.grid(row=2, column=0, pady=10)

    usn_entry = tk.Entry(frame, font=("Helvetica", 12))
    usn_entry.grid(row=3, column=0, pady=5)

    button_config = {
        "height": 2,
        "width": 15,
        "font": ("Helvetica", 12),
        "bg": "#FF5722",  # Button background color (orange)
        "fg": "white",  # Button text color
        "bd": 0,  # Border width
        "relief": "raised",  # Border style
        "activebackground": "#FF4500",  # Background color when button is pressed (dark orange)
        "activeforeground": "white"  # Text color when button is pressed
    }

    def create_dataset():
        name = name_entry.get().strip()
        usn = usn_entry.get().strip()

        if not name.isupper() or not usn.isupper():
            messagebox.showerror("Error", "Please enter name and USN in upper case")
            return

        try:
            capture_dataset(name, usn)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    create_button = tk.Button(frame, text="Create Dataset", command=create_dataset, **button_config)
    create_button.grid(row=4, column=0, pady=10)

    close_button = tk.Button(frame, text="Close", command=window.destroy, **button_config)
    close_button.grid(row=5, column=0)

    # Start the main event loop for the create dataset window
    window.mainloop()



########################################################---TRAIN_MODEL---###########################################################################################


def train_model():
    dataset = "dataset"
    embeddingFile = "output/embeddings.pickle"
    embeddingModel = "openface_nn4.small2.v1.t7"
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    conf = 0.5

    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)
    imagePaths = list(paths.list_images(dataset))

    knownEmbeddings = []
    knownNames = []
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        print("Processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    print("Embedding:{0} ".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddingFile, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Process Completed")

###############################################################---RETRAIN---#################################################################################


def retrain_model(embeddingFile="output/embeddings.pickle",
                  recognizerFile="output/recognizer.pickle",
                  labelEncFile="output/le.pickle"):

    print("Loading face embeddings...")
    data = pickle.loads(open(embeddingFile, "rb").read())

    print("Encoding labels...")
    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["names"])

    print("Training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open(recognizerFile, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open(labelEncFile, "wb")
    f.write(pickle.dumps(labelEnc))
    f.close()

    print("Model retraining completed.")

####################################################---ATENDANCE_MARKING---#########################################################################33

def attendance():
    def flatten(lis):
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in flatten(item):
                    yield x
            else:
                yield item

    embeddingFile = "output/embeddings.pickle"
    embeddingModel = "openface_nn4.small2.v1.t7"
    recognizerFile = "output/recognizer.pickle"
    labelEncFile = "output/le.pickle"
    conf = 0.5

    print("[INFO] loading face detector...")
    prototxt = "model/deploy.prototxt"
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)

    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)

    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())

    USN = ""
    box = []
    print("[INFO] starting video stream...")
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)


    present_students = []  # list to keep track of marked attendance
    # attendance_file = 'attendance.csv'  # file to save attendance

    # get current date
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d-%I:%M:%S %p")

    # get subject name from user using a GUI
    window = Tk()
    window.withdraw()
    subject_code = simpledialog.askstring("Subject", "Enter subject code:")
    if subject_code is None:
        messagebox.showerror("Error", "Subject code cannot be empty")
        return
        

    # create new subject attendance CSV file if not already available
    attendance_file = f"{subject_code}_attendance.csv"
    try:
        with open(attendance_file, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "USN", "Timestamp", "Status"])
    except FileExistsError:
        pass

    # open camera and start recognizing faces
    while True:
        _, frame = cam.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba_threshold = 0.8
                proba = preds[j]

                if proba >= proba_threshold:
                    name = le.classes_[j]
                    USN = le.classes_[j]
                    
                    # Add the snippet here for attendance marking
                    with open('student.csv', 'r') as csvFile:
                        reader = csv.reader(csvFile)
                        for row in reader:
                            if name in row:
                                USN = row[1]
                                if name not in present_students:  # mark attendance if not already marked
                                    present_students.append(name)
                                    with open(attendance_file, 'a+', newline='') as f:
                                        reader = csv.reader(f)
                                        writer = csv.writer(f)
                                        
                                        # initialize flag
                                        name_found = False
                                        
                                        # check if name and USN already exist in attendance file
                                        for row in reader:
                                            if name == row[0] and USN == row[1]:
                                                name_found = True
                                                break
                                        
                                        # write attendance record if not already exist
                                        if not name_found:
                                            writer.writerow([name, USN, date_string, "Present"])
                
                else:
                    name = "Unknown"
                    USN = "Unknown"
                
                with open('student.csv', 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    for row in reader:
                        if name in row:
                            USN = row[1]
                            if name not in present_students:  # mark attendance if not already marked
                                present_students.append(name)
                                with open(attendance_file, 'a+', newline='') as f:
                                    reader = csv.reader(f)
                                    writer = csv.writer(f)
                                    
                                    # initialize flag
                                    name_found = False
                                    
                                    # check if name exists in the attendance.csv file
                                    for row in reader:
                                        if len(row) > 0 and row[0] == name:
                                            name_found = True
                                            if date_string not in row:
                                                row.append(date_string)
                                                row.append("Present")
                                                writer.writerow(row)
                                                print(f"Attendance marked for {name}")
                                            else:
                                                print(f"Attendance already marked for {name}")
                                            break
                                    
                                    # if name not found, add it to the attendance.csv file
                                    if not name_found:
                                        writer.writerow([name, USN, date_string, "Present"])
                                        print(f"Attendance marked for {name}")
                                        
                            break
                

                text = "{} : {} : {:.2f}%".format(name, USN, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()
    exit()
    
root.state('zoomed')    
root.mainloop()                                             