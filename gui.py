import cv2
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import numpy as np
import os
from PIL import Image, ImageTk

allInputs = []
T = []
weights = np.array([])
image = None
text = "Type is : ???"

def open_image():
    global image
    path = tkFileDialog.askopenfilename(filetypes=[("Image Files", ".jpg .jpeg .png .gif")])
    if path:
        img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
        cv2.imwrite(f"images/test.jpg", resized)
        im = Image.open(f"images/test.jpg")
        tkimage = ImageTk.PhotoImage(im)
        image = tkimage
        label1.config(image=tkimage)
        label1.image = tkimage
        neural(path)

def isError(e):
    for i in range(len(e)):
        if(e[i][0] != 0):
            return True
    return False

def orthonormal(pp):
    for i in range(len(pp)):
        for j in range(len(pp[0])):
            if (i==j and pp[i][j] != 1) or (i != j and pp[i][j]!=0):
                return False
            
    return True

def training():
    global weights, T, allInputs
    S = 1
    folder_dir = "images/Healthy/"
    for image in os.listdir(folder_dir):
        allInputs.append(flatten(cv2.imread(f"{folder_dir}/{image}",cv2.IMREAD_GRAYSCALE)) )
        T.append([1 for _ in range(S)])
    folder_dir = "images/Diseased/"
    for image in os.listdir(folder_dir):
        allInputs.append(flatten(cv2.imread(f"{folder_dir}/{image}",cv2.IMREAD_GRAYSCALE)) )
        T.append([-1 for _ in range(S)])
    allInputs = np.array(allInputs)
    T = np.array(T).transpose()

    numP =  len(allInputs)
    R = len(allInputs[0])
    
    if orthonormal(np.dot(allInputs, allInputs.transpose())):
        weights = np.dot(T, allInputs)
    else:
        weights = np.dot(T, np.dot(np.linalg.inv(np.dot(allInputs, allInputs.transpose())),allInputs))

        
    

def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(-1 if el<128 else 1)
    return new_image


def neural(path):
    global weights, text
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)  # Resize to 256x256
    p = np.array(flatten(resized))
    p = p.transpose()
    #print(p.shape, weights.shape)
    a = np.dot(weights, p)
    text = "Type is : Healthy" if a[0] >= 0 else "Type is : Diseased"
    
    label2.config(text=text)
    label2.text = text

# Create a tkinter window
window = tk.Tk()
window.title("Image classification")
window.geometry("500x400")
# Load images (replace with your image filenames)
image = Image.open("none.jpg")
photo = ImageTk.PhotoImage(image)
# Create labels for images
label1 = tk.Label(window, image=photo)

# Create buttons
button1 = tk.Button(window, text="training", command=lambda: training())
button2 = tk.Button(window, text="select image", command=lambda: open_image())

# Create a label
label2 = tk.Label(window, text=text)

# Pack widgets
button1.pack()
button2.pack()
label1.pack()
label2.pack()
# Start the tkinter event loop
window.mainloop()
