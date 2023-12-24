from datetime import datetime
from Prj2_Exercise4_Helper import *
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import os

### Constants for camera image capture
frameSavingPath = os.path.join(os.getcwd(), 'CapturedFrames')
frameNamePrefix = 'capturedFrame'


### Miscelaneous interface related utilities
def enableDIPButtons():
    global buttons
    for b in buttons[1:]:
        enableButton(b)


def enableButton(button):
    button["state"] = "normal"


def disableButton(button):
    button["state"] = "disabled"


def message(text, redfont=False):
    global myLabel

    if (redfont):
        myLabel.config(fg='#f00')

    myLabel.config(text=text)


def generateFrameName(extension='jpg'):
    return frameNamePrefix + datetime.now().strftime("%Y%m%d_%H.%M.%S") + '.jpg'



### Opening from file
def selectedPictureFromFile():
    global rootFilePath
    global myImageLabel

    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png'), ('tiff Files', '*.tiff'), ('bmp Files', '*.bmp')]
    rootFilePath = filedialog.askopenfilename(title="Select a file", filetypes=f_types)
    if (rootFilePath == ''):
        return

    rootFileName = os.path.basename(rootFilePath)

    message(rootFileName)
    imgTk = ImageTk.PhotoImage(Image.open(rootFilePath))
    myImageLabel.image = imgTk
    myImageLabel.config(image=imgTk)

    enableDIPButtons()


### DIP functions

def showFaceDetection():
    global rootFilePath

    destination = faceDetectionImages(rootFilePath)

    # display message
    messageTitle = 'Ready'
    messageText = f'Results saved into {destination}'
    tk.messagebox.showinfo(messageTitle, messageText)


def showFaceRecognition():
    return ''


### Main
if __name__ == "__main__":
    global myLabel
    global myImageLabel

    # make sure destination directories for results exist
    setupFilesystem()
    # destination for captured frames
    if not os.path.exists(frameSavingPath):
        os.mkdir(frameSavingPath)

    root = tk.Tk()
    root.title('DIP Application 2.0')
    fontStyle = ('Arial', 16)

    globalFrame = tk.Frame(root)
    globalFrame.columnconfigure(0, weight=1)
    globalFrame.columnconfigure(1, weight=1)
    globalFrame.pack()

    buttonFrame = tk.Frame(globalFrame)
    buttonFrame.grid(row=0, column=0)

    # image acquisition button labels
    label = tk.Label(buttonFrame, text="Load an Image File", font=fontStyle, bg='white')
    label.pack(pady=20, fill=tk.X, expand=True)

    # image acquisition buttons
    buttons = [tk.Button(buttonFrame, text="Load Image from File", font=fontStyle, command=selectedPictureFromFile)]
    buttons[0].pack()  # grid(row=0,column=0)
    loadFromFileButton = buttons[0]

    # DIP buttons label
    buttonSetlabel = tk.Label(buttonFrame, text='DIP Operations', font=fontStyle, bg='white')
    buttonSetlabel.pack(pady=20, fill=tk.X, expand=True)  # .grid(row=2,column=0, pady = 20, expand=True)

    buttons.append(tk.Button(buttonFrame, text="Face Detection", font=fontStyle, command=showFaceDetection))
    buttons[1].pack(pady=10)  # .grid(row=1,column=0)
    faceDetection = buttons[1]

    # DIP buttons
    buttons.append(tk.Button(buttonFrame, text="Face Recognition", font=fontStyle, command=showFaceRecognition))
    buttons[2].pack(pady=10)  # .grid(row=3,column=0,padx=20,pady= 0)
    faceRecognition = buttons[2]


    for b in buttons[1:]:
        disableButton(b)

    imageFrame = tk.Frame(globalFrame)
    imageFrame.grid(row=0, column=1)

    myLabel = tk.Label(imageFrame, text='', font=('Arial', 14))
    myLabel.pack()

    myImageLabel = tk.Label(imageFrame)
    myImageLabel.pack()

    root.mainloop()
