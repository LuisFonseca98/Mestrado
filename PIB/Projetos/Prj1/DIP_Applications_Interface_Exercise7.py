#from numpy import diag_indices
from asyncio.windows_events import NULL
from datetime import datetime
from DIP_Applications_Helper_Exercise7 import *
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import os
import cv2

### Constants for camera image capture
frameSavingPath = os.path.join(os.getcwd(), 'CapturedFrames')
frameNamePrefix = 'capturedFrame'

### Miscelaneous interface related utilities
def enableDIPButtons():
	global buttons
	for b in buttons[2:]:
		enableButton(b)
		
def disableAcquisitionButtons():
	disableButton(loadFromFileButton)
	disableButton(captureFromCameraButton)

def enableAcquisitionButtons():
	enableButton(loadFromFileButton)
	enableButton(captureFromCameraButton)
	
def enableButton(button):
	button["state"] = "normal"

def disableButton(button):
	button["state"] = "disabled"	

def message(text, redfont = False):
	global myLabel

	if(redfont):
		myLabel.config(fg='#f00')
		
	myLabel.config(text=text)

def generateFrameName(extension='jpg'):
	return frameNamePrefix + datetime.now().strftime("%Y%m%d_%H.%M.%S") + '.jpg'

def highlightGetFrameButton():
	global getFrameButton
	minBrightness = 128
	maxBrightness = 224
	step = 2

	color = int(getFrameButton.cget('bg')[1:], 16)
	brightness = (step + (color >> (8 + 8))) % maxBrightness # assuming grayscale background
	if brightness < minBrightness: brightness = minBrightness 
	color = brightness; 
	color = brightness + (color << 8)
	color = brightness + (color << 8)
	getFrameButton.config(bg='#' + f"{color:#08x}"[2:])

### Capturing from camera
def getFrame():
	global keepStreaming
	keepStreaming = False
	
def cleanupCapture():
	global captureFromCameraButton
	global getFrameButton
	global videoCapture	

	enableButton(captureFromCameraButton)
	getFrameButton.destroy()
	videoCapture.release()


def stream():
		global videoCapture
		global myImageLabel
		global keepStreaming
		global getFrameButton
		global rootFilePath

		chosenFrame = None
		ret, frame = videoCapture.read()
		if ret:		
			# Ajust channel order for PIL
			displayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Display the frame
			imgTk = ImageTk.PhotoImage(Image.fromarray(displayFrame))
			myImageLabel.image = imgTk
			myImageLabel.config(image = imgTk)
		
			if(keepStreaming):
				highlightGetFrameButton()
				myImageLabel.after(1, stream) 
				return	# main exit point, streaming continues
			else:
				frameName = generateFrameName()
				framePath = os.path.join(frameSavingPath, frameName)
				cv2.imwrite(framePath, frame)
				rootFilePath = framePath
				message(rootFilePath)
				enableDIPButtons()	# graceful stream termination
				
		else:
			message('Streamimg stopped... canceling.', True)	# stream error
			
		cleanupCapture()

def capturePictureFromCamera():
	global rootFilePath
	global videoCapture
	global getFrameButton
	global keepStreaming

	disableAcquisitionButtons()
	
	videoCapture = cv2.VideoCapture(0)
	if videoCapture.isOpened():
		message('Camera stream')
		getFrameButton = tk.Button(imageFrame,text='Get Frame',font = fontStyle,command=getFrame, bg='#0F0F0F')
		getFrameButton.place(x=10, y=37)
		keepStreaming = True
		stream()
	else:
		message('Could not open camera', True)
		videoCapture.release()
		enableAcquisitionButtons()

### Opening from file
def selectedPictureFromFile():
	global rootFilePath
	global myImageLabel
	
	f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png'), ('tiff Files', '*.tiff'), ('bmp Files', '*.bmp')]
	rootFilePath = filedialog.askopenfilename(title="Select a file",filetypes=f_types)
	if(rootFilePath == ''):
		return
	
	rootFileName=os.path.basename(rootFilePath)
	
	message(rootFileName)
	imgTk = ImageTk.PhotoImage(Image.open(rootFilePath))
	myImageLabel.image = imgTk
	myImageLabel.config(image = imgTk)
	
	enableDIPButtons()

### DIP functions

def identityHiding():
	global rootFilePath

	destination = blurImageFaces(rootFilePath)
	
	# display message
	messageTitle = 'Ready'
	messageText = f'Results saved into {destination}'
	tk.messagebox.showinfo(messageTitle, messageText)

def showBlurToImage():
	global rootFilePath

	destination = blurImage(rootFilePath)
	
	# display message
	messageTitle = 'Ready'
	messageText = f'Results saved into {destination}'
	tk.messagebox.showinfo(messageTitle, messageText)
	
def showNegativeImage():
	global rootFilePath

	destination = negativeImage(rootFilePath)
	
	# display message
	messageTitle = 'Ready'
	messageText = f'Results saved into {destination}'
	tk.messagebox.showinfo(messageTitle, messageText)

def showChangeContrast():
	global rootFilePath
	
	destination = improveContrast(rootFilePath)
	
	# display message
	messageTitle = 'Ready'
	messageText = f'Results saved into {destination}'
	tk.messagebox.showinfo(messageTitle, messageText)
	
def showFaceDetection():
	global rootFilePath
	
	destination = faceDetectionImages(rootFilePath)
	
	# display message
	messageTitle = 'Ready'
	messageText = f'Results saved into {destination}'
	tk.messagebox.showinfo(messageTitle, messageText)

def showShowSavePictureJPEGCompression():
	# jpeg quality levels for generating images
	# must contain at least one
	levels = [0, 33, 66, 100]
	
	# generate levels and save them
	destinationPath = savePictureJPEGCompressionLevels(rootFilePath, levels)
	
	# compose levels info string for displaying a message
	levelsStr = str(levels[0])
	for level in levels[1:]:
		levelsStr += f', {level}'
	
	# message contains destination path and quality level values
	messageTitle = 'Ready'
	messageText = f'Files saved lossily at:\n {destinationPath}\n\n for quality levels: {levelsStr}'
	
	# display message
	tk.messagebox.showinfo(messageTitle, messageText)

### Main
if __name__ == "__main__":
	global myLabel
	global myImageLabel
	global captureFromCameraButton
	
	# make sure destination directories for results exist
	setupFilesystem()
	# destination for captured frames
	if not os.path.exists(frameSavingPath):
		os.mkdir(frameSavingPath)
		

	root = tk.Tk()
	root.title('DIP Application')
	fontStyle = ('Arial',16)
	
	globalFrame = tk.Frame(root)
	globalFrame.columnconfigure(0,weight=1)
	globalFrame.columnconfigure(1,weight=1)
	globalFrame.pack()

	buttonFrame = tk.Frame(globalFrame)
	#buttonFrame.columnconfigure(0,weight=1)
	#buttonFrame.columnconfigure(1,weight=1)
	#buttonFrame.columnconfigure(2,weight=1)
	# buttonFrame.columnconfigure(3,weight=1)
	# buttonFrame.columnconfigure(4,weight=1)	
	buttonFrame.grid(row=0, column=0)

	# image acquisition button labels
	label = tk.Label(buttonFrame, text="Load or Capture an Image File", font=fontStyle, bg='white')
	label.pack(pady=20, fill = tk.X, expand=True)
	
	# image acquisition buttons
	buttons = [tk.Button(buttonFrame,text="Load Image from File",font = fontStyle,command=selectedPictureFromFile)]
	buttons[0].pack() #grid(row=0,column=0)
	loadFromFileButton = buttons[0]
	
	buttons.append(tk.Button(buttonFrame,text="Capture image from camera",font = fontStyle,command=capturePictureFromCamera))
	buttons[1].pack(pady=10) #.grid(row=1,column=0)
	captureFromCameraButton = buttons[1]
	
	# DIP buttons label
	buttonSetlabel = tk.Label(buttonFrame, text='DIP Operations', font=fontStyle, bg='white')
	buttonSetlabel.pack(pady=20, fill = tk.X, expand = True) #.grid(row=2,column=0, pady = 20, expand=True)	

	# DIP buttons
	buttons.append(tk.Button(buttonFrame,text="Identity Hiding",font = fontStyle,command=identityHiding))
	buttons[2].pack(pady=10) #.grid(row=3,column=0,padx=20,pady= 0)
		
	buttons.append(tk.Button(buttonFrame,text="Blur Image",font = fontStyle,command=showBlurToImage))
	buttons[3].pack(pady=10) #.grid(row=4,column=0,padx=20,pady= 10)
	
	buttons.append(tk.Button(buttonFrame,text="Contrast Adjustment",font = fontStyle,command=showChangeContrast))
	buttons[4].pack(pady=10) #.grid(row=5,column=0,padx=20, pady=10)
	
	buttons.append(tk.Button(buttonFrame,text="Negative Version",font = fontStyle,command=showNegativeImage))
	buttons[5].pack(pady=10) #.grid(row=6,column=0,padx=20, pady=10)
	
	buttons.append(tk.Button(buttonFrame,text="Eyes and Mouth Detection",font = fontStyle,command=showFaceDetection))
	buttons[6].pack(pady=10) #.grid(row=7,column=0,padx=20, pady=10)
	
	buttons.append(tk.Button(buttonFrame,text="JPEG Lossy Compression",font = fontStyle,command=showShowSavePictureJPEGCompression))
	buttons[7].pack(pady=10) #.grid(row=8,column=0,padx=20, pady=10)

	# buttons.append(tk.Button(buttonFrame,text="Save Picture as PNG",font = fontStyle,command=showShowSavePicturePNGCompression))
	# buttons[6].grid(row=2,column=2,padx=20, pady=20)

	for b in buttons[2:]:
		disableButton(b)

	imageFrame = tk.Frame(globalFrame)
	imageFrame.grid(row=0, column=1)
	
	myLabel = tk.Label(imageFrame, text='', font=('Arial',14))
	myLabel.pack()
	
	myImageLabel = tk.Label(imageFrame)
	myImageLabel.pack()
	
	root.mainloop()
	