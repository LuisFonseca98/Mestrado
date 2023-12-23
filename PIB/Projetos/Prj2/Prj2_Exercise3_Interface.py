from tkinter import messagebox
from Prj2_Exercise3_Helper import *
from PIL import Image, ImageTk
import tkinter as tk

"""
Display a success or failed message according to the user input
"""
def submit_message(input_text, text_received):
    # display message

    if input_text == text_received:
        message_title = 'Success'
        message_text = f'Success!'
        tk.messagebox.showinfo(message_title, message_text)
    else:
        message_title = 'Failed'
        message_text = f'Failed!'
        tk.messagebox.showinfo(message_title, message_text)


if __name__ == "__main__":
    length_of_sequence = random.randint(1, 9)
    randomWidth = random.randint(100, 300)
    randomHeight = random.randint(100, 300)

    random_numbers = generate_random_string(length_of_sequence)
    image_captcha = generate_captcha(length_of_sequence, randomWidth, randomHeight)
    image_name = split_text(image_captcha)

    window = tk.Tk()

    print(image_name)

    window.title('Captcha Application')
    fontStyle = ('Arial', 16)
    window.geometry('500x500')

    show_image_captcha = ImageTk.PhotoImage(Image.open(image_captcha))
    labelImage = tk.Label(window, image=show_image_captcha)

    labelImage.image = show_image_captcha
    labelImage.grid(column=0, row=0)

    labelInput = tk.Label(text='Enter the letters on the captcha above', font=20)

    labelInput.grid(row=1, column=0)

    inputBox = tk.Entry(window, font=("Times New Roman", 12))
    inputBox.grid(row=1, column=1)

    def on_click_show_message():
        entry_text = inputBox.get()
        submit_message(entry_text, image_name)

    buttonSubmit = tk.Button(window, text='Submit', font=40, command=on_click_show_message)
    buttonSubmit.grid(row=2, column=0)

    window.mainloop()
