from tkinter import *
import numpy as np
from PIL import ImageGrab
import keras
import imageio

# Create GUI window
window = Tk()
window.title("Handwritten digit recognition")
l1 = Label()

# Loading model
model = keras.models.load_model('src/models/og_model.h5')

def MyProject():
    global l1

    widget = cv

    # Setting co-ordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Image is captured from canvas and is resized to (28 X 28) px
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    img.save('img/current.png')

    # Converting rgb to grayscale image
    im = imageio.imread("img/current.png")
    gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    gray = gray.reshape(1, 28, 28, 1)
    gray /= 255

    # Calling function for prediction
    prediction = model.predict(gray)

    # Displaying the result
    l1 = Label(window, text="Digit = " + str(prediction.argmax()), font=('Algerian', 20))
    l1.place(x=230, y=420)

lastx, lasty = None, None

# Clear the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()

# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

# Labels
L1 = Label(window, text="Handwritten Digit Recoginition", font=('Algerian', 25), fg="blue")
L1.place(x=35, y=10, anchor="center")

# Clear button
b1 = Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

# Predict button
b2 = Button(window, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=370)

# Setting properties of canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()