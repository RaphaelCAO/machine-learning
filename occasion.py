import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


def draw_rect(figure, pos, val):
    colors = ['red', 'magenta', 'orange', 'yellow', 'green', 'cyan', 'blue',
              'linen', 'lime', 'lightblue', 'pink', 'lightgray', 'gray', 'black']
    if val > len(colors):
        raise Exception('Value does not match any color')
    rect = plt.Rectangle(pos, 1, 1, color=colors[val])
    figure.add_patch(rect)


def draw(input1, input2, input3):
    rows1, cols1 = np.array(input1).shape
    rows2, cols2 = np.array(input2).shape

    # draw the first graph
    f = plt.figure().add_subplot(111)
    f.set_xticks(np.arange(rows1+1))
    f.set_yticks(np.arange(cols1+1))
    [draw_rect(f, (i, j), input1[i, j]) for i in range(rows1) for j in range(cols1)]
    f.set_axis_off()
    plt.savefig("first_figure.jpg")

    # draw the second graph
    f = plt.figure().add_subplot(111)
    f.set_xticks(np.arange(rows2 + 1))
    f.set_yticks(np.arange(cols2 + 1))
    [draw_rect(f, (i, j), input1[i, j]) for i in range(rows2) for j in range(cols2)]
    f.set_axis_off()
    plt.savefig("second_figure.jpg")


def test():
    print('btn available')


if __name__ == '__main__':
    # input1 = np.random.randint(0, 12, (10, 10))
    # input2 = np.random.randint(0, 12, (5, 5))
    # draw(input1, input2, [])

    root = tk.Tk()
    root.geometry('1400x800')
    root.title('Occasion')

    figure_width = 480
    figure_height = 480
    btn_width = 80
    btn_height = 80

    # put the first figure on the main window
    img = Image.open("first_figure.jpg")
    img = img.resize((figure_width, figure_height), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    first_fig = tk.Label(root, height=figure_height, width=figure_width)
    first_fig.image = img
    first_fig.configure(image=img)
    first_fig.place(x=20, y=60)

    # put the second figure on the main window
    img = Image.open("second_figure.jpg")
    img = img.resize((figure_width, figure_height), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    second_fig = tk.Label(root, height=figure_height, width=figure_width)
    second_fig.image = img
    second_fig.configure(image=img)
    second_fig.place(x=20+figure_width, y=60)

    # retrieve the full screen image
    full_screen_img = img = Image.open("full_screen.png")
    full_screen_img = img.resize((btn_width, btn_height), Image.ANTIALIAS)
    full_screen_img = ImageTk.PhotoImage(full_screen_img)

    # create the first full screen button
    full_screen_btn1 = tk.Button(root, image=full_screen_img, width=btn_width, height=btn_height,
                                 command=test, cursor="circle")
    full_screen_btn1.place(x=230, y=560)

    # create the second full screen button
    full_screen_btn2 = tk.Button(root, image=full_screen_img, width=btn_width, height=btn_height,
                                 command=test, cursor="circle")
    full_screen_btn2.place(x=700, y=560)

    root.mainloop()



