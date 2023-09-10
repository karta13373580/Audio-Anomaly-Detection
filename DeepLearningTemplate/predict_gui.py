# import
from tkinter import Tk, Button, Label, filedialog
from os.path import dirname


# class
class BasePredictGUI:
    def __init__(self, extensions) -> None:
        self.extensions = extensions
        self.filepath = None

        # window
        self.window = Tk()
        self.window.geometry('{}x{}'.format(self.window.winfo_screenwidth(),
                                            self.window.winfo_screenheight()))
        self.window.title('Prediction GUI')

        # button
        self.open_file_button = Button(master=self.window,
                                       text='Open File',
                                       command=self.open_file)
        self.recognize_button = Button(master=self.window,
                                       text='Recognize',
                                       command=self.recognize)

        # label
        self.filepath_label = Label(master=self.window)
        self.predicted_label = Label(master=self.window)
        self.result_label = Label(master=self.window, font=(None, 50))

    def reset_widget(self):
        self.filepath_label.config(text='')
        self.predicted_label.config(text='')
        self.result_label.config(text='')

    def open_file(self):
        self.reset_widget()
        initialdir = dirname(
            self.filepath) if self.filepath is not None else './'
        self.filepath = filedialog.askopenfilename(initialdir=initialdir,
                                                   filetypes=[('extensions',
                                                               self.extensions)
                                                              ])
        self.filepath_label.config(text='filepath: {}'.format(self.filepath))

    def recognize(self):
        pass

    def run(self):
        self.window.mainloop()