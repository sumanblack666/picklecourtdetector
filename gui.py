import tkinter as tk
from tkinter import filedialog
import os
from detect_pickleball_court import main as detect_court

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.select_input_folder_button = tk.Button(self)
        self.select_input_folder_button["text"] = "Select Input Folder"
        self.select_input_folder_button["command"] = self.select_input_folder
        self.select_input_folder_button.pack(side="top")

        self.input_folder_label = tk.Label(self, text="No folder selected")
        self.input_folder_label.pack(side="top")

        self.select_output_folder_button = tk.Button(self)
        self.select_output_folder_button["text"] = "Select Output Folder"
        self.select_output_folder_button["command"] = self.select_output_folder
        self.select_output_folder_button.pack(side="top")

        self.output_folder_label = tk.Label(self, text="No folder selected")
        self.output_folder_label.pack(side="top")

        self.run_button = tk.Button(self, text="Run", command=self.run_batch)
        self.run_button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory()
        self.input_folder_label.config(text=self.input_folder)

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        self.output_folder_label.config(text=self.output_folder)

    def run_batch(self):
        if not hasattr(self, 'input_folder') or not hasattr(self, 'output_folder'):
            print("Please select input and output folders")
            return

        for filename in os.listdir(self.input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(self.input_folder, filename)
                output_path = os.path.join(self.output_folder, os.path.splitext(filename)[0] + ".json")
                detect_court(image_path, output_path)

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
