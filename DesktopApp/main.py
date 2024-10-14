import json
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageApp:
    def __init__(self, root):
        config = json.load(open('config.json', 'r'))
        
        self.root = root
        self.root.title(config["app_title"])
        self.root.geometry(str(config["window_size"][0]) + "x" + str(config["window_size"][1]))
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(root, text="Choose file / Preview", bg="white", relief="sunken")
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        button_frame = tk.Frame(root)
        button_frame.grid(row=1, column=0, pady=(0, 10))
        
        load_button = tk.Button(button_frame, text="Load image", command=self.load_image)
        load_button.pack(side="left", padx=5)

        process_button = tk.Button(button_frame, text="Process photo", command=self.process_image)
        process_button.pack(side="right", padx=5)

        self.image_path = None
        self.image = None
        

    def load_image(self):
        pass
        
        
    def process_image(self):
        pass

    

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()