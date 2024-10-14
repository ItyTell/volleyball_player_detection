import json
from logging import config
import tkinter as tk

from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageApp:
    def __init__(self, root):
        self.config = json.load(open('config.json', 'r'))
        
        self.root = root
        self.root.title(self.config["app_title"])
        self.root.geometry(str(self.config["window_size"][0]) + "x" + str(self.config["window_size"][1]))
        
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

        root.drop_target_register(DND_FILES)
        root.dnd_bind('<<Drop>>', self.drop_image)
        root.bind("<Configure>", self.on_resize)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", ";".join(["*"+el for el in self.config["supported_formats"]]))])
        if self.image_path:
            self.display_image(self.image_path)

    def drop_image(self, event):
        self.image_path = event.data
        if self.image_path.lower().endswith(tuple(self.config["supported_formats"])):
            self.display_image(self.image_path)
        else:
            messagebox.showwarning("Invalid File", f"Please drop a valid image file {tuple([str.upper(el) for el in self.config['supported_formats']])}.")

    def display_image(self, path):
        self.image = Image.open(path)
        self.image.thumbnail((self.image_label.winfo_width(), self.image_label.winfo_height()))
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.image_tk, text="")
    
    def fit_image(self):
        if self.image:
            resized_image = self.image.copy()
            resized_image.thumbnail((self.image_label.winfo_width(), self.image_label.winfo_height()))
            self.image_tk = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=self.image_tk, text="")
    
    def process_image(self):
        if self.image:
            messagebox.showinfo("Process Image", "Image processed successfully!")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")
    
    def on_resize(self, event): 
        new_width = event.width 
        self.fit_image() 
    

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageApp(root)
    root.mainloop()