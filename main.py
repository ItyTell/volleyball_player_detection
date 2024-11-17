import cv2
import json
from logging import config
import numpy as np
import tkinter as tk
from cv_algorithm.model import load_model
from cv_algorithm.detect import detect_image

from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

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
        
        self.init_main_btn_frame()
        self.init_nav_btn_frame()
        self.show_main_buttons()

        self.image_path = None # str
        self.image = None # PIL Image
        self.processed_image = None # PIL Image
        
        self.sub_images = np.array([])
        self.current_sub_img = 0 # index of current subimage

        self.model = load_model(
            self.config["model"]["config"],
            self.config["model"]["checkpoint"],
        )

        root.drop_target_register(DND_FILES)
        root.dnd_bind('<<Drop>>', self.drop_image)
        root.bind("<Configure>", self.on_resize)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", " ".join(["."+el for el in self.config["supported_formats"]]))])
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.processed_image = self.image.copy()
            
            self.processed_image.thumbnail((self.image_label.winfo_width(), self.image_label.winfo_height()))
            self.display_image(self.processed_image)

    def drop_image(self, event):
        self.image_path = event.data
        
        if not self.image_path.lower().endswith(tuple(self.config["supported_formats"])):
            messagebox.showwarning("Invalid File", f"Please drop a valid image file {tuple([str.upper(el) for el in self.config['supported_formats']])}.")
            return
        
        self.image = Image.open(self.image_path)
        self.processed_image = self.image.copy()
            
        self.processed_image.thumbnail((self.image_label.winfo_width(), self.image_label.winfo_height()))
        self.display_image(self.processed_image)

    def display_image(self, image):
        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk, text="")
        
        self.fit_image()
        
    def display_current_sub_img(self):
        if not self.sub_images.size:
            return
        
        if self.current_sub_img < 0 or self.current_sub_img >= self.sub_images.size:
            return
        
        
        pad = self.config["processed_image_padding"]
        
        if np.array(self.sub_images[self.current_sub_img]).size < 4:
            return
        
        x, y, width, height = self.sub_images[self.current_sub_img]
        
        sub_image = self.image.crop([
                max(x - pad, 0),
                max(y - pad, 0),
                max(x + width + pad, 0),
                max(y + height + pad, 0)
            ])
        
        sub_image = cv2.rectangle(np.array(sub_image), 
                                  (x, y), (x + width, y + height), 
                                  color=self.config["outline_color"], thickness=self.config["outline_width"])

        self.processed_image = Image.fromarray(sub_image)
        
        self.display_image(self.processed_image)
    
    def fit_image(self):
        if self.image:
            resized_image = self.processed_image.copy()
            resized_image.thumbnail((self.image_label.winfo_width(), self.image_label.winfo_height()))
            self.image_tk = ImageTk.PhotoImage(resized_image)
            self.image_label.config(image=self.image_tk, text="")
    
    def process_image(self):
        if not self.image:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.show_nav_buttons()
        
        if not self.sub_images.size:
            messagebox.showerror("Processing Image", "Subimages array is empty.")
            return
        
        self.display_current_sub_img()
    
    def on_resize(self, event): 
        new_width = event.width 
        self.fit_image() 
    
    def init_main_btn_frame(self):
        self.main_button_frame = tk.Frame(root)

        self.load_button = tk.Button(self.main_button_frame, text="Load image", command=self.load_image)
        self.load_button.pack(side="left", padx=5)

        self.process_button = tk.Button(self.main_button_frame, text="Process photo", command=self.process_image)
        self.process_button.pack(side="right", padx=5)
    
    def show_main_buttons(self):
        self.nav_button_frame.grid_remove()
        self.main_button_frame.grid(row=1, column=0, pady=(0, 10))
    
    def go_to_main_page(self):
        self.show_main_buttons()
        
        if self.image:
            self.processed_image = self.image.copy()
            self.display_image(self.processed_image)
    
    def init_nav_btn_frame(self):
        self.nav_button_frame = tk.Frame(root)
        
        self.main_page_button = tk.Button(self.nav_button_frame, text="Main Page", command=self.go_to_main_page)
        self.main_page_button.pack(side="left", padx=5)

        self.left_button = tk.Button(self.nav_button_frame, text="←", command=self.previous_image)
        self.left_button.pack(side="left", padx=5)

        self.right_button = tk.Button(self.nav_button_frame, text="→", command=self.next_image)
        self.right_button.pack(side="left", padx=5)
    
    def show_nav_buttons(self):
        self.main_button_frame.grid_remove()
        self.nav_button_frame.grid(row=1, column=0, pady=(0, 10))

    def previous_image(self):
        if not self.sub_images.size:
            messagebox.showerror("Previous Image", "Subimages array is empty.")
            return
        
        self.current_sub_img -= 1
        self.current_sub_img %= len(self.sub_images)
        self.display_current_sub_img()
        

    def next_image(self):
        if not self.sub_images.size:
            messagebox.showerror("Next Image", "Subimages array is empty.")
            return
        
        self.current_sub_img += 1
        self.current_sub_img %= len(self.sub_images)
        self.display_current_sub_img()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageApp(root)
    root.mainloop()
