# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
# from utils.utils import *
# from utils.custom_loss import IsolatingLossFunction, load_center_radius_api
# from models.seg_hrnet import get_seg_model
# from models.seg_hrnet_config import get_cfg_defaults
# from models.NLCDetection_api import NLCDetection
# from PIL import Image

# import torch
# import torch.nn as nn
# import numpy as np
# import argparse
# import imageio as imageio

# class HiFi_Net():
#     '''
#         FENET is the multi-branch feature extractor.
#         SegNet contains the classification and localization modules.
#         LOSS_MAP is the classification loss function class.
#     '''
#     def __init__(self):
#         device = torch.device('cuda:0')
#         device_ids = [0]

#         FENet_cfg = get_cfg_defaults()
#         FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model inside.
#         SegNet = NLCDetection().to(device)
#         FENet  = nn.DataParallel(FENet)
#         SegNet = nn.DataParallel(SegNet)

#         self.FENet  = restore_weight_helper(FENet,  "weights/HRNet",  750001)
#         self.SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 750001)
#         self.FENet.eval()
#         self.SegNet.eval()

#         center, radius = load_center_radius_api()
#         self.LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

#     def _transform_image(self, image_name):
#         '''transform the image.'''
#         image = imageio.imread(image_name)
#         image = Image.fromarray(image)
#         image = image.resize((256,256), resample=Image.BICUBIC)
#         image = np.asarray(image)
#         image = image.astype(np.float32) / 255.
#         image = torch.from_numpy(image)
#         image = image.permute(2, 0, 1)
#         image = torch.unsqueeze(image, 0)
#         return image

#     def _normalized_threshold(self, res, prob, threshold=0.5, verbose=False):
#         '''to interpret detection result via omitting the detection decision.'''
#         if res > threshold:
#             decision = "Forged"
#             prob = (prob - threshold) / threshold
#         else:
#             decision = 'Real'
#             prob = (threshold - prob) / threshold
#         print(f'Image being {decision} with the confidence {prob*100:.1f}.')

#     def detect(self, image_name, verbose=False):
#         """
#             Para: image_name is string type variable for the image name.
#             Return:
#                 res: binary result for real and forged.
#                 prob: the prob being the forged image.
#         """
#         with torch.no_grad():
#             img_input = self._transform_image(image_name)
#             output = self.FENet(img_input)
#             mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
#             res, prob = one_hot_label_new(out3)
#             res = level_1_convert(res)[0]
#             if not verbose:
#                 return res, prob[0]
#             else:
#                 self._normalized_threshold(res, prob[0])

#     def localize(self, image_name):
#         """
#             Para: image_name is string type variable for the image name.
#             Return:
#                 binary_mask: forgery mask.
#         """
#         with torch.no_grad():
#             img_input = self._transform_image(image_name)
#             output = self.FENet(img_input)
#             mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
#             pred_mask, pred_mask_score = self.LOSS_MAP.inference(mask1_fea)   # inference
#             pred_mask_score = pred_mask_score.cpu().numpy()
#             ## 2.3 is the threshold used to seperate the real and fake pixels.
#             ## 2.3 is the dist between center and pixel feature in the hyper-sphere.
#             ## for center and pixel feature please refer to "IsolatingLossFunction" in custom_loss.py
#             pred_mask_score[pred_mask_score<2.3] = 0.
#             pred_mask_score[pred_mask_score>=2.3] = 1.
#             binary_mask = pred_mask_score[0]        
#             return binary_mask


# def inference(img_path):
#     HiFi = HiFi_Net()   # initialize
    
#     ## detection
#     res3, prob3 = HiFi.detect(img_path)
#     # print(res3, prob3) 1 1.0
#     HiFi.detect(img_path, verbose=True)
    
#     ## localization
#     binary_mask = HiFi.localize(img_path)
#     binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
#     binary_mask.save('pred_mask.png')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_path', type=str, default='/home/sidx/myDrive/internship/imageForgeryDetection/HiFi_IFDL/data_dir/myDatasetImage/egg.jpg')
#     args = parser.parse_args()
#     inference(args.img_path)

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import sys
from utils.utils import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius_api
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_api import NLCDetection
import torch
import torch.nn as nn
import imageio


class HiFi_Net():
    '''
        FENET is the multi-branch feature extractor.
        SegNet contains the classification and localization modules.
        LOSS_MAP is the classification loss function class.
    '''
    def __init__(self):
        # Use CPU if CUDA is not available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            device_ids = [0]

        FENet_cfg = get_cfg_defaults()
        FENet = get_seg_model(FENet_cfg).to(self.device)
        SegNet = NLCDetection().to(self.device)
        
        if torch.cuda.is_available():
            FENet = nn.DataParallel(FENet)
            SegNet = nn.DataParallel(SegNet)

        self.FENet = restore_weight_helper(FENet, "weights/HRNet", 750001)
        self.SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 750001)
        self.FENet.eval()
        self.SegNet.eval()

        center, radius = load_center_radius_api()
        self.LOSS_MAP = IsolatingLossFunction(center, radius).to(self.device)

    def _transform_image(self, image_name):
        '''transform the image.'''
        if isinstance(image_name, str):
            image = imageio.imread(image_name)
        else:
            image = image_name  # Assume it's already a numpy array
            
        image = Image.fromarray(image)
        image = image.resize((256, 256), resample=Image.BICUBIC)
        image = np.asarray(image)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, 0)
        return image.to(self.device)

    def _normalized_threshold(self, res, prob, threshold=0.5):
        '''to interpret detection result via omitting the detection decision.'''
        if res > threshold:
            decision = "Forged"
            prob = (prob - threshold) / threshold
        else:
            decision = 'Real'
            prob = (threshold - prob) / threshold
        return decision, prob

    def detect(self, image_name, verbose=False):
        """
            Para: image_name is string type variable for the image name.
            Return:
                res: binary result for real and forged.
                prob: the prob being the forged image.
        """
        with torch.no_grad():
            img_input = self._transform_image(image_name)
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            res, prob = one_hot_label_new(out3)
            res = level_1_convert(res)[0]
            if not verbose:
                return res, prob[0]
            else:
                decision, confidence = self._normalized_threshold(res, prob[0])
                return decision, confidence * 100

    def localize(self, image_name):
        """
            Para: image_name is string type variable for the image name.
            Return:
                binary_mask: forgery mask.
        """
        with torch.no_grad():
            img_input = self._transform_image(image_name)
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            pred_mask, pred_mask_score = self.LOSS_MAP.inference(mask1_fea)  # inference
            pred_mask_score = pred_mask_score.cpu().numpy()
            ## 2.3 is the threshold used to separate the real and fake pixels.
            ## 2.3 is the dist between center and pixel feature in the hyper-sphere.
            ## for center and pixel feature please refer to "IsolatingLossFunction" in custom_loss.py
            pred_mask_score[pred_mask_score < 2.3] = 0.
            pred_mask_score[pred_mask_score >= 2.3] = 1.
            binary_mask = pred_mask_score[0]
            return binary_mask


class ImageForgeryDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HiFi Image Forgery Detection")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")
        
        # Create a model instance later in a separate thread to avoid freezing UI
        self.model = None
        self.model_loaded = False
        self.loading_model = False
        
        # Variables to store the current image path and data
        self.current_image_path = None
        self.current_image_data = None
        self.result_mask = None
        
        # Main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Hierarchical Fine-Grained Image Forgery Detection", 
            font=("Arial", 16, "bold"), 
            bg="#f0f0f0"
        )
        title_label.pack(pady=(0, 20))
        
        # Image displays frame (side by side)
        self.display_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        self.original_frame = tk.LabelFrame(self.display_frame, text="Original Image", bg="#f0f0f0", padx=10, pady=10)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.original_image_label = tk.Label(self.original_frame, bg="#e0e0e0", width=40, height=20)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Result mask frame
        self.result_frame = tk.LabelFrame(self.display_frame, text="Forgery Mask", bg="#f0f0f0", padx=10, pady=10)
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.result_image_label = tk.Label(self.result_frame, bg="#e0e0e0", width=40, height=20)
        self.result_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_columnconfigure(1, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)
        
        # Results frame
        self.results_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.results_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = tk.Label(
            self.results_frame, 
            text="No image analyzed yet", 
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.result_label.pack(pady=5)
        
        # Button frame
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, pady=10)
        
        # Upload button
        self.upload_btn = tk.Button(
            self.button_frame, 
            text="Upload Image", 
            command=self.upload_image,
            width=15,
            height=2
        )
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Analyze button
        self.analyze_btn = tk.Button(
            self.button_frame, 
            text="Analyze", 
            command=self.analyze_image,
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=10)
        
        # Save mask button
        self.save_btn = tk.Button(
            self.button_frame, 
            text="Save Mask", 
            command=self.save_mask,
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome to HiFi Image Forgery Detection")
        self.status_bar = tk.Label(
            root, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load model in a separate thread
        self.load_model_thread()

    def load_model_thread(self):
        """Load the model in a separate thread to avoid freezing the UI"""
        if self.loading_model:
            return
            
        self.loading_model = True
        self.status_var.set("Loading model... This might take a moment.")
        
        def load():
            try:
                self.model = HiFi_Net()
                self.model_loaded = True
                self.root.after(0, lambda: self.status_var.set("Model loaded successfully. Ready to analyze images."))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error loading model: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {str(e)}"))
            finally:
                self.loading_model = False
        
        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    def upload_image(self):
        """Handle image upload button click"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                # Load and display the image
                self.current_image_path = file_path
                img = Image.open(file_path)
                self.current_image_data = np.array(img)
                
                # Resize for display while maintaining aspect ratio
                img.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(img)
                
                self.original_image_label.config(image=img_tk)
                self.original_image_label.image = img_tk  # Keep a reference
                
                # Clear previous results
                self.result_image_label.config(image='')
                self.result_label.config(text="Upload successful. Click 'Analyze' to detect forgery.")
                self.result_mask = None
                
                # Enable analyze button if model is loaded
                if self.model_loaded:
                    self.analyze_btn.config(state=tk.NORMAL)
                else:
                    self.status_var.set("Model is still loading. Please wait...")
                    
                # Disable save button
                self.save_btn.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")

    def analyze_image(self):
        """Handle analyze button click"""
        if not self.current_image_path or not self.model_loaded:
            messagebox.showinfo("Info", "Please upload an image first and ensure the model is loaded.")
            return
            
        self.status_var.set("Analyzing image...")
        self.analyze_btn.config(state=tk.DISABLED)
        
        def process():
            try:
                # Run detection
                decision, confidence = self.model.detect(self.current_image_path, verbose=True)
                
                # Generate forgery mask
                binary_mask = self.model.localize(self.current_image_path)
                self.result_mask = (binary_mask * 255.).astype(np.uint8)
                
                # Create mask image for display
                mask_img = Image.fromarray(self.result_mask)
                mask_img.thumbnail((300, 300))
                mask_tk = ImageTk.PhotoImage(mask_img)
                
                # Update UI from main thread
                self.root.after(0, lambda: self.result_image_label.config(image=mask_tk))
                self.root.after(0, lambda: self.result_image_label.image_ref_keep(mask_tk))  # Keep a reference
                
                result_text = f"Result: {decision} with {confidence:.1f}% confidence"
                self.root.after(0, lambda: self.result_label.config(text=result_text))
                
                if self.result_mask is not None:
                    self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
                
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
                self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
                self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
        
        # Create a reference to keep the PhotoImage from being garbage collected
        self.result_image_label.image_ref_keep = lambda img: setattr(self.result_image_label, 'image', img)
        
        # Run analysis in a separate thread
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    def save_mask(self):
        """Save the forgery mask to a file"""
        if self.result_mask is None:
            messagebox.showinfo("Info", "No mask to save. Analyze an image first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Forgery Mask",
            defaultextension=".png",
            filetypes=(
                ("PNG files", "*.png"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                mask_img = Image.fromarray(self.result_mask)
                mask_img.save(file_path)
                self.status_var.set(f"Mask saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageForgeryDetectionGUI(root)
    root.mainloop()
