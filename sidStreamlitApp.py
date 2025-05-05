import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import imageio
import time

# Import your existing modules
# Note: You'll need to make sure these modules are accessible when deploying
from utils.utils import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius_api
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_api import NLCDetection

class HiFi_Net():
    '''
        FENET is the multi-branch feature extractor.
        SegNet contains the classification and localization modules.
        LOSS_MAP is the classification loss function class.
    '''
    def __init__(self):
        # Use CPU if CUDA is not available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        st.sidebar.info(f"Using device: {self.device}")

        FENet_cfg = get_cfg_defaults()
        FENet = get_seg_model(FENet_cfg).to(self.device)
        SegNet = NLCDetection().to(self.device)
        
        if torch.cuda.is_available():
            device_ids = [0]
            FENet = nn.DataParallel(FENet)
            SegNet = nn.DataParallel(SegNet)

        self.FENet = restore_weight_helper(FENet, "weights/HRNet", 750001)
        self.SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 750001)
        self.FENet.eval()
        self.SegNet.eval()

        center, radius = load_center_radius_api()
        self.LOSS_MAP = IsolatingLossFunction(center, radius).to(self.device)

    def _transform_image(self, image_data):
        '''transform the image.'''
        if isinstance(image_data, str):
            image = imageio.imread(image_data)
        else:
            image = image_data  # Assume it's already a numpy array
            
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
        return decision, prob * 100  # Return confidence as percentage

    def detect(self, image_data):
        """
            Para: image_data can be a path string or numpy array
            Return:
                decision: "Real" or "Forged"
                confidence: confidence percentage
        """
        with torch.no_grad():
            img_input = self._transform_image(image_data)
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            res, prob = one_hot_label_new(out3)
            res = level_1_convert(res)[0]
            decision, confidence = self._normalized_threshold(res, prob[0])
            return decision, confidence

    def localize(self, image_data):
        """
            Para: image_data can be a path string or numpy array
            Return:
                binary_mask: forgery mask.
        """
        with torch.no_grad():
            img_input = self._transform_image(image_data)
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            pred_mask, pred_mask_score = self.LOSS_MAP.inference(mask1_fea)  # inference
            pred_mask_score = pred_mask_score.cpu().numpy()
            ## 2.3 is the threshold used to separate the real and fake pixels.
            pred_mask_score[pred_mask_score < 2.3] = 0.
            pred_mask_score[pred_mask_score >= 2.3] = 1.
            binary_mask = pred_mask_score[0]
            return binary_mask


# Configure the Streamlit page
def set_page_config():
    st.set_page_config(
        page_title="HiFi Image Forgery Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS to improve the look and feel
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stImage > img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .result-text {
            font-size: 1.5rem;
            font-weight: bold;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
        }
        .forged {
            background-color: rgba(255, 0, 0, 0.1);
            color: darkred;
        }
        .real {
            background-color: rgba(0, 128, 0, 0.1);
            color: darkgreen;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'result_mask' not in st.session_state:
        st.session_state.result_mask = None
    if 'detection_result' not in st.session_state:
        st.session_state.detection_result = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None

# Load the model
def load_model():
    if not st.session_state.model_loaded:
        with st.spinner('Loading the HiFi model. This might take a moment...'):
            try:
                st.session_state.model = HiFi_Net()
                st.session_state.model_loaded = True
                st.sidebar.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()

# Display the sidebar
def display_sidebar():
    st.sidebar.title("HiFi Image Forgery Detection")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app uses the Hierarchical Fine-Grained Image Forgery Detection model "
        "to detect and localize image forgeries."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown(
        "1. Upload an image\n"
        "2. Wait for the analysis to complete\n"
        "3. View the results and forgery mask"
    )
    
    # Add model information
    if st.session_state.model_loaded:
        st.sidebar.markdown("---")
        st.sidebar.success("✅ Model is loaded and ready")
    else:
        st.sidebar.warning("⚠️ Model is still loading")

# Analyze the uploaded image
def analyze_image(image_data):
    if not st.session_state.model_loaded:
        st.warning("Model is still loading. Please wait...")
        return
        
    with st.spinner("Analyzing image for forgery..."):
        try:
            # Run detection
            decision, confidence = st.session_state.model.detect(image_data)
            st.session_state.detection_result = decision
            st.session_state.confidence = confidence
            
            # Generate forgery mask
            binary_mask = st.session_state.model.localize(image_data)
            st.session_state.result_mask = (binary_mask * 255.).astype(np.uint8)
            
            return decision, confidence, st.session_state.result_mask
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None, None, None

# Main function to run the Streamlit app
def main():
    set_page_config()
    init_session_state()
    
    st.title("🔍 Hierarchical Fine-Grained Image Forgery Detection")
    st.markdown("Upload an image to detect if it has been manipulated and see the forgery mask.")
    
    # Display the sidebar
    display_sidebar()
    
    # Load the model (if not already loaded)
    load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_data = np.array(image)
        
        # Store the uploaded image
        st.session_state.uploaded_image = image_data
        
        # Create two columns for displaying images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Add analyze button
        if st.button("Analyze Image"):
            decision, confidence, mask = analyze_image(image_data)
            
            if decision and mask is not None:
                with col2:
                    st.subheader("Forgery Mask")
                    st.image(mask, use_column_width=True)
                
                # Display the result with styling
                result_class = "forged" if decision == "Forged" else "real"
                st.markdown(f"""
                    <div class='result-text {result_class}'>
                        Result: {decision} with {confidence:.1f}% confidence
                    </div>
                """, unsafe_allow_html=True)
                
                # Provide download option for the mask
                if mask is not None:
                    mask_image = Image.fromarray(mask)
                    # Convert to bytes
                    from io import BytesIO
                    buf = BytesIO()
                    mask_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Forgery Mask",
                        data=byte_im,
                        file_name=f"forgery_mask_{uploaded_file.name.split('.')[0]}.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()