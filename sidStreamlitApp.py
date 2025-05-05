import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import os
import tempfile
import time

# Set page configuration
st.set_page_config(
    page_title="HiFi Image Forgery Detection",
    page_icon="🔍",
    layout="wide"
)

# Create a placeholder for the model
class MockHiFiNet:
    """
    This is a mock implementation of your HiFi_Net class
    In a real deployment, you would include your actual model code here
    """
    def __init__(self):
        # Simulate model loading time
        time.sleep(2)
        self.loaded = True
        
    def detect(self, image_path, verbose=False):
        """Mock detection - returns random result for demo purposes"""
        time.sleep(1)  # Simulate processing time
        is_forged = np.random.choice([True, False], p=[0.7, 0.3])
        confidence = np.random.uniform(60, 95) if is_forged else np.random.uniform(60, 95)
        
        if not verbose:
            return int(is_forged), confidence / 100.0
        else:
            decision = "Forged" if is_forged else "Real"
            return decision, confidence
    
    def localize(self, image_path):
        """Mock localization - returns a random mask for demo purposes"""
        time.sleep(1)  # Simulate processing time
        
        # Read the image to get its dimensions
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = Image.fromarray(image_path)
            
        # Create a random binary mask (more complex in real implementation)
        w, h = 256, 256  # Fixed size used in the original code
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add some random "forgery" regions
        num_regions = np.random.randint(1, 4)
        for _ in range(num_regions):
            x = np.random.randint(0, w - 50)
            y = np.random.randint(0, h - 50)
            size_x = np.random.randint(30, 70)
            size_y = np.random.randint(30, 70)
            mask[y:y+size_y, x:x+size_x] = 1
            
        return mask

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['model_loaded'] = False
    st.session_state['result_mask'] = None
    st.session_state['analyzed'] = False

def load_model():
    """Load the forgery detection model"""
    with st.spinner("Loading model... This might take a moment."):
        try:
            # In a real application, this would be your actual HiFi_Net model
            st.session_state['model'] = MockHiFiNet()
            st.session_state['model_loaded'] = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

def analyze_image(image_file):
    """Analyze the uploaded image for forgery"""
    if not st.session_state['model_loaded']:
        st.warning("Model is not loaded yet. Please wait.")
        return
    
    with st.spinner("Analyzing image..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(image_file.getvalue())
                temp_path = tmp.name
            
            # Run detection
            decision, confidence = st.session_state['model'].detect(temp_path, verbose=True)
            
            # Generate forgery mask
            binary_mask = st.session_state['model'].localize(temp_path)
            st.session_state['result_mask'] = (binary_mask * 255).astype(np.uint8)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            st.session_state['analyzed'] = True
            return decision, confidence
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None, None

# App header
st.title("Hierarchical Fine-Grained Image Forgery Detection")
st.markdown("""
This application detects and localizes image forgeries using a deep learning approach.
Upload an image, and the system will analyze it for potential manipulations.
""")

# Load model on app start
if not st.session_state['model_loaded']:
    load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])

# Display image and analysis in columns
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Display original image
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Analyze button
        if st.button("Analyze Image"):
            decision, confidence = analyze_image(uploaded_file)
            if decision and confidence:
                st.session_state['decision'] = decision
                st.session_state['confidence'] = confidence
    
    # Display results
    with col2:
        if st.session_state.get('analyzed', False):
            st.subheader("Forgery Mask")
            mask_img = Image.fromarray(st.session_state['result_mask'])
            st.image(mask_img, use_column_width=True)
            
            # Display result text
            result_color = "red" if st.session_state['decision'] == "Forged" else "green"
            st.markdown(f"""
            <div style='background-color: rgba(0,0,0,0.05); padding: 15px; border-radius: 5px;'>
                <h3 style='color: {result_color};'>Result: {st.session_state['decision']}</h3>
                <h4>Confidence: {st.session_state['confidence']:.1f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Save button
            if st.download_button(
                label="Download Forgery Mask",
                data=uploaded_file.getvalue() if uploaded_file else None,  # Replace with actual mask data in bytes
                file_name="forgery_mask.png",
                mime="image/png",
                disabled=not st.session_state.get('analyzed', False)
            ):
                st.success("Mask downloaded successfully!")

# Add information about deployment
st.sidebar.title("Deployment Info")
st.sidebar.markdown("""
### How to Share This App

To share this app with your project mentor:

1. **Deploy on Streamlit Cloud**:
   - Create a GitHub repository with this code
   - Connect it to [Streamlit Cloud](https://streamlit.io/cloud)
   - Share the provided URL

2. **Run Locally and Share via Ngrok**:
   ```bash
   pip install streamlit pyngrok
   streamlit run app.py --server.port 8501
   ngrok http 8501
   ```
   Then share the ngrok URL

3. **Deploy on Heroku**:
   - Create a Procfile and requirements.txt
   - Push to Heroku
   - Share the Heroku app URL
""")

# If this is a demo version, add a note
st.sidebar.markdown("""
### Note
This is a demonstration version with mock model functionality.
For actual deployment, replace the `MockHiFiNet` class with your 
real implementation of the HiFi_Net model.
""")

# Instructions for the mentor
st.sidebar.markdown("""
### For the Mentor
1. Upload an image using the file uploader
2. Click "Analyze Image" to process
3. View the forgery mask and detection results
4. Download the mask if needed
""")

# Footer
st.markdown("""
---
*This application uses deep learning techniques to detect image forgeries.*
""")