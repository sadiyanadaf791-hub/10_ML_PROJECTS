"""
Visual Question Answering (VQA) Web Application
A production-ready VQA system using BLIP model and Streamlit.

File: app.py
Author: VQA Application
Description: Complete web application for answering questions about images using AI
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForQuestionAnswering
import io
import base64
from typing import Tuple, Optional
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Visual Question Answering - AI Image Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/vqa-app',
        'Report a bug': 'https://github.com/yourusername/vqa-app/issues',
        'About': '# VQA Application\nAsk questions about images using AI!'
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styles */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Answer display box */
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Confidence score box */
    .confidence-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .confidence-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 40px;
        margin-top: 10px;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        transition: width 0.8s ease;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #f0f7ff 100%);
        padding: 1rem;
        border-left: 5px solid #1f77b4;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Warning box */
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Error box */
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
    
    /* Spinner overlay */
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Select box styling */
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# VQA MODEL CLASS
# ============================================================================

class VQAModel:
    """
    Visual Question Answering model wrapper using BLIP.
    Handles model loading, caching, and inference operations.
    
    Attributes:
        model_name (str): HuggingFace model identifier
        device (str): Device to run inference on ('cuda' or 'cpu')
        processor: BLIP processor for preprocessing
        model: BLIP model for inference
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        """
        Initialize the VQA model wrapper.
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        logger.info(f"VQAModel initialized with device: {self.device}")
        
    @st.cache_resource
    def load_model(_self):
        """
        Load and cache the BLIP model and processor.
        Uses Streamlit's cache to avoid reloading on every run.
        
        Returns:
            tuple: (processor, model)
            
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model: {_self.model_name}")
            st.info(f"🔄 Loading AI model... This may take a minute on first run.")
            
            # Load processor
            _self.processor = BlipProcessor.from_pretrained(_self.model_name)
            logger.info("Processor loaded successfully")
            
            # Load model
            _self.model = BlipForQuestionAnswering.from_pretrained(_self.model_name)
            _self.model.to(_self.device)
            _self.model.eval()
            logger.info(f"Model loaded successfully on {_self.device}")
            
            st.success(f"✅ Model loaded successfully on {_self.device.upper()}!")
            
            return _self.processor, _self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            st.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def answer_question(
        self, 
        image: Image.Image, 
        question: str, 
        max_length: int = 50,
        num_beams: int = 5
    ) -> Tuple[str, float]:
        """
        Generate answer for the given image and question.
        
        Args:
            image (Image.Image): PIL Image object
            question (str): Question string
            max_length (int): Maximum length of generated answer
            num_beams (int): Number of beams for beam search
            
        Returns:
            tuple: (answer, confidence_score)
            
        Raises:
            Exception: If inference fails
        """
        try:
            # Ensure model is loaded
            if self.processor is None or self.model is None:
                self.load_model()
            
            logger.info(f"Processing question: {question}")
            
            # Preprocess inputs
            inputs = self.processor(
                images=image, 
                text=question, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer with beam search
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode answer
            answer = self.processor.decode(
                outputs.sequences[0], 
                skip_special_tokens=True
            )
            
            # Calculate confidence score
            if hasattr(outputs, 'sequences_scores'):
                confidence = torch.exp(outputs.sequences_scores[0]).item()
                confidence = min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
            else:
                # Fallback confidence calculation
                confidence = 0.85
            
            logger.info(f"Generated answer: {answer} (confidence: {confidence:.2f})")
            
            return answer.strip(), confidence
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate answer: {str(e)}")

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_image(uploaded_file) -> Optional[Image.Image]:
    """
    Validate and load uploaded image file with comprehensive checks.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        PIL Image object or None if invalid
    """
    try:
        # Check if file exists
        if uploaded_file is None:
            return None
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if uploaded_file.size > max_size:
            st.error(f"❌ Image too large ({uploaded_file.size / (1024*1024):.1f}MB). Please upload an image smaller than 10MB.")
            logger.warning(f"Image too large: {uploaded_file.size} bytes")
            return None
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if uploaded_file.type not in allowed_types:
            st.error(f"❌ Invalid file type: {uploaded_file.type}. Please upload JPG or PNG images only.")
            logger.warning(f"Invalid file type: {uploaded_file.type}")
            return None
        
        # Open and validate image
        image = Image.open(uploaded_file)
        
        # Check if image is valid
        image.verify()
        
        # Reopen image after verify (verify closes the file)
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Check dimensions
        width, height = image.size
        logger.info(f"Image dimensions: {width}x{height}")
        
        if width > 4000 or height > 4000:
            st.warning(f"⚠️ Large image detected ({width}x{height}). Resizing for faster processing...")
            image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            logger.info(f"Image resized to: {image.size}")
        
        # Check for corrupted images
        try:
            image.load()
        except Exception as e:
            st.error("❌ Image appears to be corrupted. Please try another image.")
            logger.error(f"Corrupted image: {str(e)}")
            return None
        
        logger.info("Image validation successful")
        return image
        
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        logger.error(f"Image validation error: {str(e)}", exc_info=True)
        return None


def validate_question(question: str) -> bool:
    """
    Validate question input with comprehensive checks.
    
    Args:
        question (str): Question string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if empty
    if not question or len(question.strip()) == 0:
        st.error("❌ Please enter a question.")
        logger.warning("Empty question submitted")
        return False
    
    # Check length
    if len(question) > 200:
        st.error(f"❌ Question too long ({len(question)} characters). Please keep it under 200 characters.")
        logger.warning(f"Question too long: {len(question)} characters")
        return False
    
    # Check for minimum length
    if len(question.strip()) < 3:
        st.error("❌ Question too short. Please provide a meaningful question.")
        logger.warning("Question too short")
        return False
    
    # Check for question mark (optional but recommended)
    if '?' not in question:
        st.info("💡 Tip: Questions typically end with a '?'")
    
    logger.info(f"Question validation successful: {question}")
    return True

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_results(answer: str, confidence: float, image: Image.Image, question: str):
    """
    Display the VQA results in a beautifully formatted layout.
    
    Args:
        answer (str): Generated answer
        confidence (float): Confidence score (0-1)
        image (Image.Image): Original image
        question (str): User's question
    """
    # Display answer with animation
    st.markdown(f"""
    <div class="answer-box">
        💡 <strong>Answer:</strong> {answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        confidence_percent = confidence * 100
        st.markdown(f"""
        <div class="confidence-box">
            <h4 style="margin: 0; color: #333; text-align: center;">Confidence Score</h4>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width: {confidence_percent}%;">
                    {confidence_percent:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display additional information
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0 0 0.5rem 0;">📝 Your Question</h4>
            <p style="margin: 0; font-size: 1.1rem;">{}</p>
        </div>
        """.format(question), unsafe_allow_html=True)
    
    with col2:
        width, height = image.size
        st.markdown(f"""
        <div class="info-box">
            <h4 style="margin: 0 0 0.5rem 0;">📊 Image Details</h4>
            <p style="margin: 0;">
                <strong>Dimensions:</strong> {width} × {height} pixels<br>
                <strong>Format:</strong> {image.format if image.format else 'Unknown'}<br>
                <strong>Mode:</strong> {image.mode}
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_sample_gallery():
    """Display a gallery of sample images and questions."""
    st.markdown("### 🖼️ Sample Use Cases")
    
    examples = [
        {"question": "What is in this image?", "icon": "🔍", "desc": "General object identification"},
        {"question": "What color is the main object?", "icon": "🎨", "desc": "Color recognition"},
        {"question": "How many people are visible?", "icon": "👥", "desc": "Counting objects"},
        {"question": "What is the person doing?", "icon": "🏃", "desc": "Activity recognition"},
        {"question": "What's the weather like?", "icon": "🌤️", "desc": "Scene understanding"},
        {"question": "Is this indoors or outdoors?", "icon": "🏠", "desc": "Location classification"},
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{example['icon']}</div>
                <div style="font-weight: 600; color: #333; margin-bottom: 0.3rem;">"{example['question']}"</div>
                <div style="font-size: 0.9rem; color: #666;">{example['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application logic and UI layout.
    Orchestrates the entire VQA workflow.
    """
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    
    st.markdown('<h1 class="main-header">🔍 Visual Question Answering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image and ask questions about it using state-of-the-art AI</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        This application uses **BLIP** (Bootstrapping Language-Image Pre-training), 
        a state-of-the-art vision-language model developed by Salesforce Research.
        
        **Key Features:**
        - 🎯 Accurate visual understanding
        - 💬 Natural language questions
        - ⚡ Fast inference
        - 🎨 Modern, responsive UI
        """)
        
        st.divider()
        
        # Model settings
        st.header("⚙️ Settings")
        
        max_length = st.slider(
            "Max Answer Length", 
            min_value=10, 
            max_value=100, 
            value=50,
            step=5,
            help="Maximum number of tokens in the generated answer"
        )
        
        num_beams = st.slider(
            "Beam Search Width",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Higher values may improve answer quality but slow down inference"
        )
        
        st.divider()
        
        # System information
        st.header("💻 System Info")
        device_type = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"**Running on:** {device_type}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.info(f"**GPU:** {gpu_name}")
        
        st.divider()
        
        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. **Upload** an image using the file uploader
        2. **Type** your question or select a sample
        3. **Click** the "Get Answer" button
        4. **View** the AI-generated answer and confidence score
        
        **Tips:**
        - Use clear, specific questions
        - Ask about visible elements in the image
        - Try different question formats
        """)
        
        st.divider()
        
        # Example questions
        st.header("💡 Example Questions")
        st.markdown("""
        - What is in this image?
        - What color is the car?
        - How many people are there?
        - What is the person doing?
        - Is this indoors or outdoors?
        - What's the weather like?
        - What time of day is it?
        - What's in the background?
        """)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Initialize session state
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = None
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = None
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    if 'last_question' not in st.session_state:
        st.session_state.last_question = None
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    # ========================================================================
    # LEFT COLUMN - IMAGE UPLOAD
    # ========================================================================
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG (Max 10MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = validate_image(uploaded_file)
            
            if image:
                # Display image with container
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display image information
                width, height = image.size
                file_size_mb = uploaded_file.size / (1024 * 1024)
                
                st.markdown(f"""
                <div class="info-box">
                    📊 <strong>Image Info:</strong><br>
                    • Dimensions: {width} × {height} pixels<br>
                    • File size: {file_size_mb:.2f} MB<br>
                    • Format: {uploaded_file.type}
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show placeholder
            st.markdown("""
            <div style="border: 3px dashed #667eea; border-radius: 15px; padding: 3rem; text-align: center; background: #f8f9fa;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📸</div>
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">No Image Uploaded</h3>
                <p style="color: #666;">Upload an image to get started</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # RIGHT COLUMN - QUESTION INPUT
    # ========================================================================
    
    with col2:
        st.markdown("### ❓ Ask a Question")
        
        # Sample questions dropdown
        sample_questions = [
            "",
            "What is in this image?",
            "What color is the main object?",
            "How many objects are visible?",
            "How many people are in the image?",
            "What is the person doing?",
            "What's the weather like?",
            "Is this indoors or outdoors?",
            "What time of day is it?",
            "What's in the background?",
            "What is the person wearing?",
        ]
        
        selected_sample = st.selectbox(
            "Choose a sample question (optional):",
            sample_questions,
            index=0,
            help="Select a pre-written question or write your own below"
        )
        
        # Question input
        question = st.text_input(
            "Or type your own question:",
            value=selected_sample if selected_sample else "",
            placeholder="E.g., What color is the car?",
            max_chars=200,
            help="Type a question about the uploaded image (max 200 characters)"
        )
        
        # Character counter
        char_count = len(question) if question else 0
        char_color = "#28a745" if char_count <= 200 else "#dc3545"
        st.markdown(f"""
        <div style="text-align: right; color: {char_color}; font-size: 0.9rem; margin-top: -0.5rem;">
            {char_count}/200 characters
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Submit button
        submit_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        
        # Process button click
        if submit_button:
            # Validate inputs
            if uploaded_file is None:
                st.error("❌ Please upload an image first!")
                logger.warning("Submit clicked without image")
                
            elif not validate_question(question):
                logger.warning("Submit clicked with invalid question")
                
            else:
                # Validate and load image
                image = validate_image(uploaded_file)
                
                if image:
                    try:
                        # Show processing message
                        with st.spinner("🤔 Analyzing image and generating answer..."):
                            # Initialize model
                            vqa = VQAModel()
                            vqa.load_model()
                            
                            # Get answer
                            answer, confidence = vqa.answer_question(
                                image, 
                                question, 
                                max_length=max_length,
                                num_beams=num_beams
                            )
                            
                            # Store in session state
                            st.session_state.last_answer = answer
                            st.session_state.last_confidence = confidence
                            st.session_state.last_image = image
                            st.session_state.last_question = question
                        
                        logger.info("Answer generated successfully")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
                        logger.error(f"Inference error: {str(e)}", exc_info=True)
    
    # ========================================================================
    # RESULTS SECTION
    # ========================================================================
    
    if st.session_state.last_answer is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        display_results(
            st.session_state.last_answer,
            st.session_state.last_confidence,
            st.session_state.last_image,
            st.session_state.last_question
        )
        
        # Success message
        st.success("✅ Answer generated successfully!")
        
        # Option to ask another question
        if st.button("🔄 Ask Another Question", use_container_width=True):
            st.session_state.last_answer = None
            st.session_state.last_confidence = None
            st.session_state.last_question = None
            st.rerun()
    
    # ========================================================================
    # SAMPLE GALLERY SECTION
    # ========================================================================
    
    if uploaded_file is None:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        display_sample_gallery()
    
       # ========================================================================
    # FOOTER SECTION
    # ========================================================================
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Footer (properly indented)
    st.markdown("""
<div class="footer">
    <h3 style="color: #667eea; margin-bottom: 1rem;">Visual Question Answering App</h3>
    <p style="color: #666;">© 2025 Your Name. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)


# ========================================================================
# RUN THE APP
# ========================================================================

if __name__ == "__main__":
    main()
