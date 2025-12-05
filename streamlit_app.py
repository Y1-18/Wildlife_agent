"""
Streamlit Frontend for Wildlife Analysis System
Connects to FastAPI backend with ConvNeXtV2-Tiny fallback
"""

import streamlit as st
import requests
from PIL import Image
import io
import json

# Configuration
API_URL = "http://localhost:8080"

# Page config
st.set_page_config(
    page_title="Wild Arabia - Wildlife Analysis",
    page_icon="🦁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c5f2d;
        padding: 20px;
        background: linear-gradient(135deg, #97b498 0%, #2c5f2d 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2c5f2d;
        color: white;
        border-radius: 5px;
        padding: 10px 30px;
        font-size: 16px;
    }
    .result-box {
        background-color: #f0f7f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2c5f2d;
        margin: 10px 0;
    }
    .model-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px;
    }
    .primary-model {
        background-color: #4CAF50;
        color: white;
    }
    .fallback-model {
        background-color: #2196F3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None


def analyze_image(image_file, question):
    """Send image to backend for analysis"""
    files = {"file": ("image.jpg", image_file, "image/jpeg")}
    data = {"question": question}
    
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            files=files,
            data=data,
            timeout=300  # 5 minutes timeout
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be loading or the image is too complex. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Make sure FastAPI server is running on port 8080."}
    except Exception as e:
        return {"error": str(e)}


def classify_image(image_file):
    """Send image to backend for classification"""
    files = {"file": ("image.jpg", image_file, "image/jpeg")}
    data = {"confidence_threshold": 0.5}
    
    try:
        response = requests.post(
            f"{API_URL}/classify",
            files=files,
            data=data,
            timeout=120  # 2 minutes timeout
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be loading. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Make sure FastAPI server is running on port 8080."}
    except Exception as e:
        return {"error": str(e)}


# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦁 Wild Arabia</h1>
        <p>AI-Powered Wildlife Identification & Analysis</p>
        <p style="font-size: 14px; margin-top: 10px;">
            <span class="model-badge primary-model">WildArabia API (Primary)</span>
            <span class="model-badge fallback-model">ConvNeXtV2-Tiny (Fallback)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check backend status
    is_healthy, health_info = check_backend_health()
    
    if not is_healthy:
        st.error("⚠️ Backend server is not running! Please start the FastAPI server first.")
        st.code("python main.py", language="bash")
        st.stop()
    else:
        # Show detailed health info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Backend Connected")
        with col2:
            if health_info and health_info.get('models', {}).get('vision_api'):
                st.info("🎯 Primary: WildArabia API")
            else:
                st.warning("⚠️ Primary: Unavailable")
        with col3:
            if health_info and health_info.get('models', {}).get('vision_fallback'):
                st.info("🔄 Fallback: ConvNeXtV2-Tiny")
            else:
                st.warning("⚠️ Fallback: Unavailable")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "📤 Upload & Analyze", "ℹ️ About", "🔧 Services"]
    )
    
    # Model info in sidebar
    if health_info:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Loaded Models")
        models = health_info.get('models', {})
        
        if models.get('vision_api'):
            st.sidebar.success("✅ Primary: WildArabia API")
            st.sidebar.caption("   ConvNeXt - Arabian species")
        
        if models.get('vision_fallback'):
            st.sidebar.success("✅ Fallback: ConvNeXtV2-Tiny")
            st.sidebar.caption("   22k pretrained - 21k classes")
        
        if models.get('qwen_available'):
            st.sidebar.success("✅ Text Gen: Qwen 2.5-3B")
            st.sidebar.caption("   Local inference")
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "📤 Upload & Analyze":
        show_upload()
    elif page == "ℹ️ About":
        show_about()
    elif page == "🔧 Services":
        show_services()


def show_home():
    """Home page"""
    st.title("Welcome to Wild Arabia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌍 Discover Arabian Wildlife
        
        Our AI-powered system helps identify and learn about the diverse wildlife 
        of the Arabian Peninsula. Using advanced computer vision and natural language 
        processing, we provide detailed information about animals in their natural habitat.
        
        **Features:**
        - 🔍 Instant animal identification
        - 🤖 Dual-model approach (API + ConvNeXtV2)
        - 📚 Comprehensive species information
        - 🎯 AI-powered analysis with RAG
        - 📊 Detailed behavioral insights
        
        **Models:**
        - **Primary**: WildArabia API - Arabian species
        - **Fallback**: ConvNeXtV2-Tiny - 21k classes
        - **Text**: Qwen 2.5-3B - Local inference
        """)
        
        if st.button("🚀 Start Analyzing", type="primary"):
            st.session_state.page = "📤 Upload & Analyze"
            st.rerun()
    
    with col2:
        st.image("https://images.unsplash.com/photo-1564760055775-d63b17a55c44", 
                 caption="Arabian Wildlife", use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 System Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Primary Model", "API", "WildArabia")
    col2.metric("Fallback Model", "ConvNeXtV2", "Tiny-22k")
    col3.metric("Text Generation", "Qwen 2.5", "3B params")
    col4.metric("Processing", "< 5s", "Average time")


def show_upload():
    """Upload and analyze page"""
    st.title("📤 Upload & Analyze Wildlife")
    
    # Two modes: Quick classify or Full analysis
    mode = st.radio("Choose analysis mode:", 
                    ["🔍 Quick Classification", "📋 Full Analysis (with RAG)"])
    
    uploaded_file = st.file_uploader(
        "Upload an animal image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of an animal for identification"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if mode == "🔍 Quick Classification":
            if st.button("🔍 Classify Animal", type="primary"):
                with st.spinner("Analyzing image with dual-model system..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    result = classify_image(uploaded_file)
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader("Classification Result")
                        
                        # Show which model was used
                        if result.get("fallback_used"):
                            st.info("🔄 Used ConvNeXtV2-Tiny fallback model")
                        else:
                            st.success("🎯 Used WildArabia API primary model")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Animal Detected", result.get("animal_name", "Unknown"))
                        with col2:
                            confidence = result.get("confidence", 0)
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Show top 3 predictions
                        if result.get("top_3"):
                            st.markdown("**Top 3 Predictions:**")
                            for i, (label, prob) in enumerate(result["top_3"], 1):
                                st.write(f"{i}. {label}: {prob*100:.1f}%")
                        
                        if result.get("success"):
                            st.success("✅ Classification successful!")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show full response in expander
                        with st.expander("📄 View Full Response"):
                            st.json(result)
        
        else:  # Full Analysis
            question = st.text_input(
                "Ask a question about the animal:",
                value="Tell me about this animal",
                help="Enter a specific question or use the default"
            )
            
            if st.button("📋 Analyze with RAG", type="primary"):
                with st.spinner("Running full analysis pipeline... This may take 1-3 minutes for the first request."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    result = analyze_image(uploaded_file, question)
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                        st.info("💡 Tip: The first request can take longer as models are loading. Please try again.")
                    else:
                        # Classification section
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader("🔍 Classification")
                        
                        # Show which model was used
                        classification = result.get("classification", {})
                        if classification.get("fallback_used"):
                            st.info("🔄 Used ConvNeXtV2-Tiny fallback model (higher confidence)")
                        else:
                            st.success("🎯 Used WildArabia API primary model")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Animal", classification.get("animal_name", "Unknown"))
                        with col2:
                            confidence = classification.get("confidence", 0)
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        with col3:
                            processing_time = result.get("processing_time", 0)
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        
                        # Show top 3
                        if classification.get("top_3"):
                            st.markdown("**Top 3 Predictions:**")
                            for i, (label, prob) in enumerate(classification["top_3"], 1):
                                st.write(f"{i}. {label}: {prob*100:.1f}%")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # AI Analysis section
                        if "analysis" in result:
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("🤖 AI Analysis (Qwen 2.5-3B)")
                            st.markdown(result["analysis"])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Timing breakdown
                        if "timing" in result:
                            with st.expander("⏱️ Performance Breakdown"):
                                timing = result["timing"]
                                st.write(f"- Classification: {timing.get('classification', 0):.3f}s")
                                st.write(f"- Knowledge Retrieval: {timing.get('retrieval', 0):.3f}s")
                                st.write(f"- Text Generation: {timing.get('generation', 0):.3f}s")
                                st.write(f"- **Total**: {timing.get('total', 0):.3f}s")
                        
                        # Knowledge base context
                        if "knowledge_base" in result:
                            kb = result["knowledge_base"]
                            with st.expander(f"📚 Knowledge Base ({kb.get('source', 'Unknown')})"):
                                st.write(f"Has context: {kb.get('has_context', False)}")
                                st.write(f"Context length: {kb.get('context_length', 0)} chars")
                        
                        # Full response
                        with st.expander("📄 View Full Response"):
                            st.json(result)


def show_about():
    """About page"""
    st.title("ℹ️ About Wild Arabia")
    
    st.markdown("""
    ### 🎯 Our Mission
    
    Wild Arabia is an AI-powered platform dedicated to wildlife conservation and education 
    in the Arabian Peninsula. We combine cutting-edge computer vision and natural language 
    processing to help researchers, educators, and nature enthusiasts identify and learn 
    about Arabian wildlife.
    
    ### 🔬 Technology Stack
    
    **Vision Models (Dual-System):**
    - **Primary**: WildArabia API (ConvNeXt) - Specialized Arabian species
    - **Fallback**: ConvNeXtV2-Tiny-22k - 21k ImageNet-22k classes
    
    **Language Model:**
    - **Qwen 2.5-3B-Instruct**: Local inference, no API keys needed
    
    **RAG System:**
    - ChromaDB for enhanced knowledge retrieval
    - Wikipedia fallback for comprehensive coverage
    
    **Backend & Frontend:**
    - FastAPI with async processing
    - Streamlit for interactive UI
    
    ### 🌟 Features
    
    1. **Dual-Model Classification**: API for specialized Arabian wildlife, ConvNeXt for broader coverage
    2. **Intelligent Fallback**: Automatically uses the best model based on confidence
    3. **Real-time Analysis**: AI-powered insights using RAG pipeline
    4. **Knowledge Base**: Comprehensive information about Arabian wildlife
    5. **User-Friendly**: Simple interface for anyone to use
    6. **Privacy-First**: Local text generation, minimal external API usage
    
    ### 👥 Team
    
    Developed by wildlife conservation enthusiasts and AI researchers passionate 
    about protecting Arabian biodiversity.
    
    ### 📧 Contact
    
    For questions, feedback, or collaboration opportunities, please reach out to us.
    """)


def show_services():
    """Services page"""
    st.title("🔧 Services & API")
    
    st.markdown("""
    ### 🛠️ Available Services
    
    Our platform offers multiple ways to interact with the wildlife analysis system:
    """)
    
    # Service cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔍 Quick Classification
        
        - Fast animal identification
        - Dual-model system (API + ConvNeXtV2)
        - Confidence scoring
        - Automatic fallback
        - < 2 second response time
        
        **Use case**: Quick field identification
        """)
        
        st.markdown("""
        #### 📋 Full Analysis (RAG)
        
        - Complete species analysis
        - Knowledge base integration
        - Custom questions
        - Qwen-powered insights
        - Detailed reports
        
        **Use case**: Research & education
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Dual-Model System
        
        - WildArabia API (Primary): Arabian species
        - ConvNeXtV2-Tiny (Fallback): 21k classes
        - Intelligent confidence-based selection
        - Best of both worlds
        
        **Use case**: Maximum accuracy
        """)
        
        st.markdown("""
        #### 🔌 API Access
        
        - RESTful API
        - JSON responses
        - Easy integration
        - Full documentation
        
        **Use case**: Third-party applications
        """)
    
    # API Documentation
    st.markdown("---")
    st.subheader("📖 API Endpoints")
    
    with st.expander("🔍 POST /classify"):
        st.code("""
# Quick classification endpoint (dual-model)
curl -X POST http://localhost:8080/classify \\
  -F "file=@animal.jpg" \\
  -F "confidence_threshold=0.5"
        """, language="bash")
    
    with st.expander("📋 POST /analyze"):
        st.code("""
# Full analysis with RAG (WildArabia API + ConvNeXtV2 + Qwen)
curl -X POST http://localhost:8080/analyze \\
  -F "file=@animal.jpg" \\
  -F "question=Tell me about this animal" \\
  -F "confidence_threshold=0.5"
        """, language="bash")
    
    with st.expander("❤️ GET /health"):
        st.code("""
# Check system health and model status
curl http://localhost:8080/health
        """, language="bash")
    
    # Test backend connection
    st.markdown("---")
    st.subheader("🧪 Test Backend Connection")
    
    if st.button("Test Connection"):
        with st.spinner("Testing..."):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Backend is responsive!")
                    with st.expander("View Health Status"):
                        st.json(response.json())
                else:
                    st.error(f"❌ Backend returned status {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")


if __name__ == "__main__":
    main()