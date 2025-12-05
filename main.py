"""
FastAPI Wildlife Chatbot System with Qwen Integration
Supports conversational interactions about wildlife
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import torch
import time
import os
import pathlib
from typing import Optional
from pydantic import BaseModel

# Import modules
from model_loader import ModelManager
from rag_utils import RAGSystem
from inference import ChatbotInferencePipeline

# Global instances
model_manager = None
rag_system = None
inference_pipeline = None

# Lightning AI environment detection
LIGHTNING_AI = os.getenv("LIGHTNING_APP_STATE") is not None
PORT = int(os.getenv("PORT", 8080))
HOST = "0.0.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown"""
    global model_manager, rag_system, inference_pipeline
    
    print("\n" + "="*80)
    print("🚀 INITIALIZING WILDLIFE CHATBOT SYSTEM")
    print("🦁 Vision: WildArabia API (Primary) + ConvNeXtV2-Tiny (Fallback)")
    print("🤖 Text Generation: Qwen 2.5-3B (Conversational)")
    print("💬 Mode: Interactive Chat")
    print("="*80 + "\n")
    
    try:
        # Initialize model manager
        print("📦 Loading models...")
        model_manager = ModelManager()
        model_manager.load_all_models()
        
        # Initialize RAG system
        print("\n📚 Initializing knowledge base...")
        rag_system = RAGSystem()
        rag_system.initialize()
        
        # Initialize chatbot inference pipeline
        print("\n🔧 Setting up chatbot inference pipeline...")
        inference_pipeline = ChatbotInferencePipeline(model_manager, rag_system)
        
        print("\n" + "="*80)
        print("✅ CHATBOT SYSTEM READY")
        print("🤖 Qwen ready for conversational responses")
        print("🖼️ Vision models loaded and ready")
        print("💬 Chat mode enabled")
        print("="*80 + "\n")
        
        yield
        
    except Exception as e:
        print(f"\n❌ STARTUP ERROR: {e}")
        raise
    finally:
        print("\n🔄 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Wildlife Chatbot API",
    description="Conversational AI-powered wildlife identification and chat",
    version="4.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Wildlife Chatbot API - Interactive Conversational System",
        "version": "4.0.0",
        "environment": "Lightning AI" if LIGHTNING_AI else "Local",
        "features": [
            "🦁 Multi-model wildlife classification",
            "💬 Natural conversation about animals",
            "📚 RAG-powered knowledge retrieval",
            "🤖 Qwen 2.5-3B conversational AI",
            "🔄 Context-aware responses"
        ],
        "models": {
            "primary": "WildArabia API (ConvNeXt)",
            "fallback": "facebook/convnextv2-tiny-22k-224",
            "text_generation": "Qwen/Qwen2.5-3B-Instruct"
        },
        "endpoints": {
            "health": "/health",
            "classify": "/classify (POST) - Quick classification",
            "chat": "/chat (POST) - Conversational chat",
            "analyze": "/analyze (POST) - Complete analysis",
            "reset_chat": "/reset_chat (POST) - Reset conversation"
        }
    }


@app.get("/health")
async def health_check():
    """Check system health"""
    if model_manager is None:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "message": "System starting up"}
        )
    
    health_status = {
        "status": "healthy",
        "environment": "Lightning AI" if LIGHTNING_AI else "Local",
        "mode": "chatbot",
        "timestamp": time.time(),
        "models": {
            "qwen_available": model_manager.qwen_available,
            "qwen_model": model_manager.qwen_model_name if model_manager.qwen_available else None,
            "vision_api": model_manager.api_available,
            "vision_api_url": model_manager.api_url if model_manager.api_available else None,
            "vision_fallback": model_manager.fallback_available,
            "vision_fallback_name": model_manager.fallback_model_name if model_manager.fallback_available else None,
        },
        "rag": {
            "vector_db": rag_system.vectordb is not None if rag_system else False,
            "wikipedia_fallback": True
        },
        "device": str(model_manager.device) if model_manager else "unknown",
        "cuda_available": torch.cuda.is_available()
    }
    
    return health_status


@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Quick image classification
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    temp_path = None
    
    try:
        # Validate file
        content_type = file.content_type or ""
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save temporary file
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        temp_path = os.path.join(temp_dir, f"classify_{int(time.time()*1000)}_{file.filename}")
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run classification
        result = inference_pipeline.classify_animal(temp_path, confidence_threshold)
        result["processing_time"] = round(time.time() - start_time, 3)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/chat")
async def chat_about_animal(
    file: Optional[UploadFile] = File(None),
    question: str = Form(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Chat about an animal (with or without new image)
    
    Args:
        file: Optional new image (if not provided, uses previously classified animal)
        question: User's question
        confidence_threshold: Classification threshold
    
    Returns:
        Conversational response about the animal
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    temp_path = None
    
    try:
        # If image provided, save it temporarily
        if file:
            content_type = file.content_type or ""
            if not content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            temp_dir = "/tmp" if os.path.exists("/tmp") else "."
            temp_path = os.path.join(temp_dir, f"chat_{int(time.time()*1000)}_{file.filename}")
            
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
        
        # Chat about the animal
        result = inference_pipeline.chat_about_animal(
            question=question,
            image_path=temp_path,
            confidence_threshold=confidence_threshold
        )
        
        # Add metadata
        result["timestamp"] = time.time()
        result["method"] = "conversational_chat"
        result["api_version"] = "4.0.0"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form("Tell me about this animal"),
    confidence_threshold: float = Form(0.5)
):
    """
    Complete analysis pipeline (for backward compatibility)
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    temp_path = None
    
    try:
        content_type = file.content_type or ""
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        temp_path = os.path.join(temp_dir, f"analyze_{int(time.time()*1000)}_{file.filename}")
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        result = inference_pipeline.analyze_complete(
            image_path=temp_path,
            question=question,
            confidence_threshold=confidence_threshold
        )
        
        result["timestamp"] = time.time()
        result["method"] = "full_analysis"
        result["api_version"] = "4.0.0"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/reset_chat")
async def reset_chat():
    """
    Reset conversation history
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        inference_pipeline.reset_conversation()
        return {
            "success": True,
            "message": "Conversation history reset",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print(f"\n🌩️ Configuration:")
    print(f"   HOST: {HOST}")
    print(f"   PORT: {PORT}")
    print(f"   Environment: {'Lightning AI' if LIGHTNING_AI else 'Local'}")
    print(f"\n🌐 Server available at: http://localhost:{PORT}")
    print(f"📚 API docs: http://localhost:{PORT}/docs")
    print(f"🔍 Health check: http://localhost:{PORT}/health")
    print(f"🔌 Version: 4.0.0 (Chatbot Mode)")
    print(f"💬 Conversational AI with Qwen\n")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
        timeout_keep_alive=300
    )