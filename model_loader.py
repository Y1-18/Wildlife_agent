"""
Model Manager with API Integration
Uses Wild Arabia API instead of local model
Keeps fallback model and Qwen integration
"""

import torch
import os
import json
import requests
import io
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict
import base64
from io import BytesIO

class ModelManager:
    """Manages vision models (API + fallback) and Qwen text generation"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # API configuration
        self.api_url = "https://wild-arabia-api-524882900334.us-central1.run.app/analyze"
        self.api_available = False
        
        # Fallback vision model
        self.fallback_processor = None
        self.fallback_model = None
        self.fallback_available = False
        self.fallback_model_name = "facebook/convnextv2-tiny-22k-224"
        
        # Qwen model for text generation
        self.qwen_model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.qwen_tokenizer = None
        self.qwen_model = None
        self.qwen_available = False
        
    def load_all_models(self):
        """Load fallback vision model and Qwen model, test API"""
        print("\n" + "="*60)
        print("📦 LOADING MODELS")
        print("="*60 + "\n")
        
        # Test API availability
        self._test_api()
        
        # Load fallback vision model
        self._load_fallback_model()
        
        # Load Qwen model
        self._load_qwen_model()
        
        print("\n" + "="*60)
        print("✅ MODEL LOADING COMPLETE")
        print("="*60)
        print(f"Primary Vision (API): {'✅ Available' if self.api_available else '❌ Unavailable'}")
        print(f"Fallback Vision (ConvNeXtV2): {'✅ Loaded' if self.fallback_available else '❌ Failed'}")
        print(f"Text Generation (Qwen): {'✅ Loaded' if self.qwen_available else '❌ Failed'}")
        print("="*60 + "\n")
        
        if not self.api_available and not self.fallback_available:
            raise RuntimeError("❌ Failed to load any vision model")
    
    def _test_api(self):
        """Test if API is available"""
        print("🌐 Testing Wild Arabia API...")
        print(f"   URL: {self.api_url}")
        
        try:
            # Create a small test image
            test_image = Image.new('RGB', (224, 224), color='white')
            buffered = BytesIO()
            test_image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Try multipart/form-data format (most common for image APIs)
            print("   Testing with multipart/form-data...")
            try:
                files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
                response = requests.post(
                    self.api_url,
                    files=files,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.api_available = True
                    print(f"   ✅ API is available (multipart format)")
                    return
                else:
                    print(f"   ⚠️ Multipart returned status {response.status_code}: {response.text[:100]}")
            except Exception as e:
                print(f"   ⚠️ Multipart test failed: {e}")
            
            # Try JSON format
            print("   Testing with JSON format...")
            try:
                response = requests.post(
                    self.api_url,
                    json={"image": img_base64},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.api_available = True
                    print(f"   ✅ API is available (JSON format)")
                    return
                else:
                    print(f"   ⚠️ JSON returned status {response.status_code}: {response.text[:100]}")
            except Exception as e:
                print(f"   ⚠️ JSON test failed: {e}")
            
            print(f"   ⚠️ API test failed with all formats")
            print(f"   Will use fallback model if available")
            self.api_available = False
                
        except Exception as e:
            print(f"   ⚠️ API test failed: {e}")
            print(f"   Will use fallback model if available")
            self.api_available = False
    
    def _load_fallback_model(self):
        """Load fallback vision model"""
        print(f"\n📦 Loading Fallback Model: {self.fallback_model_name}")
        
        try:
            print("   Loading processor...")
            self.fallback_processor = AutoImageProcessor.from_pretrained(self.fallback_model_name)
            print("   ✅ Processor loaded")
            
            print("   Loading model...")
            self.fallback_model = AutoModelForImageClassification.from_pretrained(
                self.fallback_model_name
            ).to(self.device)
            self.fallback_model.eval()
            print("   ✅ Model loaded")
            
            self.fallback_available = True
            print(f"\n   🎉 FALLBACK MODEL LOADED SUCCESSFULLY!")
            print(f"   Model: {self.fallback_model_name}")
            print(f"   Type: ConvNeXtV2-Tiny - Pretrained on ImageNet-22k!")
            print(f"   ImageNet-22k classes: 21,841")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"\n   ❌ FALLBACK MODEL FAILED TO LOAD")
            print(f"   Error: {e}")
            print(f"   Error Type: {type(e).__name__}")
            self.fallback_available = False
    
    def _load_qwen_model(self):
        """Load Qwen model for text generation"""
        print(f"\n🤖 Loading Qwen model: {self.qwen_model_name}")
        
        try:
            # Load tokenizer
            print("   Loading tokenizer...")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model_name,
                trust_remote_code=True
            )
            print("   ✅ Tokenizer loaded")
            
            # Load model with optimizations
            print("   Loading model (this may take a while)...")
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                self.qwen_model = self.qwen_model.to(self.device)
            
            self.qwen_model.eval()
            self.qwen_available = True
            
            print(f"\n   🎉 QWEN MODEL LOADED SUCCESSFULLY!")
            print(f"   Model: {self.qwen_model_name}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"\n   ❌ QWEN MODEL FAILED TO LOAD")
            print(f"   Error: {e}")
            print(f"   Text generation will be unavailable")
            self.qwen_available = False
    
    def validate_qwen_available(self) -> Dict:
        """Check if Qwen model is available"""
        if self.qwen_available:
            return {
                "success": True,
                "message": "Qwen model is loaded and ready",
                "model": self.qwen_model_name
            }
        else:
            return {
                "success": False,
                "message": "Qwen model is not available. Please check model loading."
            }
    
    # ==========================
    # CLASSIFICATION METHODS - API FIRST, THEN FALLBACK
    # ==========================
    
    def _classify_with_api(self, image: Image.Image) -> Dict:
        """
        Classify using Wild Arabia API (Multipart only)
        """
        if not self.api_available:
            raise Exception("API not available")

        try:
            # Convert PIL image → bytes
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            # Send multipart/form-data
            files = {
                "file": ("image.jpg", img_bytes, "image/jpeg")
            }

            response = requests.post(
                self.api_url,
                files=files,
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"API returned {response.status_code}: {response.text}")

            result = response.json()

            label = result.get("scientific_name", "Unknown")

            return {
                "label": label,
                "confidence": 1.0,   # API doesn't return confidence → set as fixed
                "model": "API (WildArabia)",
                "top_3": [(label, 1.0)]  # API doesn't have top_3 → repeat same answer
            }

        except Exception as e:
            raise Exception(f"API classification failed: {str(e)}")

    
    def _classify_with_fallback(self, image: Image.Image) -> Dict:
        """
        Classify with Fallback Model (ConvNeXtV2-Tiny)
        
        Args:
            image: PIL Image
            
        Returns:
            dict with label, confidence, model name, and top_3
            
        Raises:
            Exception if fallback model not available
        """
        if not self.fallback_available:
            raise Exception("Fallback model not available")
        
        try:
            inputs = self.fallback_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.fallback_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Get top prediction
            predicted_idx = logits.argmax(-1).item()
            confidence = probs[predicted_idx].item()
            label = self.fallback_model.config.id2label[predicted_idx]
            
            # Get top 3
            top3_indices = torch.topk(probs, 3).indices.tolist()
            top3_results = [(self.fallback_model.config.id2label[idx], probs[idx].item())
                           for idx in top3_indices]
            
            return {
                "label": label,
                "confidence": confidence,
                "model": f"Fallback (ConvNeXtV2-Tiny)",
                "top_3": top3_results
            }
            
        except Exception as e:
            raise Exception(f"Fallback model classification failed: {str(e)}")
    
    def classify_image(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """
        Legacy method for backward compatibility
        Uses API first, then fallback if needed
        
        Args:
            image: PIL Image
            confidence_threshold: Minimum confidence (not used here, handled in inference)
            
        Returns:
            Classification result
        """
        api_result =None
        # Try API first
        if self.api_available:
            try:
                api_result = self._classify_with_api(image)
                api_label = api_result.get("label", "Unknown")
                if api_label.lower() == "unknown":
                    api_conf = 0.
                if api_conf < confidence_threshold:
                    print(f"⚠️ API confidence too low ({api_conf}), switching to fallback...")
                    raise Exception("Low confidence")
                return api_result
  
            except Exception as e:
                print(f"   ⚠️ API failed: {e}, trying fallback...")
        
        # Fallback
        if self.fallback_available:
            try:
                return self._classify_with_fallback(image)
            except Exception as e:
                print(f"   ⚠️ Fallback also failed: {e}")
        
        raise Exception("No models available for classification")
    
    # ==========================
    # TEXT GENERATION WITH QWEN
    # ==========================
    
    def generate_with_qwen(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate text using Qwen model"""
        if not self.qwen_available:
            return "⚠️ Qwen model is not available. Please check model loading."
        
        try:
            # Prepare messages in chat format
            messages = [
                {"role": "system", "content": "You are a knowledgeable wildlife expert providing detailed and accurate information about animals."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=40,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.pad_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response
            
        except Exception as e:
            return f"❌ Qwen generation error: {str(e)}"
    
    def generate_analysis(
        self,
        animal_name: str,
        context: str,
        question: str
    ) -> str:
        """Generate comprehensive wildlife analysis using Qwen"""
        # Check if we have good context
        if context and len(context) > 100:
            # With context from knowledge base
            prompt = f"""Based on the classification and available knowledge, provide a comprehensive summary about the {animal_name}.

Classification Details:
- Animal: {animal_name}

Available Knowledge:
{context[:3000]}

Please create a well-structured summary covering:
1. **Species Overview**: Scientific classification and common names
2. **Physical Characteristics**: Appearance, size, distinctive features
3. **Habitat & Distribution**: Where they live, geographic range
4. **Behavior & Diet**: Hunting/feeding habits, social structure
5. **Conservation Status**: Current threats and protection efforts
6. **Interesting Facts**: Unique characteristics or behaviors

Keep it informative, accurate, and organized with clear sections."""
        else:
            # Limited knowledge - generate from general knowledge
            prompt = f"""Provide comprehensive information about the {animal_name}:

Please structure your response with these sections:
1. **Species Overview**: Scientific classification and common names
2. **Physical Characteristics**: Appearance, size, distinctive features
3. **Habitat & Distribution**: Where they live, geographic range
4. **Behavior & Diet**: Hunting/feeding habits, social structure
5. **Conservation Status**: Current threats and protection efforts
6. **Interesting Facts**: Unique characteristics or behaviors

Be detailed and educational."""

        return self.generate_with_qwen(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )