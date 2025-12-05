"""
Enhanced Inference Pipeline for Chatbot Mode
Supports conversational interactions with context awareness
"""

from PIL import Image
import time
from typing import Dict, Optional, List

class ChatbotInferencePipeline:
    """Enhanced inference pipeline with chat support"""
    
    def __init__(self, model_manager, rag_system):
        self.model_manager = model_manager
        self.rag_system = rag_system
        self.conversation_history = []
        self.current_animal = None
        self.current_classification = None
        
    def reset_conversation(self):
        """Reset conversation context"""
        self.conversation_history = []
        self.current_animal = None
        self.current_classification = None
    
    def classify_animal(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """Quick classification with intelligent fallback"""
        try:
            image = Image.open(image_path).convert("RGB")
            result = self._classify_with_fallback(image, confidence_threshold)
            success = result["confidence"] >= confidence_threshold
            
            # Store current classification
            if success:
                self.current_animal = result["label"]
                self.current_classification = result
            
            return {
                "success": success,
                "animal_name": result["label"],
                "confidence": result["confidence"],
                "model_used": result["model"],
                "fallback_used": result.get("fallback_used", False),
                "primary_result": result.get("primary_result"),
                "top_3": result.get("top_3", []),
                "threshold": confidence_threshold,
                "message": result.get("message", "Classification successful" if success else "Confidence below threshold")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Classification failed"
            }
    
    def _classify_with_fallback(self, image: Image.Image, threshold: float) -> Dict:
        """Smart classification with fallback logic - Check API confidence >= 0.5"""
        api_result = None
        fallback_result = None
        
        # Try API First
        if self.model_manager.api_available:
            try:
                api_result = self.model_manager._classify_with_api(image)
                api_conf = api_result.get("confidence", 0)

                # ✅ MODIFIED LOGIC: If API confidence >= 0.5, use API result
                if api_conf >= 0.5 and api_result.get("label") not in ["Unknown", None, ""]:
                    return {
                        **api_result,
                        "fallback_used": False,
                        "message": "Classified with API (WildArabia)"
                    }
                # If API confidence < 0.5, continue to fallback
                    
            except Exception as e:
                print(f"API failed: {e}")
        
        # Try Fallback Model
        if self.model_manager.fallback_available:
            try:
                fallback_result = self.model_manager._classify_with_fallback(image)
                fallback_conf = fallback_result.get("confidence", 0)
                
                if fallback_conf >= threshold:
                    return {
                        **fallback_result,
                        "fallback_used": True,
                        "primary_result": api_result,
                        "message": "Using ConvNeXtV2-Tiny (higher confidence)"
                    }
                
                return {
                    "label": "Unknown",
                    "confidence": max(
                        api_result.get("confidence", 0) if api_result else 0,
                        fallback_conf
                    ),
                    "model": "Fallback Model",
                    "fallback_used": True,
                    "primary_result": api_result,
                    "fallback_result": fallback_result,
                    "message": (
                        f"Both API and fallback below threshold "
                        f"(API={api_result.get('confidence', 0) if api_result else 'N/A'}, "
                        f"Fallback={fallback_conf:.2f})"
                    )
                }   
            except Exception as e:
                print(f"Fallback also failed: {e}")
                if api_result:
                    return {
                        **api_result,
                        "fallback_used": False,
                        "message": "API result (ConvNeXt failed)"
                    }
                else:
                    raise Exception("Both models failed")
        
        if api_result:
            return {
                **api_result,
                "fallback_used": False,
                "message": "API result (ConvNeXt unavailable)"
            }
        
        raise Exception("No models available")
    
    def chat_about_animal(
        self,
        question: str,
        image_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Chat mode: Answer questions about the current or new animal
        
        Args:
            question: User's question
            image_path: Optional new image (if None, uses current animal)
            confidence_threshold: Classification threshold
            
        Returns:
            dict with conversational response
        """
        total_start = time.time()
        timing = {}
        
        # Check if Qwen is available
        if not self.model_manager.qwen_available:
            return {
                "success": False,
                "error": "Qwen model is not loaded",
                "message": "Text generation unavailable",
                "timing": {"total": round(time.time() - total_start, 3)}
            }
        
        try:
            # If new image provided, classify it
            if image_path:
                print("\n🔍 Classifying new image...")
                classify_start = time.time()
                classification = self.classify_animal(image_path, confidence_threshold)
                timing['classification'] = round(time.time() - classify_start, 3)
                
                if not classification["success"]:
                    return {
                        "success": False,
                        "classification": classification,
                        "error": "Classification failed or below threshold",
                        "timing": timing
                    }
                
                animal_name = classification["animal_name"]
            elif self.current_animal:
                # Use existing animal
                animal_name = self.current_animal
                classification = self.current_classification
                timing['classification'] = 0  # No new classification
            else:
                return {
                    "success": False,
                    "error": "No animal to discuss. Please provide an image.",
                    "timing": timing
                }
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": question,
                "animal": animal_name
            })
            
            # Retrieve context from RAG
            print(f"\n📚 Retrieving knowledge about {animal_name}...")
            retrieval_start = time.time()
            context_info = self.rag_system.retrieve_knowledge(animal_name, question)
            timing['retrieval'] = round(time.time() - retrieval_start, 3)
            
            context_text = context_info.get('context', '')
            source = context_info.get('source', 'Unknown')
            
            # Generate conversational response
            print(f"\n✨ Generating response...")
            generation_start = time.time()
            
            response = self._generate_conversational_response(
                animal_name=animal_name,
                question=question,
                context=context_text,
                conversation_history=self.conversation_history[-5:]  # Last 5 exchanges
            )
            
            timing['generation'] = round(time.time() - generation_start, 3)
            timing['total'] = round(time.time() - total_start, 3)
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "animal": animal_name
            })
            
            # Return complete result
            return {
                "success": True,
                "classification": {
                    "animal_name": animal_name,
                    "confidence": classification.get("confidence", 0),
                    "model_used": classification.get("model_used", "Unknown"),
                    "fallback_used": classification.get("fallback_used", False),
                    "top_3": classification.get("top_3", [])
                } if classification else None,
                "knowledge_base": {
                    "source": source,
                    "has_context": bool(context_text and len(context_text) > 100),
                    "context_length": len(context_text) if context_text else 0
                },
                "analysis": response,
                "question": question,
                "timing": timing,
                "processing_time": timing['total']
            }
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            timing['total'] = round(time.time() - total_start, 3)
            return {
                "success": False,
                "error": str(e),
                "message": "Chat failed",
                "timing": timing
            }
    
    def _generate_conversational_response(
        self,
        animal_name: str,
        question: str,
        context: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate a natural conversational response"""
        
        # Build conversation context
        history_text = ""
        if len(conversation_history) > 1:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[:-1]:  # Exclude current question
                if msg["role"] == "user":
                    history_text += f"User: {msg['content']}\n"
                else:
                    history_text += f"Assistant: {msg['content'][:200]}...\n"
        
        # Create prompt based on question type
        if any(word in question.lower() for word in ["what", "tell me", "describe", "explain"]):
            # Informational question
            if context and len(context) > 100:
                prompt = f"""You are a friendly wildlife expert having a conversation about a {animal_name}.

The user asks: "{question}"
{history_text}

Based on this knowledge:
{context[:2500]}

Provide a natural, conversational response that:
- Directly answers their question
- Is friendly and engaging
- Includes interesting details
- Is concise (2-4 paragraphs)
- Refers to the animal naturally

Don't use formal sections or bullet points. Just chat naturally like an expert friend."""
            else:
                prompt = f"""You are a friendly wildlife expert having a conversation about a {animal_name}.

The user asks: "{question}"
{history_text}

Provide a natural, conversational response about the {animal_name}:
- Share what you know about them
- Be engaging and friendly
- Keep it concise (2-4 paragraphs)
- Include interesting facts

Chat naturally like an expert friend."""
        
        elif any(word in question.lower() for word in ["how", "why", "when", "where"]):
            # Specific question
            prompt = f"""You are a friendly wildlife expert discussing a {animal_name}.

The user asks: "{question}"
{history_text}

{"Based on this information:\n" + context[:2500] if context else "Based on your knowledge:"}

Answer their specific question naturally:
- Be direct and clear
- Provide relevant details
- Stay conversational
- Keep it focused (1-3 paragraphs)"""
        
        elif any(word in question.lower() for word in ["can", "do", "does", "is", "are"]):
            # Yes/no question
            prompt = f"""You are a friendly wildlife expert discussing a {animal_name}.

The user asks: "{question}"
{history_text}

{"Based on this information:\n" + context[:2500] if context else "Based on your knowledge:"}

Answer their yes/no question:
- Start with a clear answer (yes/no)
- Explain why
- Add interesting context
- Keep it conversational (1-2 paragraphs)"""
        
        else:
            # General conversation
            prompt = f"""You are a friendly wildlife expert having a conversation about a {animal_name}.

The user says: "{question}"
{history_text}

{"Based on this knowledge:\n" + context[:2500] if context else ""}

Respond naturally and helpfully:
- Address what they said/asked
- Be friendly and engaging
- Share relevant information
- Keep it conversational"""
        
        return self.model_manager.generate_with_qwen(
            prompt=prompt,
            max_tokens=800,
            temperature=0.7
        )
    
    # Keep original method for backward compatibility
    def analyze_complete(
        self,
        image_path: str,
        question: str = "Tell me about this animal",
        confidence_threshold: float = 0.5
    ) -> Dict:
        """Original complete analysis method"""
        return self.chat_about_animal(
            question=question,
            image_path=image_path,
            confidence_threshold=confidence_threshold
        )