"""
RAG (Retrieval-Augmented Generation) System
Handles ChromaDB vector database and knowledge retrieval
"""

import os
from typing import Dict, List, Optional, Any

# Suppress ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class RAGSystem:
    """Manages vector database and knowledge retrieval"""
    
    def __init__(self):
        self.vectordb = None
        self.retriever = None
        self.embeddings = None
        self.chroma_dir = None
        
        # Possible ChromaDB locations in Lightning AI
        self.CHROMA_PATHS = [
            "./chroma_db",
            "/teamspace/studios/this_studio/.lightning_studio/chroma_db",
            os.path.expanduser("~/.lightning_studio/chroma_db"),
            "/content/chroma_db",
            "/workspace/chroma_db"
        ]
    
    def initialize(self):
        """Initialize vector database and retrieval system"""
        print("\n" + "="*60)
        print("INITIALIZING RAG SYSTEM")
        print("="*60)
        
        self._load_embeddings()
        self._load_vectordb()
        
        print("="*60)
        if self.vectordb:
            print("✅ RAG SYSTEM READY")
        else:
            print("⚠️  RAG SYSTEM RUNNING WITH WIKIPEDIA FALLBACK")
        print("="*60)
    
    def _load_embeddings(self):
        """Load embedding model"""
        try:
            print("\n📦 Loading embedding model...")
            
            # Try langchain_huggingface first (newer)
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                print("   Using langchain_huggingface")
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                print("   Using langchain_community")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="./embeddings_cache",
                model_kwargs={'device': 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'}
            )
            
            print("   ✅ Embedding model loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to load embeddings: {e}")
            self.embeddings = None
    
    def _load_vectordb(self):
        """Load ChromaDB vector database"""
        if self.embeddings is None:
            print("\n⚠️  Cannot load vector DB without embeddings")
            return
        
        print("\n📚 Searching for ChromaDB...")
        
        # Find ChromaDB directory
        for path in self.CHROMA_PATHS:
            if os.path.exists(path):
                self.chroma_dir = path
                print(f"   ✓ Found ChromaDB at: {path}")
                break
        
        if not self.chroma_dir:
            print(f"   ⚠️  ChromaDB not found")
            print(f"   Searched locations:")
            for path in self.CHROMA_PATHS:
                print(f"      - {path}")
            print(f"\n   💡 You can create ChromaDB or use Wikipedia fallback")
            return
        
        # Load ChromaDB
        try:
            # Try langchain_chroma first (newer)
            try:
                from langchain_chroma import Chroma
                print("   Using langchain_chroma")
            except ImportError:
                from langchain_community.vectorstores import Chroma
                print("   Using langchain_community")
            
            self.vectordb = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
            
            # Create retriever
            self.retriever = self.vectordb.as_retriever(
                search_kwargs={"k": 5}
            )
            
            # Test retrieval
            test_results = self.retriever.invoke("test")
            print(f"   ✅ Vector database loaded successfully")
            print(f"   📊 Test query returned {len(test_results)} results")
            
        except Exception as e:
            print(f"   ❌ Failed to load ChromaDB: {e}")
            self.vectordb = None
            self.retriever = None
    
    def retrieve_knowledge(
        self,
        animal_name: str,
        question: str = ""
    ) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge about an animal
        
        Args:
            animal_name: Name of the animal
            question: Optional specific question
            
        Returns:
            Dict with source, context, and animal_name
        """
        print(f"\n🔍 Retrieving knowledge for: {animal_name}")
        
        # If no vector DB, use Wikipedia immediately
        if self.vectordb is None or self.retriever is None:
            print("   📖 Using Wikipedia (no vector DB)")
            return self._search_wikipedia(animal_name)
        
        try:
            # Clean the animal name
            clean_name = animal_name.split(',')[0].strip()
            
            # Generate search queries
            queries = [
                clean_name,
                animal_name,
                f"{clean_name} habitat behavior",
                f"{clean_name} conservation"
            ]
            
            # Search vector database
            all_results = []
            for query in queries[:2]:  # Use first 2 queries
                try:
                    results = self.retriever.invoke(query)
                    all_results.extend(results)
                except Exception as e:
                    print(f"   ⚠️  Query failed: {e}")
            
            # Remove duplicates
            seen_content = set()
            unique_results = []
            for doc in all_results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(doc)
            
            # Get top results
            top_results = unique_results[:5]
            context = "\n\n".join([d.page_content for d in top_results])
            
            # Check if we got good results
            if not context.strip() or len(context) < 100:
                print("   ⚠️  Vector DB results insufficient, trying Wikipedia...")
                return self._search_wikipedia(animal_name)
            
            print(f"   ✓ Retrieved {len(top_results)} documents from vector DB")
            print(f"   📝 Context length: {len(context)} characters")
            
            return {
                "source": "PDF Knowledge Base",
                "context": context,
                "animal_name": clean_name,
                "num_documents": len(top_results)
            }
            
        except Exception as e:
            print(f"   ❌ Retrieval error: {e}")
            return self._search_wikipedia(animal_name)
    
    def _search_wikipedia(self, animal_name: str) -> Dict[str, Any]:
        """Fallback to Wikipedia search"""
        try:
            import wikipedia
            
            clean_name = animal_name.split(',')[0].strip()
            print(f"   🔍 Searching Wikipedia for: {clean_name}")
            
            # Search Wikipedia
            summary = wikipedia.summary(clean_name, sentences=10, auto_suggest=True)
            
            if summary and len(summary) > 50:
                print(f"   ✓ Wikipedia article found ({len(summary)} chars)")
                return {
                    "source": "Wikipedia",
                    "context": summary,
                    "animal_name": clean_name,
                    "num_documents": 1
                }
            else:
                print("   ⚠️  Wikipedia returned insufficient content")
                return {
                    "source": "none",
                    "context": f"No detailed information found for {clean_name}.",
                    "animal_name": clean_name,
                    "num_documents": 0
                }
                
        except Exception as e:
            print(f"   ❌ Wikipedia error: {e}")
            return {
                "source": "none",
                "context": f"Unable to retrieve information about {animal_name}.",
                "animal_name": animal_name,
                "error": str(e),
                "num_documents": 0
            }
    
    def add_document(self, text: str, metadata: Dict = None) -> bool:
        """
        Add a document to the vector database
        
        Args:
            text: Document text
            metadata: Optional metadata dict
            
        Returns:
            Success status
        """
        if self.vectordb is None:
            raise Exception("Vector database not initialized")
        
        try:
            from langchain.schema import Document
            
            if metadata is None:
                metadata = {}
            
            doc = Document(page_content=text, metadata=metadata)
            self.vectordb.add_documents([doc])
            
            print(f"✓ Document added to vector database")
            return True
            
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            "vectordb_available": self.vectordb is not None,
            "chroma_path": self.chroma_dir,
            "embeddings_loaded": self.embeddings is not None,
            "retriever_available": self.retriever is not None,
            "wikipedia_fallback": True
        }
        
        if self.vectordb:
            try:
                # Try to get collection stats
                collection = self.vectordb._collection
                stats["document_count"] = collection.count()
            except:
                stats["document_count"] = "unknown"
        
        return stats

if __name__ == "__main__":
    print("🚀 Starting RAG System Initialization...\n")
    rag = RAGSystem()
    rag.initialize()

    print("\n📊 RAG Statistics:")
    print(rag.get_stats())

    print("\n✅ DONE — RAG system tested.\n")
