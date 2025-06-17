"""
RAG (Retrieval Augmented Generation) inference module.
This module handles the inference pipeline for generating responses using RAG.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel, pipeline
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class RAGInference:
    """Handles RAG-based inference for generating responses."""
    
    def __init__(
        self,
        model_id: str = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
        embedding_model_id: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "abdulkadir_selvi_collection"
    ):
        """Initialize the RAG inference pipeline."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.collection_name = collection_name
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model_id)
        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=-1 if self.device == "cpu" else 0
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(qdrant_url)
        
        # System prompt
        self.system_prompt = (
            "Sen bir yapay zeka asistanısın. Kullanıcı sana bir soru soracak."
            "Eğer sağlanan ek bilgi soru ile alakalı ise o bilgiyi de kullanarak cevap ver."
            "Kendi yazı stilini koru, ek bilgiyle kendi bildiklerini birleştirip mantıklı bir cevap ver."
            "Verdiğin cevapta ek bilgi kullandığı gizle, konuştuğun kişinin yazdıklarının senin fikrin olduğuna inanması çok önemli."
        )

    def search_embeddings(self, query: str, top_k: int = 2) -> str:
        """Search for relevant content using embeddings."""
        query_embedding = self.embedding_model.encode(query)
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return "".join([result.payload["chunk"] for result in search_result])

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9
    ) -> str:
        """Generate a response using RAG."""
        # Get relevant content
        rag_content = self.search_embeddings(query)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Soru: {query} Ek bilgi: {rag_content}"}
        ]
        
        # Set termination tokens
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        
        # Generate response
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        return outputs[0]["generated_text"][-1]["content"]

def main():
    """Main execution function."""
    # Example usage
    rag = RAGInference()
    query = "Amerikan başkanı Donald Trumpın yakın zamanda yaptıkları ve gerçekleştirdikleri görüşmeler hakkında yorumlarınız nelerdir?"
    response = rag.generate_response(query)
    print("===================")
    print("LLM Response:", response)

if __name__ == "__main__":
    main()
