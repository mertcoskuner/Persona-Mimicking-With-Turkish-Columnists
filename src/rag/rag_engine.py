"""
RAG (Retrieval Augmented Generation) engine implementation.
This module handles the evaluation and comparison of different language models.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import openai
from evaluate import load
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
OUTPUT_DIR = Path("results/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class RAGEvaluator:
    """Evaluates and compares different language models using RAG metrics."""
    
    def __init__(self, openai_api_key: str):
        """Initialize the evaluator with required models and configurations."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        self.bert_score_metric = load("bertscore")
        self.semantic_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        openai.api_key = openai_api_key

    def load_data(self, predicted_file: str, reference_file: str) -> Tuple[List[Dict], List[Dict]]:
        """Load predicted and reference data from files."""
        with open(predicted_file, "r", encoding="utf-8") as f:
            predicted_data = [json.loads(line) for line in f]
        with open(reference_file, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        return predicted_data, reference_data

    def get_base_model_response(self, instruction: str) -> str:
        """Generate response from the base model."""
        input_ids = self.tokenizer.encode(instruction, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            input_ids,
            max_length=150,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def calculate_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Tuple[List[float], List[float]]:
        """Calculate BERT F1 and semantic similarity scores."""
        # BERT F1
        bert_results = self.bert_score_metric.compute(
            predictions=predictions,
            references=references,
            lang="tr",
            model_type="dbmdz/bert-base-turkish-cased"
        )
        bert_f1_scores = bert_results["f1"]

        # Semantic similarity
        semantic_similarities = []
        for pred, ref in zip(predictions, references):
            pred_embedding = self.semantic_model.encode(pred, convert_to_tensor=True)
            ref_embedding = self.semantic_model.encode(ref, convert_to_tensor=True)
            similarity = util.cos_sim(pred_embedding, ref_embedding).item()
            semantic_similarities.append(similarity)

        return bert_f1_scores, semantic_similarities

    def evaluate_with_llm(
        self,
        question: str,
        reference_answer: str,
        predicted_answer: str
    ) -> Optional[float]:
        """Evaluate responses using LLM and return a single score."""
        messages = [
            {"role": "system", "content": "Bir değerlendirme yargıcı olarak cevapları analiz et."},
            {"role": "user", "content": f"""
            Aşağıdaki iki cevabı karşılaştır ve bir skor ver:

            Soru: {question}
            Referans Yanıt: {reference_answer}
            Modelin Tahmini Yanıtı: {predicted_answer}

            Değerlendirme Kriterleri:
            - Doğruluk, bağlama uygunluk ve dil-stil uyumu gibi faktörleri göz önünde bulundur.
            - Ancak yalnızca 0 ile 1 arasında bir sayı döndür. Açıklama yapma. Sadece sayı yaz.
            """}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            return float(response['choices'][0]['message']['content'].strip())
        except openai.error.OpenAIError as e:
            print(f"OpenAI API hatası: {e}")
            return None

    def plot_metrics(
        self,
        metric_name: str,
        predicted_values: List[float],
        base_values: List[float]
    ) -> None:
        """Plot and save metric distributions."""
        plt.figure(figsize=(10, 6))
        sns.histplot(predicted_values, bins=20, kde=True, color="blue", label="Predicted", stat="count")
        sns.histplot(base_values, bins=20, kde=True, color="orange", label="Base Model", stat="count")
        plt.title(f"{metric_name} Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(OUTPUT_DIR / f"{metric_name.lower().replace(' ', '_')}_distribution.png")
        plt.close()

    def evaluate_models(
        self,
        predicted_file: str,
        reference_file: str,
        output_scores_file: str
    ) -> None:
        """Main evaluation pipeline."""
        # Load data
        predicted_data, reference_data = self.load_data(predicted_file, reference_file)
        
        # Extract answers and instructions
        predicted_answers = [item["answer"] for item in predicted_data[:100]]
        reference_answers = [item["answer"] for item in reference_data[:100]]
        instructions = [item["instruction"] for item in predicted_data[:100]]

        # Generate base model answers
        base_model_answers = [self.get_base_model_response(instr) for instr in instructions]

        # Calculate metrics
        predicted_bert_f1, predicted_semantic_similarity = self.calculate_metrics(
            predicted_answers, reference_answers
        )
        base_model_bert_f1, base_model_semantic_similarity = self.calculate_metrics(
            base_model_answers, reference_answers
        )

        # Get LLM scores
        predicted_llm_scores = [
            self.evaluate_with_llm(instr, ref, pred)
            for instr, ref, pred in zip(instructions, reference_answers, predicted_answers)
        ]
        base_model_llm_scores = [
            self.evaluate_with_llm(instr, ref, base)
            for instr, ref, base in zip(instructions, reference_answers, base_model_answers)
        ]

        # Prepare score data
        score_data = []
        for i, instruction in enumerate(instructions):
            score_data.append({
                "instruction": instruction,
                "predicted_answer": predicted_answers[i],
                "base_model_answer": base_model_answers[i],
                "reference_answer": reference_answers[i],
                "predicted_bert_f1": predicted_bert_f1[i],
                "base_model_bert_f1": base_model_bert_f1[i],
                "predicted_semantic_similarity": predicted_semantic_similarity[i],
                "base_model_semantic_similarity": base_model_semantic_similarity[i],
                "predicted_llm_score": predicted_llm_scores[i],
                "base_model_llm_score": base_model_llm_scores[i],
            })

        # Save scores
        with open(output_scores_file, "w", encoding="utf-8") as f:
            json.dump(score_data, f, ensure_ascii=False, indent=4)

        # Plot metrics
        metrics = [
            ("BERT F1", predicted_bert_f1, base_model_bert_f1),
            ("Semantic Similarity", predicted_semantic_similarity, base_model_semantic_similarity),
            ("LLM Score", predicted_llm_scores, base_model_llm_scores),
        ]

        for metric_name, predicted_values, base_values in metrics:
            self.plot_metrics(metric_name, predicted_values, base_values)

def main():
    """Main execution function."""
    evaluator = RAGEvaluator(openai_api_key="your-api-key")
    evaluator.evaluate_models(
        predicted_file="data/predictions.jsonl",
        reference_file="data/references.json",
        output_scores_file="results/scores.json"
    )

if __name__ == "__main__":
    main()
