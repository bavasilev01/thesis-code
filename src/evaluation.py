import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoProcessor
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re
from collections import defaultdict
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

class ReportEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        
    def evaluate_reports(self, predictions, references, chexpert_pred=None, chexpert_true=None):
        """
        Comprehensive evaluation of medical reports
        
        Args:
            predictions: List of predicted report texts
            references: List of reference/ground truth report texts
            chexpert_pred: Predicted CheXpert labels (optional)
            chexpert_true: True CheXpert labels (optional)
        """
        results = {}
        
        # Text-based metrics
        results.update(self._evaluate_text_metrics(predictions, references))
        
        # Medical concept extraction and evaluation
        results.update(self._evaluate_medical_concepts(predictions, references))
        
        # CheXpert label evaluation if provided
        if chexpert_pred is not None and chexpert_true is not None:
            results.update(self._evaluate_chexpert_labels(chexpert_pred, chexpert_true))
            
        return results
    
    def _evaluate_text_metrics(self, predictions, references):
        """Evaluate text-based metrics like BLEU and ROUGE"""
        bleu_scores = []
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            # Tokenize for BLEU
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            
            # BLEU score
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothie)
            bleu_scores.append(bleu)
            
            # ROUGE scores
            rouge_result = self.rouge_scorer.score(ref, pred)
            for metric, score in rouge_result.items():
                rouge_scores[f"{metric}_f1"].append(score.fmeasure)
                rouge_scores[f"{metric}_precision"].append(score.precision)
                rouge_scores[f"{metric}_recall"].append(score.recall)
        
        results = {
            'bleu_mean': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores)
        }
        
        for metric, scores in rouge_scores.items():
            results[f"{metric}_mean"] = np.mean(scores)
            results[f"{metric}_std"] = np.std(scores)
            
        return results
    
    def _evaluate_medical_concepts(self, predictions, references):
        """Evaluate medical concept extraction accuracy"""
        # Define medical terms to look for
        medical_terms = [
            'pneumonia', 'atelectasis', 'consolidation', 'edema', 'effusion',
            'cardiomegaly', 'pneumothorax', 'nodule', 'mass', 'opacity',
            'infiltrate', 'normal', 'clear', 'lungs', 'heart', 'mediastinum'
        ]
        
        concept_precision = []
        concept_recall = []
        concept_f1 = []
        
        for pred, ref in zip(predictions, references):
            pred_concepts = self._extract_concepts(pred.lower(), medical_terms)
            ref_concepts = self._extract_concepts(ref.lower(), medical_terms)
            
            if len(ref_concepts) == 0:
                continue
                
            tp = len(pred_concepts & ref_concepts)
            fp = len(pred_concepts - ref_concepts)
            fn = len(ref_concepts - pred_concepts)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            concept_precision.append(precision)
            concept_recall.append(recall)
            concept_f1.append(f1)
        
        return {
            'concept_precision_mean': np.mean(concept_precision),
            'concept_recall_mean': np.mean(concept_recall),
            'concept_f1_mean': np.mean(concept_f1),
            'concept_precision_std': np.std(concept_precision),
            'concept_recall_std': np.std(concept_recall),
            'concept_f1_std': np.std(concept_f1)
        }
    
    def _extract_concepts(self, text, terms):
        """Extract medical concepts from text"""
        found_terms = set()
        for term in terms:
            if term in text:
                found_terms.add(term)
        return found_terms
    
    def _evaluate_chexpert_labels(self, pred_labels, true_labels):
        """Evaluate CheXpert label predictions"""
        # CheXpert has 14 labels
        chexpert_classes = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        results = {}
        
        # Convert to numpy arrays
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        
        # Overall accuracy (exact match)
        exact_match = np.all(pred_labels == true_labels, axis=1)
        results['chexpert_exact_match'] = np.mean(exact_match)
        
        # Per-class metrics
        for i, class_name in enumerate(chexpert_classes):
            if i < pred_labels.shape[1]:
                # Convert to binary (positive vs negative/uncertain)
                pred_binary = (pred_labels[:, i] == 1).astype(int)
                true_binary = (true_labels[:, i] == 1).astype(int)
                
                if len(np.unique(true_binary)) > 1:  # Only if both classes present
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        true_binary, pred_binary, average='binary', zero_division=0
                    )
                    
                    results[f'chexpert_{class_name.lower().replace(" ", "_")}_precision'] = precision
                    results[f'chexpert_{class_name.lower().replace(" ", "_")}_recall'] = recall
                    results[f'chexpert_{class_name.lower().replace(" ", "_")}_f1'] = f1
        
        return results

def run_comprehensive_evaluation(results_df=None, csv_path=None):
    """Run comprehensive evaluation on inference results"""
    
    # Load inference results
    if results_df is not None:
        print("Using provided DataFrame for evaluation")
    else:
        # Try to load from provided path or default location
        if csv_path is None:
            csv_path = "../results/inference_results.csv"
        
        try:
            results_df = pd.read_csv(csv_path)
            print(f"Loaded inference results from: {csv_path}")
        except FileNotFoundError:
            print(f"No inference results file found at {csv_path}. Please run inference first.")
            return None
    
    # Extract data for evaluation
    predictions = results_df['predicted_report'].tolist()
    references = results_df['ground_truth_report'].tolist()
    
    # Handle CheXpert labels if available
    chexpert_pred = None
    chexpert_true = None
    
    if 'true_chexpert' in results_df.columns:
        # Convert string representation back to lists if needed
        chexpert_true = []
        for labels in results_df['true_chexpert']:
            if isinstance(labels, str):
                try:
                    # Handle string representation of lists
                    chexpert_true.append(eval(labels))
                except:
                    # If eval fails, assume it's already a proper format
                    chexpert_true.append(labels)
            else:
                chexpert_true.append(labels)
    
    # Initialize evaluator
    evaluator = ReportEvaluator()
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.evaluate_reports(
        predictions, references, chexpert_pred, chexpert_true
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nDataset Info:")
    print(f"Number of samples: {len(predictions)}")
    print(f"Average prediction length: {np.mean([len(p) for p in predictions]):.1f} characters")
    print(f"Average reference length: {np.mean([len(r) for r in references]):.1f} characters")
    
    print(f"\nText-based Metrics:")
    print(f"BLEU Score: {results['bleu_mean']:.4f} ± {results['bleu_std']:.4f}")
    print(f"ROUGE-1 F1: {results['rouge1_f1_mean']:.4f} ± {results['rouge1_f1_std']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge2_f1_mean']:.4f} ± {results['rouge2_f1_std']:.4f}")
    print(f"ROUGE-L F1: {results['rougeL_f1_mean']:.4f} ± {results['rougeL_f1_std']:.4f}")
    
    print(f"\nMedical Concept Metrics:")
    print(f"Concept Precision: {results['concept_precision_mean']:.4f} ± {results['concept_precision_std']:.4f}")
    print(f"Concept Recall: {results['concept_recall_mean']:.4f} ± {results['concept_recall_std']:.4f}")
    print(f"Concept F1: {results['concept_f1_mean']:.4f} ± {results['concept_f1_std']:.4f}")
    
    if 'chexpert_exact_match' in results:
        print(f"\nCheXpert Label Metrics:")
        print(f"Exact Match Accuracy: {results['chexpert_exact_match']:.4f}")
        
        # Print per-class results
        chexpert_classes = [
            'no_finding', 'enlarged_cardiomediastinum', 'cardiomegaly',
            'lung_opacity', 'lung_lesion', 'edema', 'consolidation',
            'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion',
            'pleural_other', 'fracture', 'support_devices'
        ]
        
        for class_name in chexpert_classes:
            f1_key = f'chexpert_{class_name}_f1'
            if f1_key in results:
                print(f"{class_name.replace('_', ' ').title()} F1: {results[f1_key]:.4f}")
    
    # Save detailed results
    os.makedirs("../results", exist_ok=True)
    eval_results_df = pd.DataFrame([results])
    eval_results_df.to_csv("../results/evaluation_metrics.csv", index=False)
    print(f"\nDetailed results saved to: ../results/evaluation_metrics.csv")
    
    return results

if __name__ == "__main__":
    run_comprehensive_evaluation()
