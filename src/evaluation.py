import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re
from collections import defaultdict
import os
from radgraph import RadGraph, F1RadGraph

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class ReportEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        # Initialize RadGraph
        try:
            self.radgraph = RadGraph()
            self.f1_radgraph = F1RadGraph(reward_level="all")
            self.radgraph_available = True
            print("RadGraph & F1RadGraph initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize RadGraph/F1RadGraph: {e}")
            self.radgraph_available = False
        
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
        
        # RadGraph evaluation if available
        if self.radgraph_available:
            results.update(self.evaluate_radgraph(predictions, references))
        
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

    def evaluate_radgraph(self, predictions, references):
        """Evaluate using RadGraph if available"""
        mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = \
            self.f1_radgraph(hyps=predictions, refs=references)
            
        # Unpack the three axes of reward
        simple_list, partial_list, complete_list = reward_list
        mean_simple, mean_partial, mean_complete = mean_reward

        rad_results = {}

        # Base F1 stats
        rad_results.update({
            'radgraph_simple_mean':   float(mean_simple),
            'radgraph_simple_std':    float(np.std(simple_list)),
            'radgraph_partial_mean':  float(mean_partial),
            'radgraph_partial_std':   float(np.std(partial_list)),
            'radgraph_complete_mean': float(mean_complete),
            'radgraph_complete_std':  float(np.std(complete_list)),
        })

        # --- extra: use the annotation lists for entity/relation counts ---
        ent_precisions, ent_recalls = [], []
        hyp_counts, ref_counts = [], []

        for hyp_ann, ref_ann in zip(hypothesis_annotation_lists, reference_annotation_lists):
            # Each ann list is a list of dicts; pull out triples
            hyp_triples = {(e[0], e[1], e[2]) for e in hyp_ann}
            ref_triples = {(e[0], e[1], e[2]) for e in ref_ann}

            tp = len(hyp_triples & ref_triples)
            p = tp / len(hyp_triples) if hyp_triples else 0.0
            r = tp / len(ref_triples) if ref_triples else 0.0

            ent_precisions.append(p)
            ent_recalls.append(r)
            hyp_counts.append(len(hyp_triples))
            ref_counts.append(len(ref_triples))

        # summarize
        rad_results.update({
            'radgraph_entity_precision_mean': np.mean(ent_precisions),
            'radgraph_entity_precision_std':  np.std(ent_precisions),
            'radgraph_entity_recall_mean':    np.mean(ent_recalls),
            'radgraph_entity_recall_std':     np.std(ent_recalls),
            'radgraph_avg_hyp_entities':      np.mean(hyp_counts),
            'radgraph_avg_ref_entities':      np.mean(ref_counts),
        })

        return rad_results
    
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
        if csv_path is None:
            csv_path = "../results/inference_results.csv"
        try:
            results_df = pd.read_csv(csv_path)
            print(f"Loaded inference results from: {csv_path}")
        except FileNotFoundError:
            print(f"No inference results file found at {csv_path}. Please run inference first.")
            return None
    
    # Extract data
    predictions = results_df['predicted_report'].tolist()
    references  = results_df['ground_truth_report'].tolist()
    
    # Handle CheXpert labels if available
    chexpert_pred = None
    chexpert_true = None
    if 'true_chexpert' in results_df.columns:
        chexpert_true = []
        for labels in results_df['true_chexpert']:
            if isinstance(labels, str):
                try:
                    chexpert_true.append(eval(labels))
                except:
                    chexpert_true.append(labels)
            else:
                chexpert_true.append(labels)
    
    evaluator = ReportEvaluator()
    print("Running comprehensive evaluation...")
    results = evaluator.evaluate_reports(predictions, references, chexpert_pred, chexpert_true)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nDataset Info:")
    print(f"  Number of samples:            {len(predictions)}")
    print(f"  Avg. prediction length:       {np.mean([len(p) for p in predictions]):.1f} chars")
    print(f"  Avg. reference length:        {np.mean([len(r) for r in references]):.1f} chars")
    
    print(f"\nText-based Metrics:")
    print(f"  BLEU Score:                   {results['bleu_mean']:.4f} ± {results['bleu_std']:.4f}")
    print(f"  ROUGE-1 F1:                   {results['rouge1_f1_mean']:.4f} ± {results['rouge1_f1_std']:.4f}")
    print(f"  ROUGE-2 F1:                   {results['rouge2_f1_mean']:.4f} ± {results['rouge2_f1_std']:.4f}")
    print(f"  ROUGE-L F1:                   {results['rougeL_f1_mean']:.4f} ± {results['rougeL_f1_std']:.4f}")
    
    if 'chexpert_exact_match' in results:
        print(f"\nCheXpert Label Metrics:")
        print(f"  Exact Match Accuracy:         {results['chexpert_exact_match']:.4f}")
        chexpert_classes = [
            'no_finding', 'enlarged_cardiomediastinum', 'cardiomegaly',
            'lung_opacity', 'lung_lesion', 'edema', 'consolidation',
            'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion',
            'pleural_other', 'fracture', 'support_devices'
        ]
        for cls in chexpert_classes:
            key = f'chexpert_{cls}_f1'
            if key in results:
                print(f"  {cls.replace('_', ' ').title():<30} {results[key]:.4f}")
    
    # RadGraph F1 axes
    if 'radgraph_simple_mean' in results:
        print(f"\nRadGraph F1 Rewards (reward_level='all'):")
        print(f"  Simple   F1:                  {results['radgraph_simple_mean']:.4f} ± {results['radgraph_simple_std']:.4f}")
        print(f"  Partial  F1:                  {results['radgraph_partial_mean']:.4f} ± {results['radgraph_partial_std']:.4f}")
        print(f"  Complete F1:                  {results['radgraph_complete_mean']:.4f} ± {results['radgraph_complete_std']:.4f}")
        
        print(f"\nRadGraph Entity-level Stats:")
        print(f"  Entity Precision:             {results['radgraph_entity_precision_mean']:.4f} ± {results['radgraph_entity_precision_std']:.4f}")
        print(f"  Entity Recall:                {results['radgraph_entity_recall_mean']:.4f} ± {results['radgraph_entity_recall_std']:.4f}")
        print(f"  Avg. # Predicted Entities:    {results['radgraph_avg_hyp_entities']:.1f}")
        print(f"  Avg. # Reference Entities:    {results['radgraph_avg_ref_entities']:.1f}")
    
    # Save detailed results
    os.makedirs("../results", exist_ok=True)
    pd.DataFrame([results]).to_csv("../results/evaluation_metrics.csv", index=False)
    print(f"\nDetailed results saved to: ../results/evaluation_metrics.csv")
    
    return results


if __name__ == "__main__":
    run_comprehensive_evaluation(csv_path="../results/inference_results.csv")
