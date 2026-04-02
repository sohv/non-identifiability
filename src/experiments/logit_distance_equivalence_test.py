"""
Logit-Level Distance Equivalence Test

Tests whether v ≈ v + v_perp holds by comparing logit distances,
rather than relying on KL divergence which has numerical stability issues.

Usage:
    # All traits (formality, politeness, sentiment, truthfulness) for both models
    python src/experiments/logit_distance_equivalence_test.py --traits all
    
    # Single model with all traits
    python src/experiments/logit_distance_equivalence_test.py --models Qwen/Qwen2.5-3B-Instruct --traits all
    
    # Specific traits only
    python src/experiments/logit_distance_equivalence_test.py --traits formality politeness
    
    # Custom output directory
    python src/experiments/logit_distance_equivalence_test.py --traits all --output results/logit_new
    
    # Both models, specific traits
    python src/experiments/logit_distance_equivalence_test.py \
        --models Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.1-8B-Instruct \
        --traits sentiment formality
"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import json
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Handle both relative and absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from .persona_vector_experiment import PersonaVectorExperiment
except ImportError:
    try:
        from persona_vector_experiment import PersonaVectorExperiment
    except ImportError:
        from src.experiments.persona_vector_experiment import PersonaVectorExperiment


class LogitDistanceEquivalenceTest:
    """
    Test equivalence of steering vectors using direct logit distance.
    
    Core idea: If v ≈ v + v_perp, then logits should be nearly identical.
    We measure this as L2 distance between logit vectors.
    """
    
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.experiment = PersonaVectorExperiment(model_name, device)
    
    def get_logits_with_steering(self, prompt: str, steering_vector: torch.Tensor,
                                 alpha: float = 1.0, layer: int = None) -> torch.Tensor:
        """
        Get logits at the next token position with steering applied.
        
        Args:
            prompt: Input prompt
            steering_vector: Steering vector to apply
            alpha: Steering strength
            layer: Layer to apply steering (default: middle layer)
            
        Returns:
            Logits tensor of shape (vocab_size,)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        if layer is None:
            layer_idx = len(self.model.model.layers) // 2
        else:
            layer_idx = layer
        
        def steering_hook(module, input, output):
            """Hook that modifies hidden states by adding steering vector."""
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()
            
            # Add steering to the last token
            steering_gpu = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] += alpha * steering_gpu
            
            if rest:
                return (hidden_states,) + rest
            else:
                return hidden_states
        
        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(steering_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits[0, -1, :]  # Get logits at last token position
        finally:
            hook_handle.remove()
        
        return logits.cpu()
    
    def logit_distance(self, logits1: torch.Tensor, logits2: torch.Tensor) -> float:
        """
        Compute L2 distance between logit vectors.
        
        Args:
            logits1: First logit vector
            logits2: Second logit vector
            
        Returns:
            L2 distance
        """
        return float(torch.norm(logits1 - logits2, p=2).item())
    
    def token_agreement(self, logits1: torch.Tensor, logits2: torch.Tensor) -> float:
        """
        Compute fraction of agreement on argmax token.
        
        Args:
            logits1: First logit vector
            logits2: Second logit vector
            
        Returns:
            1.0 if same top-1 token, 0.0 otherwise
        """
        token1 = torch.argmax(logits1)
        token2 = torch.argmax(logits2)
        return float((token1 == token2).item())
    
    def top_k_overlap(self, logits1: torch.Tensor, logits2: torch.Tensor, k: int = 10) -> float:
        """
        Compute fraction of overlap in top-k tokens.
        
        Args:
            logits1: First logit vector
            logits2: Second logit vector
            k: Number of top tokens to consider
            
        Returns:
            Fraction of overlap ([0, 1])
        """
        top_k_1 = set(torch.topk(logits1, k).indices.tolist())
        top_k_2 = set(torch.topk(logits2, k).indices.tolist())
        overlap = len(top_k_1 & top_k_2) / k
        return overlap
    
    def test_equivalence(self, trait: str, test_prompts: List[str], 
                        n_orthogonal_samples: int = 5,
                        alpha: float = 1.0, layer: int = None) -> Dict:
        """
        Test whether v ≈ v + v_perp using logit distance.
        
        Args:
            trait: Semantic trait
            test_prompts: Prompts to test on
            n_orthogonal_samples: Number of orthogonal vectors to sample
            alpha: Steering strength
            layer: Layer to apply steering
            
        Returns:
            Comprehensive results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Logit Distance Equivalence Test: {trait}")
        print(f"{'='*80}\n")
        
        # Extract steering vector
        print(f"Extracting {trait} steering vector...")
        steering_vector = self.experiment.extract_steering_vector(trait, n_pairs=50, layer=layer)
        v_norm = torch.norm(steering_vector).item()
        print(f"Steering vector norm: {v_norm:.4f}\n")
        
        # 1. Baseline: v vs no steering
        print(f"Baseline: distance between v and no steering...")
        baseline_distances = []
        for prompt in tqdm(test_prompts, desc="Baseline", leave=False):
            logits_baseline = self.get_logits_with_steering(prompt, torch.zeros_like(steering_vector),
                                                           alpha=0, layer=layer)
            logits_v = self.get_logits_with_steering(prompt, steering_vector, alpha=alpha, layer=layer)
            
            dist = self.logit_distance(logits_v, logits_baseline)
            baseline_distances.append(dist)
        
        mean_baseline = np.mean(baseline_distances)
        std_baseline = np.std(baseline_distances)
        print(f"Distance(v || no steering): μ={mean_baseline:.4f}, σ={std_baseline:.4f}\n")
        
        # 2. Main test: v vs v + v_perp
        print(f"Main test: distance(v || v + v_perp) for {n_orthogonal_samples} orthogonal vectors...\n")
        vperp_distances_all = []
        vperp_agreement_all = []
        vperp_topk_all = []
        
        for sample_idx in range(n_orthogonal_samples):
            print(f"Orthogonal sample {sample_idx + 1}/{n_orthogonal_samples}")
            
            # Create random orthogonal vector
            random_vec = torch.randn_like(steering_vector)
            v_perp = random_vec - (random_vec @ steering_vector) / (steering_vector @ steering_vector) * steering_vector
            v_perp = v_perp / torch.norm(v_perp) * v_norm
            
            v_plus_perp = steering_vector + v_perp
            
            print(f"  ||v_perp|| = {torch.norm(v_perp).item():.4f}")
            print(f"  v · v_perp = {(steering_vector @ v_perp).item():.8f}")
            print(f"  ||v + v_perp|| = {torch.norm(v_plus_perp).item():.4f}")
            
            distances = []
            agreements = []
            topk_overlaps = []
            
            for prompt in tqdm(test_prompts, desc=f"  Sample {sample_idx+1}", leave=False):
                logits_v = self.get_logits_with_steering(prompt, steering_vector, 
                                                        alpha=alpha, layer=layer)
                logits_vperp = self.get_logits_with_steering(prompt, v_plus_perp, 
                                                            alpha=alpha, layer=layer)
                
                dist = self.logit_distance(logits_v, logits_vperp)
                agreement = self.token_agreement(logits_v, logits_vperp)
                topk = self.top_k_overlap(logits_v, logits_vperp, k=10)
                
                distances.append(dist)
                agreements.append(agreement)
                topk_overlaps.append(topk)
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            mean_agreement = np.mean(agreements)
            mean_topk = np.mean(topk_overlaps)
            
            vperp_distances_all.extend(distances)
            vperp_agreement_all.extend(agreements)
            vperp_topk_all.extend(topk_overlaps)
            
            print(f"  Distance(v || v+v_perp): μ={mean_dist:.4f}, σ={std_dist:.4f}")
            print(f"  Token agreement: {mean_agreement*100:.1f}%")
            print(f"  Top-10 overlap: {mean_topk*100:.1f}%\n")
        
        # 3. Comparison with random directions
        print(f"Comparison: distance(v || random) for 3 random vectors...\n")
        random_distances = []
        random_agreement = []
        random_topk = []
        
        for i in range(3):
            print(f"Random vector {i+1}/3")
            random_vec = torch.randn_like(steering_vector)
            random_vec = random_vec / torch.norm(random_vec) * v_norm
            
            distances = []
            agreements = []
            topks = []
            
            for prompt in tqdm(test_prompts, desc=f"  Random {i+1}", leave=False):
                logits_v = self.get_logits_with_steering(prompt, steering_vector, 
                                                        alpha=alpha, layer=layer)
                logits_random = self.get_logits_with_steering(prompt, random_vec, 
                                                             alpha=alpha, layer=layer)
                
                dist = self.logit_distance(logits_v, logits_random)
                agreement = self.token_agreement(logits_v, logits_random)
                topk = self.top_k_overlap(logits_v, logits_random, k=10)
                
                distances.append(dist)
                agreements.append(agreement)
                topks.append(topk)
            
            mean_dist = np.mean(distances)
            mean_agreement = np.mean(agreements)
            mean_topk = np.mean(topks)
            
            random_distances.extend(distances)
            random_agreement.extend(agreements)
            random_topk.extend(topks)
            
            print(f"  Distance(v || random): μ={mean_dist:.4f}")
            print(f"  Token agreement: {mean_agreement*100:.1f}%")
            print(f"  Top-10 overlap: {mean_topk*100:.1f}%\n")
        
        # 4. Comparison with different trait
        print(f"Comparison: distance(v_{trait} || v_different_trait)...\n")
        different_trait = "formality" if trait != "formality" else "sentiment"
        
        steering_vector_diff = self.experiment.extract_steering_vector(different_trait, n_pairs=50, layer=layer)
        steering_vector_diff = steering_vector_diff / torch.norm(steering_vector_diff) * v_norm
        
        diff_trait_distances = []
        diff_trait_agreement = []
        diff_trait_topk = []
        
        for prompt in tqdm(test_prompts, desc="Different trait", leave=False):
            logits_v = self.get_logits_with_steering(prompt, steering_vector, 
                                                    alpha=alpha, layer=layer)
            logits_diff = self.get_logits_with_steering(prompt, steering_vector_diff, 
                                                       alpha=alpha, layer=layer)
            
            dist = self.logit_distance(logits_v, logits_diff)
            agreement = self.token_agreement(logits_v, logits_diff)
            topk = self.top_k_overlap(logits_v, logits_diff, k=10)
            
            diff_trait_distances.append(dist)
            diff_trait_agreement.append(agreement)
            diff_trait_topk.append(topk)
        
        mean_diff_trait = np.mean(diff_trait_distances)
        mean_diff_agreement = np.mean(diff_trait_agreement)
        mean_diff_topk = np.mean(diff_trait_topk)
        print(f"Distance(v_{trait} || v_{different_trait}): μ={mean_diff_trait:.4f}")
        print(f"Token agreement: {mean_diff_agreement*100:.1f}%")
        print(f"Top-10 overlap: {mean_diff_topk*100:.1f}%\n")
        
        # Summary
        print(f"{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}\n")
        
        mean_vperp_dist = np.mean(vperp_distances_all)
        std_vperp_dist = np.std(vperp_distances_all)
        mean_random_dist = np.mean(random_distances)
        std_random_dist = np.std(random_distances)
        
        print(f"Distance(v || baseline):          μ={mean_baseline:.4f}, σ={std_baseline:.4f}")
        print(f"Distance(v || v_perp):           μ={mean_vperp_dist:.4f}, σ={std_vperp_dist:.4f}")
        print(f"Distance(v || random):           μ={mean_random_dist:.4f}, σ={std_random_dist:.4f}")
        print(f"Distance(v || different_trait):  μ={mean_diff_trait:.4f}\n")
        
        # Key ratios
        ratio_random = mean_vperp_dist / (mean_random_dist + 1e-10)
        ratio_trait = mean_vperp_dist / (mean_diff_trait + 1e-10)
        
        print(f"Distance(v_perp) / Distance(random):       {ratio_random:.3f}")
        print(f"Distance(v_perp) / Distance(different):    {ratio_trait:.3f}\n")
        
        print(f"Token agreement with v_perp:  {np.mean(vperp_agreement_all)*100:.1f}%")
        print(f"Token agreement with random:  {np.mean(random_agreement)*100:.1f}%\n")
        
        results = {
            'trait': trait,
            'model': self.model_name,
            'distances': {
                'baseline': {
                    'mean': float(mean_baseline),
                    'std': float(std_baseline)
                },
                'v_vs_vperp': {
                    'mean': float(mean_vperp_dist),
                    'std': float(std_vperp_dist),
                    'samples': [float(x) for x in vperp_distances_all]
                },
                'v_vs_random': {
                    'mean': float(mean_random_dist),
                    'std': float(std_random_dist),
                    'samples': [float(x) for x in random_distances]
                },
                'v_vs_different_trait': {
                    'mean': float(mean_diff_trait),
                    'trait_compared': different_trait,
                    'samples': [float(x) for x in diff_trait_distances]
                }
            },
            'token_agreement': {
                'v_vs_vperp': float(np.mean(vperp_agreement_all)),
                'v_vs_random': float(np.mean(random_agreement)),
                'v_vs_different_trait': float(mean_diff_agreement)
            },
            'top_k_overlap': {
                'v_vs_vperp': float(np.mean(vperp_topk_all)),
                'v_vs_random': float(np.mean(random_topk)),
                'v_vs_different_trait': float(mean_diff_topk)
            },
            'ratios': {
                'distance_vperp_over_random': float(ratio_random),
                'distance_vperp_over_different_trait': float(ratio_trait)
            },
            'parameters': {
                'n_prompts': len(test_prompts),
                'n_orthogonal_samples': n_orthogonal_samples,
                'alpha': alpha,
                'steering_vector_norm': float(v_norm)
            }
        }
        
        return results


def run_logit_distance_test(model_names: List[str] = None, traits: List[str] = None, 
                           output_dir: str = "../results"):
    """
    Run the logit distance equivalence test.
    
    Args:
        model_names: List of model names to test
        traits: List of traits to test
        output_dir: Directory to save results
    """
    if model_names is None:
        model_names = [
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct"
        ]
    
    if traits is None:
        traits = ["formality", "politeness", "sentiment", "truthfulness", "agreeableness"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test prompts
    test_prompts = [
        "Tell me about your day:",
        "What do you think about",
        "Can you explain",
        "Let me share my thoughts on",
        "I want to discuss",
        "Have you considered",
        "In my opinion,",
        "From my perspective,",
        "Let me tell you about",
        "I recently learned",
        "The interesting thing is",
        "What surprises me is",
        "I find it fascinating",
        "It's worth noting",
        "Consider this:",
        "This makes me think",
        "Interestingly,",
        "What I've observed",
        "The thing is,",
        "You know what",
    ]
    
    # Create traits filename suffix
    traits_suffix = "_".join(traits)
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}\n")
        
        test = LogitDistanceEquivalenceTest(model_name)
        
        for trait in traits:
            try:
                results = test.test_equivalence(trait, test_prompts, 
                                               n_orthogonal_samples=5,
                                               alpha=1.0)
                
                model_key = model_name.split("/")[-1]
                if model_key not in all_results:
                    all_results[model_key] = {}
                all_results[model_key][trait] = results
                
            except Exception as e:
                print(f"Error testing {trait} on {model_name}: {e}\n")
                import traceback
                traceback.print_exc()
    
    # Save combined results with model name and traits in filename
    model_part = model_names[0].split("/")[-1] if len(model_names) == 1 else "all_models"
    combined_file = os.path.join(output_dir, 
                                f"logit_distance_{model_part}_{traits_suffix}.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {combined_file}\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Logit distance equivalence test for steering vectors")
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        help="Models to test (default: both Qwen and Llama)"
    )
    
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=["formality", "politeness", "sentiment", "truthfulness", "agreeableness"],
        help="Traits to test. Use 'all' to test all 5 traits (default: all 5 traits). Available: formality, politeness, sentiment, truthfulness, agreeableness"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="../results",
        help="Output directory for results (default: ../results)"
    )
    
    args = parser.parse_args()
    
    # Expand 'all' to full trait list
    traits = []
    for trait in args.traits:
        if trait.lower() == "all":
            traits.extend(["formality", "politeness", "sentiment", "truthfulness", "agreeableness"])
        else:
            traits.append(trait)
    
    # Remove duplicates while preserving order
    seen = set()
    traits = [t for t in traits if not (t in seen or seen.add(t))]
    
    results = run_logit_distance_test(
        model_names=args.models,
        traits=traits,
        output_dir=args.output
    )
