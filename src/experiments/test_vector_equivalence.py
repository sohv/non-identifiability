"""
Non-Orthogonal Vector Equivalence Test.

Tests whether geometrically distinct vectors (extracted via different methods)
produce statistically indistinguishable output distributions. This validates that
equivalence classes contain genuinely non-similar vectors, not just orthogonal ones.

Method:
  1. Extract v₁ via contrast vector method (Turner et al. 2023)
  2. Extract v₂ via PCA on difference vectors (Zou et al. 2023)
  3. Compute cos(v₁, v₂)
  4. Generate 100 outputs steered by v₁ and v₂
  5. Measure equivalence: Cohen's d effect size
  6. Report cos(v₁, v₂) alongside Cohen's d

Expected: Low cos similarity + low Cohen's d = equivalence class contains 
geometrically distant vectors.

Usage:
    cd src/experiments && python test_vector_equivalence.py \
        --models Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.1-8B-Instruct \
        --traits formality politeness sentiment truthfulness agreeableness
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm

try:
    from persona_vector_experiment import PersonaVectorExperiment
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from persona_vector_experiment import PersonaVectorExperiment


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d (effect size) between two groups.
    
    Args:
        group1: Array of values from first group
        group2: Array of values from second group
        
    Returns:
        Cohen's d value (positive = group1 higher)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)





def generate_with_steering(experiment: PersonaVectorExperiment,
                           prompt: str,
                           steering_vector: torch.Tensor,
                           alpha: float = 1.0,
                           max_new_tokens: int = 40,
                           temperature: float = 0.8) -> str:
    """
    Generate text with steering applied.
    
    Args:
        experiment: PersonaVectorExperiment instance
        prompt: Input prompt
        steering_vector: Steering vector to apply
        alpha: Steering strength multiplier
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    inputs = experiment.tokenizer(prompt, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512)
    inputs = {k: v.to(experiment.device) for k, v in inputs.items()}
    
    layer_idx = len(experiment.model.model.layers) // 2
    
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
        
        steering_gpu = steering_vector.to(device=hidden_states.device,
                                         dtype=hidden_states.dtype)
        hidden_states[:, -1, :] += alpha * steering_gpu
        
        if rest:
            return (hidden_states,) + rest
        else:
            return hidden_states
    
    hook_handle = experiment.model.model.layers[layer_idx].register_forward_hook(
        steering_hook)
    
    with torch.no_grad():
        generated_ids = experiment.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=experiment.tokenizer.eos_token_id
        )
    
    hook_handle.remove()
    
    generated_ids = generated_ids[0, inputs['input_ids'].shape[1]:]
    text = experiment.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return text


def test_vector_equivalence(model_name: str, trait: str = "formality",
                           n_samples: int = 100):
    """
    Test equivalence between contrast vector and PCA vector.
    
    Args:
        model_name: HuggingFace model name
        trait: Semantic trait
        n_samples: Number of samples to generate per vector
        
    Returns:
        Dictionary with equivalence metrics
    """
    print(f"  Testing {trait}...", flush=True)
    
    experiment = PersonaVectorExperiment(model_name)
    
    # Test prompts
    test_prompts = [
        "Write about your thoughts on",
        "Describe your experience with",
        "Share your opinion about",
        "Explain your view on",
        "Discuss your perspective about",
        "Tell me what you think of",
        "Express your feelings about",
        "Give me your take on",
        "Talk about your experience with",
        "What are your thoughts on",
    ] * 10  # 100 total
    
    # Extract vectors
    v1 = experiment.extract_steering_vector(trait, n_pairs=50)
    v2 = experiment.extract_steering_vector_pca(trait, n_pairs=50)
    
    # Normalize for fair comparison
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)
    
    # Compute cosine similarity
    cos_sim = (v1 @ v2).item()
    
    print(f"    cos(v₁, v₂) = {cos_sim:.4f}")
    
    # Generate outputs
    scores_v1 = []
    scores_v2 = []
    
    for prompt in tqdm(test_prompts, desc=f"    Generating with v₁"):
        try:
            text_v1 = generate_with_steering(
                experiment, prompt, v1, alpha=1.0, max_new_tokens=40)
            score_v1 = experiment.compute_semantic_score(text_v1, trait)
            scores_v1.append(score_v1)
        except Exception as e:
            continue
    
    for prompt in tqdm(test_prompts, desc=f"    Generating with v₂"):
        try:
            text_v2 = generate_with_steering(
                experiment, prompt, v2, alpha=1.0, max_new_tokens=40)
            score_v2 = experiment.compute_semantic_score(text_v2, trait)
            scores_v2.append(score_v2)
        except Exception as e:
            continue
    
    # Filter NaN values
    scores_v1 = np.array([s for s in scores_v1 if not np.isnan(s)])
    scores_v2 = np.array([s for s in scores_v2 if not np.isnan(s)])
    
    # Compute Cohen's d
    cohens_d = 0.0
    if len(scores_v1) > 1 and len(scores_v2) > 1:
        cohens_d = compute_cohens_d(scores_v1, scores_v2)
    
    return {
        'model': model_name,
        'trait': trait,
        'cos_similarity': float(cos_sim),
        'n_samples': len(scores_v1),
        'metrics': {
            'mean_score_v1': float(np.mean(scores_v1)) if len(scores_v1) > 0 else 0.0,
            'mean_score_v2': float(np.mean(scores_v2)) if len(scores_v2) > 0 else 0.0,
            'cohens_d': float(cohens_d),
        },
        'v_norms': {
            'v1': float(torch.norm(v1).item()),
            'v2': float(torch.norm(v2).item()),
        }
    }


def main():
    """Run vector equivalence tests across models and traits."""
    parser = argparse.ArgumentParser(
        description="Non-orthogonal vector equivalence test")
    parser.add_argument('--models', nargs='+',
                       default=['Qwen/Qwen2.5-3B-Instruct',
                               'meta-llama/Llama-3.1-8B-Instruct'],
                       help='Models to test')
    parser.add_argument('--traits', nargs='+',
                       default=['formality', 'sentiment'],
                       choices=['formality', 'politeness', 'sentiment',
                               'truthfulness', 'agreeableness'],
                       help='Traits to test')
    
    args = parser.parse_args()
    models = args.models
    traits = sorted(args.traits)
    
    print(f"\n{'='*70}")
    print(f"Non-Orthogonal Vector Equivalence Test")
    print(f"{'='*70}")
    print(f"Models: {models}")
    print(f"Traits: {traits}")
    print(f"Testing: v₁ (contrast) vs v₂ (PCA)")
    print(f"Metrics: cosine similarity, Cohen's d, JS divergence")
    print(f"{'='*70}\n")
    
    all_results = {}
    total_tests = len(models) * len(traits)
    test_count = 0
    
    for model_name in models:
        model_short = 'Llama' if 'llama' in model_name.lower() else 'Qwen'
        print(f"\nTesting {model_short}...", flush=True)
        all_results[model_name] = {}
        
        for trait in traits:
            test_count += 1
            print(f"[{test_count}/{total_tests}] {trait.upper()}...", flush=True)
            
            try:
                results = test_vector_equivalence(model_name, trait=trait)
                all_results[model_name][trait] = results
                print(f"  ✓ Complete (cos={results['cos_similarity']:.4f}, "
                      f"d={results['metrics']['cohens_d']:.3f})")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    os.makedirs("../results", exist_ok=True)
    traits_suffix = "_".join(traits)
    output_file = f"../results/vector_equivalence_{traits_suffix}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[SAVED] {output_file}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("Summary: Vector Equivalence Metrics")
    print(f"{'='*70}")
    cohens_label = "Cohen's d"
    print(f"{'Trait':<15} {'Model':<12} {'cos(v₁,v₂)':<12} {cohens_label:<12}")
    print(f"{'-'*70}")
    
    for model_name in models:
        model_short = 'Llama' if 'llama' in model_name.lower() else 'Qwen'
        for trait in traits:
            if trait in all_results.get(model_name, {}):
                r = all_results[model_name][trait]
                print(f"{trait:<15} {model_short:<12} "
                      f"{r['cos_similarity']:<12.4f} "
                      f"{r['metrics']['cohens_d']:<12.3f}")
    
    print(f"{'='*70}\n")
    
    # Interpretation
    print("Interpretation:")
    print("  low cos(v₁,v₂) + low Cohen's d → equivalence class is large (vectors non-orthogonal but equivalent)")
    print("  high cos(v₁,v₂) + high Cohen's d → equivalence class is narrow (similar vectors, similar effects)")
    print()


if __name__ == "__main__":
    main()
