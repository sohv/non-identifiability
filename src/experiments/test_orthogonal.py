"""
Test orthogonal component irrelevance for non-identifiability.

If steering vectors are non-identifiable, then adding components
orthogonal to v should not change the semantic effect (they're in the null space).

Usage:
    cd src/experiments && python test_orthogonal.py --traits formality politeness humor --n_seeds 5
"""

import json
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm

# Handle imports when run from src/experiments/
try:
    from persona_vector_experiment import PersonaVectorExperiment
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from persona_vector_experiment import PersonaVectorExperiment


def test_orthogonal_irrelevance(model_name: str, trait: str, n_orthogonal: int = 5):
    """
    Test if orthogonal components to steering vector v are irrelevant.
    
    Args:
        model_name: HuggingFace model name
        trait: 'formality' or 'sentiment'
        n_orthogonal: Number of random orthogonal vectors to test
    
    Returns:
        Dictionary with test results
    """
    print(f"\nOrthogonal Component Irrelevance Test")
    print(f"Model: {model_name}")
    print(f"Trait: {trait}\n")
    
    # Initialize
    experiment = PersonaVectorExperiment(model_name)
    
    # Load prompts from config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
    with open(os.path.join(config_dir, 'prompts.json'), 'r') as f:
        config = json.load(f)
    test_prompts = config['orthogonal_prompts']
    
    # Step 1: Extract steering vector
    print(f"Step 1: Extracting {trait} steering vector...")
    v = experiment.extract_steering_vector(trait, n_pairs=50)
    v_norm = torch.norm(v).item()
    print(f"  [OK] Steering vector extracted")
    print(f"  [OK] Shape: {v.shape}")
    print(f"  [OK] Norm: {v_norm:.4f}\n")
    
    # Step 2: Generate with original vector v
    print(f"Step 2: Generating with original vector v...")
    scores_v = []
    for prompt in tqdm(test_prompts[:20], desc="Original v"):
        try:
            texts = experiment.generate_with_steering(
                prompt, v, alpha=1.0, max_new_tokens=40, num_return_sequences=5
            )
            for text in texts:
                score = experiment.compute_semantic_score(text, trait)
                scores_v.append(score)
        except Exception as e:
            continue
    
    mean_v = np.mean(scores_v)
    std_v = np.std(scores_v)
    print(f"  [OK] Generated {len(scores_v)} samples")
    print(f"  [OK] Semantic scores: mean={mean_v:.4f}, std={std_v:.4f}\n")
    
    # Step 3: Test v + orthogonal components
    print(f"Step 3: Testing v + orthogonal components...")
    results_v_plus_perp = []
    
    for i in range(n_orthogonal):
        print(f"\n  Testing orthogonal vector {i+1}/{n_orthogonal}...")
        
        # Create random vector orthogonal to v using Gram-Schmidt
        random_vec = torch.randn_like(v)
        v_perp = random_vec - (random_vec @ v) / (v @ v) * v
        v_perp = v_perp / torch.norm(v_perp) * v_norm  # Same magnitude as v
        v_plus_perp = v + v_perp
        
        # Verify orthogonality
        dot_product = (v @ v_perp).item()
        print(f"    v · v_perp = {dot_product:.6f} (should be ~0)")
        
        # Generate with v + v_perp
        scores_perp = []
        for prompt in tqdm(test_prompts[:20], desc=f"    v+perp {i+1}", leave=False):
            try:
                texts = experiment.generate_with_steering(
                    prompt, v_plus_perp, alpha=1.0, max_new_tokens=40, num_return_sequences=5
                )
                for text in texts:
                    score = experiment.compute_semantic_score(text, trait)
                    scores_perp.append(score)
            except Exception as e:
                continue
        
        mean_perp = np.mean(scores_perp)
        std_perp = np.std(scores_perp)
        
        # Compute metrics
        mean_diff = abs(mean_perp - mean_v)
        pooled_std = np.sqrt((std_v**2 + std_perp**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        correlation = np.corrcoef(scores_v[:len(scores_perp)], scores_perp)[0, 1]
        
        print(f"    Scores: mean={mean_perp:.4f}, std={std_perp:.4f}")
        print(f"    Cohen's d: {cohens_d:.4f}")
        print(f"    Correlation: {correlation:.4f}")
        
        results_v_plus_perp.append({
            'cohens_d': float(cohens_d),
            'correlation': float(correlation),
            'mean_diff': float(mean_diff),
            'mean_perp': float(mean_perp),
            'std_perp': float(std_perp)
        })
    
    # Step 4: Test pure orthogonal components (without v)
    print(f"\nStep 4: Testing pure orthogonal components (without v)...")
    perp_only_effects = []
    
    for i in range(n_orthogonal):
        # Create orthogonal vector
        random_vec = torch.randn_like(v)
        v_perp = random_vec - (random_vec @ v) / (v @ v) * v
        v_perp = v_perp / torch.norm(v_perp) * v_norm
        
        scores_only_perp = []
        for prompt in tqdm(test_prompts[:20], desc=f"  perp-only {i+1}", leave=False):
            try:
                texts = experiment.generate_with_steering(
                    prompt, v_perp, alpha=1.0, max_new_tokens=40, num_return_sequences=5
                )
                for text in texts:
                    score = experiment.compute_semantic_score(text, trait)
                    scores_only_perp.append(score)
            except Exception as e:
                continue
        
        mean_only_perp = np.mean(scores_only_perp)
        effect_ratio = abs(mean_only_perp) / abs(mean_v) if abs(mean_v) > 1e-6 else 0
        perp_only_effects.append(float(effect_ratio))
        
        print(f"    Orthogonal {i+1}: effect = {abs(mean_only_perp):.4f} ({effect_ratio*100:.1f}% of v)")
    
    cohens_ds = [r['cohens_d'] for r in results_v_plus_perp]
    correlations = [r['correlation'] for r in results_v_plus_perp]
    
    print(f"\nResults Summary")
    print(f"Original vector v:")
    print(f"  Mean semantic score: {mean_v:.4f} ± {std_v:.4f}")
    print(f"\nv + orthogonal components:")
    print(f"  Cohen's d: {np.mean(cohens_ds):.4f} ± {np.std(cohens_ds):.4f}")
    print(f"  Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"\nPure orthogonal components:")
    print(f"  Mean effect: {np.mean(perp_only_effects)*100:.1f}% ± {np.std(perp_only_effects)*100:.1f}%\n")
    
    return {
        'model': model_name,
        'trait': trait,
        'v_scores': {'mean': float(mean_v), 'std': float(std_v)},
        'v_plus_perp': results_v_plus_perp,
        'perp_only_effects': perp_only_effects,
        'summary': {
            'mean_cohens_d': float(np.mean(cohens_ds)),
            'std_cohens_d': float(np.std(cohens_ds)),
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'mean_perp_effect': float(np.mean(perp_only_effects))
        }
    }


def main():
    """Run orthogonal component test with command line arguments."""
    parser = argparse.ArgumentParser(description="Test orthogonal component irrelevance")
    parser.add_argument('--traits', nargs='+', default=['formality', 'politeness', 'sentiment'],
                       choices=['formality', 'politeness', 'sentiment', 'truthfulness', 'agreeableness'],
                       help='Traits to test')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of orthogonal seeds to test')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                       help='Model name')
    
    args = parser.parse_args()
    model_name = args.model
    traits = args.traits
    n_seeds = args.n_seeds
    
    all_results = {}
    
    for trait in traits:
        print(f"\n\nTESTING TRAIT: {trait.upper()}\n")
        
        try:
            results = test_orthogonal_irrelevance(model_name, trait, n_orthogonal=n_seeds)
            all_results[trait] = results
        except Exception as e:
            print(f"\n[ERROR] Failed testing {trait}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    os.makedirs("../results", exist_ok=True)

    # Determine model suffix for filename
    if "qwen" in model_name.lower():
        model_suffix = "qwen"
    elif "llama" in model_name.lower():
        model_suffix = "llama"
    else:
        # Use a sanitized version of model name as fallback
        model_suffix = model_name.split('/')[-1].lower().replace('-', '_')

    # Build traits suffix (combine multiple traits with underscore)
    traits_suffix = "_".join(traits)
    
    output_file = f"../results/orthogonal_test_results_{traits_suffix}_{model_suffix}_{n_seeds}_seeds.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFINAL RESULTS")
    for trait, results in all_results.items():
        if 'summary' in results:
            d = results['summary']['mean_cohens_d']
            r = results['summary']['mean_correlation']
            print(f"{trait.capitalize():15s}: (d={d:.3f}, r={r:.3f})")
    print(f"\n[SAVED] Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
