"""
Alpha sweep test for all traits across multiple models.

Tests steering strength α ∈ [0.0, 0.5, 1.0, 2.0] for formality, politeness, 
humor, and sentiment across two models. Creates subplot visualization with 
one row per model, one column per trait.

Usage:
    cd src/experiments && python alpha_sweep.py --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-3B-Instruct --traits formality politeness humor sentiment --alphas 0.0 0.5 1.0 2.0 --n_seeds 10
"""

import json
import torch
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Handle imports when run from src/experiments/
try:
    from persona_vector_experiment import PersonaVectorExperiment
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from persona_vector_experiment import PersonaVectorExperiment

# ACL paper formatting: set before creating figures
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
})


def test_alpha_sweep(model_name: str, trait: str = "formality", 
                    alphas: list = None, n_seeds: int = 10):
    """
    Test steering strength α sweep for a specific trait.
    
    If non-identifiable, v and v+v_perp should scale similarly across alphas.
    
    Args:
        model_name: HuggingFace model name
        trait: Semantic trait (formality, politeness, humor, sentiment)
        alphas: List of steering strength multipliers
        n_seeds: Number of random orthogonal seeds per alpha
    
    Returns:
        Dictionary with alpha sweep results
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0]
    
    print(f"  Testing {trait}...", flush=True)
    
    # Initialize
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
    ]
    
    # Extract steering vector
    v = experiment.extract_steering_vector(trait, n_pairs=50)
    v_norm = torch.norm(v).item()
    
    # Alpha sweep
    alpha_results = {}
    
    for alpha in alphas:
        alpha_data = {
            'alpha': alpha,
            'seeds': []
        }
        
        for seed_idx in range(n_seeds):
            # Create random orthogonal vector
            random_vec = torch.randn_like(v)
            v_perp = random_vec - (random_vec @ v) / (v @ v) * v
            v_perp = v_perp / torch.norm(v_perp) * v_norm
            
            # Test vectors: v, v+v_perp
            v_steered = v * alpha
            v_plus_perp_steered = (v + v_perp) * alpha
            
            # Generate with v
            scores_v = []
            for prompt in test_prompts[:5]:
                try:
                    texts = experiment.generate_with_steering(
                        prompt, v_steered, alpha=1.0, max_new_tokens=40, num_return_sequences=3
                    )
                    for text in texts:
                        score = experiment.compute_semantic_score(text, trait)
                        scores_v.append(score)
                except:
                    continue
            
            # Generate with v + v_perp
            scores_v_perp = []
            for prompt in test_prompts[:5]:
                try:
                    texts = experiment.generate_with_steering(
                        prompt, v_plus_perp_steered, alpha=1.0, max_new_tokens=40, num_return_sequences=3
                    )
                    for text in texts:
                        score = experiment.compute_semantic_score(text, trait)
                        scores_v_perp.append(score)
                except:
                    continue
            
            mean_v = np.mean(scores_v) if scores_v else 0.0
            mean_perp = np.mean(scores_v_perp) if scores_v_perp else 0.0
            
            # Compute effect
            effect_diff = abs(mean_perp - mean_v)
            
            alpha_data['seeds'].append({
                'seed': seed_idx,
                'mean_v': float(mean_v),
                'mean_v_perp': float(mean_perp),
                'effect_diff': float(effect_diff),
            })
        
        # Compute alpha statistics
        effect_diffs = [s['effect_diff'] for s in alpha_data['seeds']]
        alpha_data['summary'] = {
            'mean_effect_diff': float(np.mean(effect_diffs)),
            'std_effect_diff': float(np.std(effect_diffs)),
        }
        
        alpha_results[f"alpha_{alpha}"] = alpha_data
    
    # Overall summary
    all_diffs = [data['summary']['mean_effect_diff'] for data in alpha_results.values()]
    
    return {
        'model': model_name,
        'trait': trait,
        'alphas': alphas,
        'n_seeds': n_seeds,
        'alpha_results': alpha_results,
        'v_norm': float(v_norm),
        'overall_summary': {
            'mean_effect_across_alphas': float(np.mean(all_diffs)),
            'std_effect_across_alphas': float(np.std(all_diffs))
        }
    }


def plot_multi_trait_sweep(all_results: dict, alphas: list, output_dir: str = "figures"):
    """
    Plot α-sweep for all traits across models in grid layout.
    One row per model, one column per trait.

    Args:
        all_results: Dictionary {model: {trait: results}}
        alphas: List of alpha values tested
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data by model and trait
    models = list(all_results.keys())
    traits = list(all_results[models[0]].keys())
    n_models = len(models)
    n_traits = len(traits)
    
    # Create figure: 2 rows x 4 columns
    fig, axes = plt.subplots(n_models, n_traits, figsize=(12, 4.5))
    
    alphas_array = np.array(alphas)
    
    for model_idx, model_name in enumerate(models):
        for trait_idx, trait in enumerate(traits):
            ax = axes[model_idx, trait_idx] if n_models > 1 else axes[trait_idx]
            
            results = all_results[model_name][trait]
            alpha_results = results['alpha_results']
            
            # Extract data for each alpha
            means_v = []
            stds_v = []
            means_perp = []
            stds_perp = []
            
            for alpha_key in sorted(alpha_results.keys()):
                data = alpha_results[alpha_key]
                
                scores_v = []
                scores_perp = []
                for seed_data in data['seeds']:
                    scores_v.append(seed_data['mean_v'])
                    scores_perp.append(seed_data['mean_v_perp'])
                
                means_v.append(np.mean(scores_v))
                stds_v.append(np.std(scores_v))
                means_perp.append(np.mean(scores_perp))
                stds_perp.append(np.std(scores_perp))
            
            # Plot on this axis
            # v line with band
            ax.plot(alphas_array, means_v, 'o-', linewidth=1.5, markersize=4,
                    label='v', color='#1f77b4')
            ax.fill_between(alphas_array, 
                             np.array(means_v) - np.array(stds_v),
                             np.array(means_v) + np.array(stds_v),
                             alpha=0.2, color='#1f77b4')
            
            # v+v_perp line with band
            ax.plot(alphas_array, means_perp, 's-', linewidth=1.5, markersize=4,
                    label='v + v⊥', color='#ff7f0e')
            ax.fill_between(alphas_array,
                             np.array(means_perp) - np.array(stds_perp),
                             np.array(means_perp) + np.array(stds_perp),
                             alpha=0.2, color='#ff7f0e')
            
            # Labels and formatting
            ax.set_xticks(alphas_array)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Title at top
            if model_idx == 0:
                ax.set_title(f'{trait.capitalize()}', fontsize=8, fontweight='bold')
            
            # Y-axis label on left
            if trait_idx == 0:
                model_short = 'Llama' if 'llama' in model_name.lower() else 'Qwen'
                ax.set_ylabel(f'{model_short}\nScore', fontsize=8)
            
            # X-axis label at bottom
            if model_idx == n_models - 1:
                ax.set_xlabel('α', fontsize=8)
            
            # Legend only on first subplot
            if model_idx == 0 and trait_idx == 0:
                ax.legend(fontsize=7, loc='upper left', framealpha=0.95)
    
    plt.tight_layout()
    
    # Save plot
    output_pdf = f"{output_dir}/alpha_sweep_multi_trait.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PDF (paper-ready): {output_pdf}")

    output_png = output_pdf.replace('.pdf', '.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"[SAVED] PNG (preview): {output_png}")

    plt.close()


def main():
    """Run alpha sweep test across multiple models and traits."""
    parser = argparse.ArgumentParser(description="Alpha sweep test for multiple traits and models")
    parser.add_argument('--models', nargs='+', 
                       default=['Qwen/Qwen2.5-3B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'],
                       help='Model names to test')
    parser.add_argument('--traits', nargs='+', 
                       default=['formality', 'politeness', 'sentiment'],
                       choices=['formality', 'politeness', 'sentiment', 'truthfulness', 'agreeableness'],
                       help='Traits to test')
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.5, 1.0, 2.0],
                       help='Alpha values to test')
    parser.add_argument('--n_seeds', type=int, default=10,
                       help='Number of seeds per alpha')
    
    args = parser.parse_args()
    models = args.models
    traits = sorted(args.traits)
    alphas = sorted(args.alphas)
    n_seeds = args.n_seeds
    
    print(f"\nMulti-Trait Alpha Sweep Configuration:")
    print(f"  Models: {models}")
    print(f"  Traits: {traits}")
    print(f"  Alphas: {alphas}")
    print(f"  Seeds per alpha: {n_seeds}\n")
    
    # Run tests
    all_results = {}
    total_tests = len(models) * len(traits)
    test_count = 0
    
    for model_name in models:
        model_short = 'Llama' if 'llama' in model_name.lower() else 'Qwen'
        print(f"\nTesting {model_short}...")
        all_results[model_name] = {}
        
        for trait in traits:
            test_count += 1
            print(f"  [{test_count}/{total_tests}] {trait}...", flush=True)
            
            try:
                results = test_alpha_sweep(model_name, trait=trait, alphas=alphas, n_seeds=n_seeds)
                all_results[model_name][trait] = results
                print(f"    ✓ {trait} complete")
            except Exception as e:
                print(f"    ✗ {trait} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    os.makedirs("../results", exist_ok=True)
    
    # Build filename with trait names
    traits_suffix = "_".join(traits)
    output_file = f"../results/alpha_sweep_multi_trait_{traits_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[SAVED] Results: {output_file}")
    
    # Generate multi-trait plot
    try:
        plot_multi_trait_sweep(all_results, alphas, output_dir="../results/figures")
    except Exception as e:
        print(f"[WARNING] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()


if __name__ == "__main__":
    main()
