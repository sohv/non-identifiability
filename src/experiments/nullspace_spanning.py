#!/usr/bin/env python
"""
Null-Space Spanning: Subspace Equivalence

Constructs k mutually orthogonal directions via iterative Gram-Schmidt.
Tests if the equivalence class forms a high-dimensional affine subspace.

Experiment structure:
  (a) Individual direction checks: verify each v⊥_i individually achieves
      equivalent efficacy (Cohen's d < 0.2 vs. v)
  (b) Subspace direction steering: for each k, sample 5 random unit vectors
      from span{v⊥₁,...,v⊥_k} and measure Cohen's d. Plot mean d vs. k.

Theoretical prediction: d stays near zero for all k up to null-space dimension,
then rises sharply at k ≈ NF(l)×d (inflection point).

Usage:
    python src/experiments/nullspace_spanning.py --trait formality --n_individual_checks 50 --n_subspace_samples 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

try:
    from persona_vector_experiment import PersonaVectorExperiment
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from persona_vector_experiment import PersonaVectorExperiment


def gram_schmidt_orthogonal(v: torch.Tensor, n_orthogonal: int = 50) -> List[torch.Tensor]:
    """
    Generate n orthogonal directions to v using Gram-Schmidt.
    
    Args:
        v: Steering vector (reference direction)
        n_orthogonal: Number of orthogonal directions to generate
        
    Returns:
        List of orthogonal vectors with same magnitude as v
    """
    v_norm = torch.norm(v).item()
    v_unit = v / (torch.norm(v) + 1e-8)
    
    orthogonal_dirs = []
    
    for _ in range(n_orthogonal):
        random_vec = torch.randn_like(v)
        
        projection_onto_v = (random_vec @ v_unit) * v_unit
        v_perp = random_vec - projection_onto_v
        
        v_perp = v_perp / (torch.norm(v_perp) + 1e-8)
        v_perp = v_perp * v_norm
        
        dot_product = (v_perp @ v_unit).item()
        assert abs(dot_product) < 1e-3, f"Orthogonality check failed: dot={dot_product:.6f}"
        
        orthogonal_dirs.append(v_perp)
    
    return orthogonal_dirs


def compute_jsd(probs1: np.ndarray, probs2: np.ndarray) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        probs1: Probability distribution 1
        probs2: Probability distribution 2
        
    Returns:
        JSD value (always in [0, 1])
    """
    probs1 = np.maximum(probs1, 1e-8)
    probs1 = probs1 / np.sum(probs1)
    
    probs2 = np.maximum(probs2, 1e-8)
    probs2 = probs2 / np.sum(probs2)
    
    jsd = jensenshannon(probs1, probs2)
    
    if np.isnan(jsd) or np.isinf(jsd):
        return 0.0
    
    return float(jsd)


def compute_effect_size(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """
    Compute Cohen's d between two score distributions.
    
    Args:
        scores1: Scores from first distribution
        scores2: Scores from second distribution
        
    Returns:
        Cohen's d effect size
    """
    if len(scores1) == 0 or len(scores2) == 0:
        return 0.0
    
    scores1 = np.array(scores1, dtype=np.float32)
    scores2 = np.array(scores2, dtype=np.float32)
    
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    std1 = np.std(scores1)
    std2 = np.std(scores2)
    
    if std1 == 0 and std2 == 0:
        return 0.0
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = abs(mean1 - mean2) / pooled_std
    
    if np.isnan(cohens_d) or np.isinf(cohens_d):
        return 0.0
    
    return float(np.clip(cohens_d, 0, 10))


def subexperiment_a_individual_checks(
    experiment: PersonaVectorExperiment,
    trait: str,
    v: torch.Tensor,
    orthogonal_dirs: List[torch.Tensor],
    test_prompts: List[str],
    alpha: float = 1.0
) -> Dict:
    """
    Sub-experiment (a): Verify each v⊥_i individually achieves equivalent efficacy.
    
    Args:
        experiment: Experiment instance
        trait: Semantic trait
        v: Base steering vector
        orthogonal_dirs: List of orthogonal directions
        test_prompts: Prompts for testing
        alpha: Steering strength
        
    Returns:
        Results dictionary with per-direction metrics
    """
    print(f"\nSub-experiment (a): Individual Direction Checks")
    print(f"  Testing {len(orthogonal_dirs)} orthogonal directions...")
    
    scores_v = []
    for prompt in tqdm(test_prompts[:30], desc="Generating with v", leave=False):
        try:
            texts = experiment.generate_with_steering(
                prompt, v, alpha=alpha, max_new_tokens=50, num_return_sequences=3
            )
            for text in texts:
                score = experiment.compute_semantic_score(text, trait)
                if not np.isnan(score) and not np.isinf(score):
                    scores_v.append(score)
        except Exception:
            continue
    
    if len(scores_v) == 0:
        return {'error': 'Could not generate samples with v'}
    
    mean_v = np.mean(scores_v)
    std_v = np.std(scores_v)
    
    individual_results = []
    
    for i, v_perp in enumerate(orthogonal_dirs):
        scores_perp = []
        
        for prompt in tqdm(
            test_prompts[:30],
            desc=f"Direction {i+1}/{len(orthogonal_dirs)}",
            leave=False
        ):
            try:
                texts = experiment.generate_with_steering(
                    prompt, v_perp, alpha=alpha, max_new_tokens=50, num_return_sequences=3
                )
                for text in texts:
                    score = experiment.compute_semantic_score(text, trait)
                    if not np.isnan(score) and not np.isinf(score):
                        scores_perp.append(score)
            except Exception:
                continue
        
        if len(scores_perp) > 0:
            cohens_d = compute_effect_size(scores_v, scores_perp)
            jsd = compute_jsd(
                np.histogram(scores_v, bins=10, range=(0, 1))[0],
                np.histogram(scores_perp, bins=10, range=(0, 1))[0]
            )
            
            individual_results.append({
                'direction_idx': i,
                'cohens_d': cohens_d,
                'jsd': jsd,
                'mean_score': float(np.mean(scores_perp)),
                'std_score': float(np.std(scores_perp))
            })
    
    return {
        'reference_v': {
            'mean_score': float(mean_v),
            'std_score': float(std_v),
            'n_samples': len(scores_v)
        },
        'individual_directions': individual_results,
        'n_directions_tested': len(individual_results),
        'mean_cohens_d': float(np.mean([r['cohens_d'] for r in individual_results])),
        'max_cohens_d': float(np.max([r['cohens_d'] for r in individual_results]))
    }


def subexperiment_b_subspace_steering(
    experiment: PersonaVectorExperiment,
    trait: str,
    v: torch.Tensor,
    orthogonal_dirs: List[torch.Tensor],
    test_prompts: List[str],
    k_values: List[int] = None,
    n_samples_per_k: int = 5,
    alpha: float = 1.0
) -> Dict:
    """
    Sub-experiment (b): For each k, sample random unit vectors from span{v⊥₁,...,v⊥_k}.
    Measure Cohen's d between subspace direction and v.
    
    Args:
        experiment: Experiment instance
        trait: Semantic trait
        v: Base steering vector
        orthogonal_dirs: List of orthogonal directions
        test_prompts: Prompts for testing
        k_values: Values of k to test
        n_samples_per_k: Number of random samples per k
        alpha: Steering strength
        
    Returns:
        Results dictionary with d vs. k trajectory
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, min(50, len(orthogonal_dirs))]
    
    print(f"\nSub-experiment (b): Subspace Direction Steering")
    print(f"  Testing k={k_values} with {n_samples_per_k} samples per k...")
    
    scores_v = []
    for prompt in tqdm(test_prompts[:30], desc="Reference v", leave=False):
        try:
            texts = experiment.generate_with_steering(
                prompt, v, alpha=alpha, max_new_tokens=50, num_return_sequences=3
            )
            for text in texts:
                score = experiment.compute_semantic_score(text, trait)
                if not np.isnan(score) and not np.isinf(score):
                    scores_v.append(score)
        except Exception:
            continue
    
    if len(scores_v) == 0:
        return {'error': 'Could not generate samples with v'}
    
    k_results = {}
    
    for k in k_values:
        if k > len(orthogonal_dirs):
            continue
        
        print(f"\n  Testing k={k}...")
        
        subspace_basis = orthogonal_dirs[:k]
        
        per_sample_cohens_d = []
        per_sample_jsd = []
        
        for sample_idx in range(n_samples_per_k):
            coefficients = np.random.randn(k)
            coefficients = coefficients / (np.linalg.norm(coefficients) + 1e-8)
            
            subspace_direction = torch.zeros_like(v)
            for coeff, direction in zip(coefficients, subspace_basis):
                subspace_direction = subspace_direction + coeff * direction
            
            subspace_direction = subspace_direction / (torch.norm(subspace_direction) + 1e-8)
            subspace_direction = subspace_direction * torch.norm(v)
            
            scores_subspace = []
            
            for prompt in tqdm(
                test_prompts[:30],
                desc=f"k={k}, sample {sample_idx+1}/{n_samples_per_k}",
                leave=False
            ):
                try:
                    texts = experiment.generate_with_steering(
                        prompt, subspace_direction, alpha=alpha,
                        max_new_tokens=50, num_return_sequences=3
                    )
                    for text in texts:
                        score = experiment.compute_semantic_score(text, trait)
                        if not np.isnan(score) and not np.isinf(score):
                            scores_subspace.append(score)
                except Exception:
                    continue
            
            if len(scores_subspace) > 0:
                cohens_d = compute_effect_size(scores_v, scores_subspace)
                jsd = compute_jsd(
                    np.histogram(scores_v, bins=10, range=(0, 1))[0],
                    np.histogram(scores_subspace, bins=10, range=(0, 1))[0]
                )
                
                per_sample_cohens_d.append(cohens_d)
                per_sample_jsd.append(jsd)
        
        if len(per_sample_cohens_d) > 0:
            k_results[k] = {
                'cohens_d_per_sample': per_sample_cohens_d,
                'jsd_per_sample': per_sample_jsd,
                'mean_cohens_d': float(np.mean(per_sample_cohens_d)),
                'std_cohens_d': float(np.std(per_sample_cohens_d)),
                'mean_jsd': float(np.mean(per_sample_jsd)),
                'std_jsd': float(np.std(per_sample_jsd))
            }
    
    return {
        'k_results': k_results,
        'k_tested': sorted(k_results.keys())
    }


def detect_inflection_point(k_values: List[int], d_values: List[float]) -> Optional[int]:
    """
    Detect inflection point where Cohen's d rises sharply.
    
    Args:
        k_values: k values tested
        d_values: Cohen's d values at each k
        
    Returns:
        k value at inflection point, or None if no clear inflection
    """
    if len(k_values) < 3:
        return None
    
    d_values = np.array(d_values)
    
    second_diff = np.diff(d_values, n=2)
    
    threshold = np.std(second_diff) + np.mean(second_diff)
    
    inflection_candidates = np.where(second_diff > threshold)[0]
    
    if len(inflection_candidates) > 0:
        return k_values[inflection_candidates[0] + 1]
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="A5 — Test null-space spanning and equivalence class dimensionality"
    )
    
    parser.add_argument(
        "--traits",
        nargs='+',
        type=str,
        default=["formality"],
        choices=["formality", "politeness", "sentiment", "truthfulness", "agreeableness"],
        help="Traits to test (default: formality)"
    )
    
    parser.add_argument(
        "--models",
        nargs='+',
        type=str,
        default=["Qwen/Qwen2.5-3B-Instruct"],
        help="Models to test (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    
    parser.add_argument(
        "--n_individual_checks",
        type=int,
        default=50,
        help="Number of individual orthogonal directions to test (part a)"
    )
    
    parser.add_argument(
        "--n_subspace_samples",
        type=int,
        default=5,
        help="Number of random samples per k (part b)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: src/results)"
    )
    
    args = parser.parse_args()
    
    # Models and traits to run
    models = args.models
    traits = args.traits
    
    # List to aggregate all results
    all_results = []
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, '..', 'results')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"A5 — Null-Space Spanning: Subspace Equivalence")
    print(f"{'='*80}\n")
    print(f"Models: {models}")
    print(f"Traits: {traits}")
    print(f"Individual checks: {args.n_individual_checks} directions")
    print(f"Subspace samples: {args.n_subspace_samples} per k\n")
    
    # Loop over all model-trait combinations
    total_tests = len(models) * len(traits)
    test_count = 0
    
    for model_name in models:
        for trait in traits:
            test_count += 1
            print(f"[{test_count}/{total_tests}] {trait.upper()} on {model_name.split('/')[-1]}...")
            
            try:
                experiment = PersonaVectorExperiment(model_name)
                config_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'config'
                )
                with open(os.path.join(config_dir, 'prompts.json'), 'r') as f:
                    config = json.load(f)
                
                test_prompts = config['orthogonal_prompts'][:50]
                
                print(f"  Extracting steering vector...")
                v = experiment.extract_steering_vector(trait, n_pairs=50)
                
                print(f"  Generating {args.n_individual_checks} orthogonal directions...")
                orthogonal_dirs = gram_schmidt_orthogonal(v, args.n_individual_checks)
                
                results_a = subexperiment_a_individual_checks(
                    experiment, trait, v, orthogonal_dirs, test_prompts, alpha=args.alpha
                )
                
                results_b = subexperiment_b_subspace_steering(
                    experiment, trait, v, orthogonal_dirs, test_prompts,
                    n_samples_per_k=args.n_subspace_samples, alpha=args.alpha
                )
                
                inflection_k = None
                if 'k_results' in results_b:
                    k_values = sorted(results_b['k_results'].keys())
                    d_values = [results_b['k_results'][k]['mean_cohens_d'] for k in k_values]
                    inflection_k = detect_inflection_point(k_values, d_values)
                
                final_results = {
                    'trait': trait,
                    'model': model_name,
                    'subexperiment_a': results_a,
                    'subexperiment_b': results_b,
                    'inflection_point_k': inflection_k,
                    'alpha': args.alpha
                }
                
                # Append to aggregated results list
                all_results.append(final_results)
                
                print(f"  ✓ Collected result")

            
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Write aggregated results to single file
    if all_results:
        # Generate output filename with all traits and models used
        sorted_traits = sorted(set(r['trait'] for r in all_results))
        traits_str = '_'.join(sorted_traits)
        
        # Extract model short names (last part after /)
        model_shorts = []
        for model_name in models:
            short_name = model_name.split('/')[-1].lower()
            # Map common model names to short names
            if 'qwen' in short_name:
                short_name = 'qwen'
            elif 'llama' in short_name:
                short_name = 'llama'
            model_shorts.append(short_name)
        models_str = '_'.join(sorted(set(model_shorts)))
        
        output_filename = f"nullspace_spanning_{traits_str}_{models_str}.json"
        output_file = Path(args.output_dir) / output_filename
        
        aggregated_output = {
            'results': all_results,
            'summary': {
                'num_tests': len(all_results),
                'traits_tested': sorted_traits,
                'models_tested': list(set(models))
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(aggregated_output, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_filename}")
        print(f"  Tests completed: {len(all_results)}")
        print(f"  Traits: {', '.join(sorted_traits)}")
        print(f"  Models: {', '.join(set(model_shorts))}")
        print(f"{'='*80}\n")
    else:
        print("\nNo results to save.")
    
    # Summary statistics
    if all_results:
        print(f"Summary of results:")
        for result in all_results:
            trait = result['trait']
            model_short = result['model'].split('/')[-1].split('-')[0].lower()
            results_a = result['subexperiment_a']
            results_b = result['subexperiment_b']
            
            print(f"\n  [{trait.upper()} on {model_short}]")
            print(f"    Sub-experiment (a):")
            print(f"      Directions tested: {results_a.get('n_directions_tested', 'N/A')}")
            print(f"      Mean Cohen's d: {results_a.get('mean_cohens_d', 'N/A'):.4f}")
            print(f"      Max Cohen's d: {results_a.get('max_cohens_d', 'N/A'):.4f}")
            
            if 'k_results' in results_b and len(results_b['k_results']) > 0:
                k_values = sorted(results_b['k_results'].keys())
                print(f"    Sub-experiment (b): k ∈ {k_values}")
                inflection_k = result.get('inflection_point_k')
                if inflection_k:
                    print(f"      Inflection point at k ≈ {inflection_k}")


if __name__ == "__main__":
    main()
