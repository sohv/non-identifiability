"""

Tests Proposition 2's multi-environment condition by validating causal robustness
and invariance across different prompt environments.

This experiment tests:
- Whether v ≡ v+v⊥ holds across environments  
- If different environments produce mutually orthogonal v vectors
- Correlation of semantic scores across environments
- Evidence for environment-dependent identifiability

Usage:
    python multi_environment_validation.py --model Qwen/Qwen2.5-7B-Instruct --traits formality politeness --environments all
"""

import json
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from typing import List, Dict, Tuple
from scipy.stats import pearsonr

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.experiments.persona_vector_experiment import PersonaVectorExperiment


class MultiEnvironmentValidation:
    """Validation of steering vector identifiability across prompt environments."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.experiment = PersonaVectorExperiment(model_name)
        
        # Load environment definitions and validation prompts from config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
        with open(os.path.join(config_dir, 'prompts.json'), 'r') as f:
            config = json.load(f)
        
        multi_env_config = config.get('multi_environment_prompts', {})
        
        # Build environments dictionary from config
        self.environments = {}
        for env_key in ['in_distribution', 'topic_shift', 'genre_shift']:
            if env_key in multi_env_config:
                env_data = multi_env_config[env_key]
                self.environments[env_key] = {
                    'name': env_data['name'],
                    'description': env_data['description'],
                    'prompts': {
                        trait: trait_data 
                        for trait, trait_data in env_data.items() 
                        if trait not in ['name', 'description']
                    }
                }
        
        # Load validation prompts from config
        self.validation_prompts = multi_env_config.get('validation_prompts', [])
    
    def generate_environment_validation_prompts(self, trait: str, environment: str, n_prompts: int = 9) -> List[str]:
        """
        Generate validation prompts specific to an environment using its contexts.
        This ensures orthogonality testing uses the same semantic context as vector extraction.
        
        Args:
            trait: Trait name
            environment: Environment key
            n_prompts: Number of validation prompts to generate (default 9 to match Experiment 3)
            
        Returns:
            List of validation prompts for this environment
        """
        env_config = self.environments[environment]
        trait_prompts = env_config['prompts'][trait]
        
        prompts = []
        
        # Different environments/traits use different field names
        if 'contexts' in trait_prompts:
            contexts = trait_prompts.get('contexts', [])
            for i in range(min(n_prompts, len(contexts))):
                context = contexts[i]
                
                # Generate a neutral prompt (not positive/negative template)
                if 'domains' in trait_prompts:
                    domains = trait_prompts.get('domains', ['general'])
                    domain = domains[i % len(domains)]
                    prompt = f"Discuss {context} in a {domain} context."
                elif 'genres' in trait_prompts:
                    genres = trait_prompts.get('genres', ['general'])
                    genre = genres[i % len(genres)]
                    prompt = f"Write about {context} in the style of a {genre}."
                else:
                    prompt = f"Discuss {context}."
                
                prompts.append(prompt)
        
        elif 'requests' in trait_prompts and 'reasons' in trait_prompts:
            # For safety_style formality: uses requests and reasons
            requests = trait_prompts.get('requests', [])
            reasons = trait_prompts.get('reasons', [])
            for i in range(min(n_prompts, len(requests))):
                request = requests[i]
                reason = reasons[i % len(reasons)]
                prompt = f"Explain why declining to {request} is important due to {reason}."
                prompts.append(prompt)
        
        elif 'requests' in trait_prompts and 'issues' in trait_prompts:
            # For safety_style politeness: uses requests and issues
            requests = trait_prompts.get('requests', [])
            issues = trait_prompts.get('issues', [])
            for i in range(min(n_prompts, len(requests))):
                request = requests[i]
                issue = issues[i % len(issues)]
                prompt = f"Discuss why {request} is problematic due to {issue}."
                prompts.append(prompt)
        
        return prompts
    
    def extract_environment_steering_vector(self, trait: str, environment: str, n_pairs: int = 30) -> torch.Tensor:
        """
        Extract steering vector using prompts from a specific environment.
        
        Args:
            trait: Trait to extract ('formality', 'politeness', 'humor', 'sentiment')
            environment: Environment key ('in_distribution', 'topic_shift', etc.)
            n_pairs: Number of contrastive pairs
            
        Returns:
            Steering vector for this environment
        """
        print(f"Extracting {trait} vector from {environment} environment...")
        
        env_config = self.environments[environment]
        trait_prompts = env_config['prompts'][trait]
        
        # Generate contrastive pairs for this environment
        positive_prompts = []
        negative_prompts = []
        
        # Handle different prompt structures across environments
        # Check more specific conditions FIRST (with domains/genres) before generic ones
        
        if 'domains' in trait_prompts:
            # Domain-cross-topic prompts (topic_shift)
            domains = trait_prompts['domains']
            contexts = trait_prompts['contexts']
            for i in range(n_pairs):
                domain = domains[i % len(domains)]
                context = contexts[i % len(contexts)]
                positive_prompts.append(trait_prompts['positive_template'].format(domain=domain, context=context))
                negative_prompts.append(trait_prompts['negative_template'].format(domain=domain, context=context))
        
        elif 'genres' in trait_prompts:
            # Genre-based prompts (genre_shift)
            genres = trait_prompts['genres']
            contexts = trait_prompts['contexts']
            for i in range(n_pairs):
                genre = genres[i % len(genres)]
                context = contexts[i % len(contexts)]
                positive_prompts.append(trait_prompts['positive_template'].format(genre=genre, context=context))
                negative_prompts.append(trait_prompts['negative_template'].format(genre=genre, context=context))
        
        elif 'requests' in trait_prompts:
            # Safety-style prompts
            requests = trait_prompts['requests']
            
            # Check which key is used for reasons/issues
            if 'reasons' in trait_prompts:
                reasons = trait_prompts['reasons']
                reason_key = 'reason'
            elif 'issues' in trait_prompts:
                reasons = trait_prompts['issues']
                reason_key = 'issue'
            else:
                reasons = ['general concerns']
                reason_key = 'reason'
            
            for i in range(n_pairs):
                request = requests[i % len(requests)]
                reason = reasons[i % len(reasons)]
                positive_prompts.append(trait_prompts['positive_template'].format(request=request, **{reason_key: reason}))
                negative_prompts.append(trait_prompts['negative_template'].format(request=request, **{reason_key: reason}))
        
        else:
            # Context-based prompts (all other traits use 'contexts')
            contexts = trait_prompts['contexts']
            for i in range(n_pairs):
                context = contexts[i % len(contexts)]
                positive_prompts.append(trait_prompts['positive_template'].format(context=context))
                negative_prompts.append(trait_prompts['negative_template'].format(context=context))
        
        # Extract steering vector using environment-specific prompts
        v = self._extract_steering_vector_with_custom_prompts(
            trait, positive_prompts[:n_pairs], negative_prompts[:n_pairs]
        )
        
        print(f"  [OK] {environment} vector shape: {v.shape}")
        return v
    
    def _extract_steering_vector_with_custom_prompts(self, trait: str, positive_prompts: List[str], 
                                                   negative_prompts: List[str]) -> torch.Tensor:
        """
        Extract steering vector using custom positive and negative prompts.
        
        Args:
            trait: Trait being extracted
            positive_prompts: List of positive example prompts
            negative_prompts: List of negative example prompts
            
        Returns:
            Steering vector
        """
        # Directly compute the activations like the extract_steering_vector method does
        positive_activations = []
        negative_activations = []

        print(f"    Computing activations from {len(positive_prompts)} prompt pairs...")
        
        for pos, neg in zip(positive_prompts, negative_prompts):
            pos_hidden = self.experiment.get_hidden_states(pos)
            neg_hidden = self.experiment.get_hidden_states(neg)

            positive_activations.append(pos_hidden)
            negative_activations.append(neg_hidden)

        # Average and compute difference (same as original method)
        pos_mean = torch.stack(positive_activations).mean(dim=0)
        neg_mean = torch.stack(negative_activations).mean(dim=0)

        steering_vector = pos_mean - neg_mean
        return steering_vector
    
    def test_cross_environment_equivalence(self, trait: str, environments: List[str], 
                                         n_orthogonal: int = 5) -> Dict:
        """
        Test if v ≡ v+v⊥ holds across different environments.
        
        Args:
            trait: Trait to test
            environments: List of environment names
            n_orthogonal: Number of orthogonal tests per environment
            
        Returns:
            Dictionary with cross-environment equivalence results
        """
        print(f"\n=== Testing cross-environment equivalence for {trait} ===")
        
        results = {
            'trait': trait,
            'environments': environments,
            'environment_vectors': {},
            'cross_environment_tests': {},
            'environment_orthogonality': {}
        }
        
        # Extract SINGLE steering vector from in_distribution only
        # (tests whether this in-dist v remains orthogonal across different environments)
        v = self.extract_environment_steering_vector(trait, 'in_distribution')
        results['environment_vectors']['in_distribution'] = {
            'norm': torch.norm(v).item(),
            'shape': list(v.shape),
            'note': 'Single vector extracted from in_distribution, tested across all environments'
        }
        
        # Test v ≡ v+v⊥ across each environment using the SAME v
        for env in environments:
            print(f"\n--- Testing {env} environment (using in_distribution steering vector) ---")
            
            # Generate environment-specific validation prompts (9 to match Experiment 3)
            env_validation_prompts = self.generate_environment_validation_prompts(trait, env, n_prompts=9)
            
            # Generate baseline scores with environment vector
            baseline_scores = []
            for prompt in env_validation_prompts:
                completion = self.experiment.generate_with_steering(
                    prompt, v, alpha=1.0, max_new_tokens=150
                )[0]
                score = self.experiment.compute_semantic_score(completion, trait)
                baseline_scores.append(score)
            
            baseline_scores = np.array(baseline_scores)
            
            # Test orthogonal components
            effect_sizes = []
            correlations = []
            
            for seed in range(n_orthogonal):
                # Generate orthogonal component
                torch.manual_seed(seed + 42)
                random_vec = torch.randn_like(v)
                v_perp = random_vec - torch.dot(random_vec, v) / torch.dot(v, v) * v
                v_perp = v_perp / torch.norm(v_perp) * torch.norm(v)
                
                # Test v + v⊥
                v_combined = v + v_perp
                perp_scores = []
                
                for prompt in env_validation_prompts:
                    completion = self.experiment.generate_with_steering(
                        prompt, v_combined, alpha=1.0, max_new_tokens=150
                    )[0]
                    score = self.experiment.compute_semantic_score(completion, trait)
                    perp_scores.append(score)
                
                perp_scores = np.array(perp_scores)
                
                # Compute metrics
                pooled_std = np.sqrt(((len(baseline_scores)-1) * np.var(baseline_scores, ddof=1) + 
                                    (len(perp_scores)-1) * np.var(perp_scores, ddof=1)) / 
                                   (len(baseline_scores) + len(perp_scores) - 2))
                effect_size = np.abs(np.mean(baseline_scores) - np.mean(perp_scores)) / pooled_std
                effect_sizes.append(effect_size)
                
                # Check variance before computing correlation to avoid NaN from zero-variance arrays
                baseline_var = np.var(baseline_scores, ddof=1)
                perp_var = np.var(perp_scores, ddof=1)
                if baseline_var < 1e-10 or perp_var < 1e-10:
                    correlation = np.nan
                else:
                    correlation = np.corrcoef(baseline_scores, perp_scores)[0,1]
                correlations.append(correlation)
            
            results['cross_environment_tests'][env] = {
                'mean_effect_size': np.nanmean(effect_sizes),
                'std_effect_size': np.nanstd(effect_sizes),
                'mean_correlation': np.nanmean(correlations),
                'std_correlation': np.nanstd(correlations),
                'effect_sizes': effect_sizes,
                'correlations': correlations,
                'baseline_scores': baseline_scores.tolist()
            }
            
            print(f"  {env}: Effect size = {np.nanmean(effect_sizes):.4f} ± {np.nanstd(effect_sizes):.4f}")
            print(f"  {env}: Correlation = {np.nanmean(correlations):.4f} ± {np.nanstd(correlations):.4f}")
        
        # Note: environment vector orthogonality test not applicable since we test the same v across environments
        # (we extract v only from in_distribution, not multiple v's per environment)
        
        return results
    
    def analyze_environment_robustness(self, traits: List[str], environments: List[str]) -> Dict:
        """
        Analyze the robustness of identifiability patterns across environments.
        
        Args:
            traits: List of traits to analyze
            environments: List of environments to test
            
        Returns:
            Dictionary with robustness analysis
        """
        print(f"\n=== Analyzing Environment Robustness ===")
        
        robustness_results = {
            'traits': traits,
            'environments': environments,
            'cross_trait_consistency': {},
            'environment_stability': {}
        }
        
        # Collect effect sizes across traits and environments
        all_effect_sizes = {}
        all_correlations = {}
        
        for trait in traits:
            trait_results = self.test_cross_environment_equivalence(trait, environments)
            
            all_effect_sizes[trait] = {}
            all_correlations[trait] = {}
            
            for env in environments:
                all_effect_sizes[trait][env] = trait_results['cross_environment_tests'][env]['mean_effect_size']
                all_correlations[trait][env] = trait_results['cross_environment_tests'][env]['mean_correlation']
        
        # Analyze cross-trait consistency
        for env in environments:
            env_effect_sizes = [all_effect_sizes[trait][env] for trait in traits]
            env_correlations = [all_correlations[trait][env] for trait in traits]
            
            robustness_results['cross_trait_consistency'][env] = {
                'effect_size_variance': np.nanvar(env_effect_sizes),
                'correlation_variance': np.nanvar(env_correlations),
                'mean_effect_size': np.nanmean(env_effect_sizes),
                'mean_correlation': np.nanmean(env_correlations),
                'effect_sizes': env_effect_sizes,
                'correlations': env_correlations
            }
        
        # Analyze environment stability (across environments for each trait)
        for trait in traits:
            trait_effect_sizes = [all_effect_sizes[trait][env] for env in environments]
            trait_correlations = [all_correlations[trait][env] for env in environments]
            
            robustness_results['environment_stability'][trait] = {
                'effect_size_variance': np.nanvar(trait_effect_sizes),
                'correlation_variance': np.nanvar(trait_correlations),
                'mean_effect_size': np.nanmean(trait_effect_sizes),
                'mean_correlation': np.nanmean(trait_correlations),
                'effect_sizes': trait_effect_sizes,
                'correlations': trait_correlations
            }
        
        return robustness_results
    
    def run_full_validation(self, traits: List[str], environments: List[str], 
                          n_orthogonal: int = 5) -> Dict:
        """
        Run complete multi-environment validation.
        
        Args:
            traits: List of traits to test
            environments: List of environments to validate across
            n_orthogonal: Number of orthogonal tests per environment
            
        Returns:
            Complete validation results
        """
        print(f"=== Multi-Environment Validation ===")
        print(f"Model: {self.model_name}")
        print(f"Traits: {traits}")
        print(f"Environments: {environments}")
        print(f"Orthogonal tests per environment: {n_orthogonal}\n")
        
        results = {
            'model_name': self.model_name,
            'traits': traits,
            'environments': environments,
            'n_orthogonal': n_orthogonal,
            'environment_descriptions': {env: self.environments[env]['name'] + ': ' + 
                                       self.environments[env]['description'] 
                                       for env in environments},
            'trait_results': {},
            'robustness_analysis': None,
            'summary': {}
        }
        
        # Test each trait across environments
        for trait in traits:
            print(f"\n{'='*50}")
            print(f"TRAIT: {trait.upper()}")
            print(f"{'='*50}")
            
            trait_results = self.test_cross_environment_equivalence(
                trait, environments, n_orthogonal
            )
            results['trait_results'][trait] = trait_results
        
        # Run robustness analysis  
        robustness_results = self.analyze_environment_robustness(traits, environments)
        results['robustness_analysis'] = robustness_results
        
        # Compute summary
        self._compute_summary(results)
        
        return results
    
    def _compute_summary(self, results: Dict):
        """Compute summary statistics and interpretation."""
        print(f"\n{'='*50}")
        print("SUMMARY ANALYSIS")
        print(f"{'='*50}")
        
        summary = {
            'global_patterns': {},
            'environment_effects': {}
        }
        
        # Global patterns across all traits and environments
        all_effect_sizes = []
        all_correlations = []
        
        for trait in results['traits']:
            for env in results['environments']:
                test_results = results['trait_results'][trait]['cross_environment_tests'][env]
                all_effect_sizes.append(test_results['mean_effect_size'])
                all_correlations.append(test_results['mean_correlation'])
        
        summary['global_patterns'] = {
            'overall_mean_effect_size': np.nanmean(all_effect_sizes),
            'overall_std_effect_size': np.nanstd(all_effect_sizes),
            'overall_mean_correlation': np.nanmean(all_correlations),
            'overall_std_correlation': np.nanstd(all_correlations),
            'proportion_small_effects': np.nanmean(np.array(all_effect_sizes) < 0.3),
            'proportion_high_correlations': np.nanmean(np.array(all_correlations) > 0.6)
        }
        
        # Environment-specific effects
        for env in results['environments']:
            env_effect_sizes = []
            env_correlations = []
            
            for trait in results['traits']:
                test_results = results['trait_results'][trait]['cross_environment_tests'][env]
                env_effect_sizes.append(test_results['mean_effect_size'])
                env_correlations.append(test_results['mean_correlation'])
            
            summary['environment_effects'][env] = {
                'mean_effect_size': np.nanmean(env_effect_sizes),
                'mean_correlation': np.nanmean(env_correlations),
                'effect_size_consistency': np.nanstd(env_effect_sizes)  # Lower = more consistent
            }
        
        
        results['summary'] = summary
        
        # Print key findings
        print(f"\nKey Findings:")
        print(f"  Overall effect size: {summary['global_patterns']['overall_mean_effect_size']:.4f} ± {summary['global_patterns']['overall_std_effect_size']:.4f}")
        print(f"  Overall correlation: {summary['global_patterns']['overall_mean_correlation']:.4f} ± {summary['global_patterns']['overall_std_correlation']:.4f}")
        print(f"  Proportion with small effects: {summary['global_patterns']['proportion_small_effects']:.2%}")
        print(f"  Proportion with high correlations: {summary['global_patterns']['proportion_high_correlations']:.2%}")
        
    def save_results(self, results: Dict, output_dir: str = "."):
        """Save results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        results = self._convert_to_json_serializable(results)
        
        # Save raw results
        model_short = results['model_name'].split('/')[-1]
        trait_names = '_'.join(sorted(results['traits']))
        filename = f"multi_environment_validation_{trait_names}_{model_short}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filepath}")
    
    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            val = float(obj) if isinstance(obj, np.floating) else int(obj)
            # Handle NaN and Inf
            if np.isnan(val):
                return None  # JSON doesn't support NaN, use null
            elif np.isinf(val):
                return float(val)  # Keep inf as is
            return val
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return self._convert_to_json_serializable(obj.tolist())
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description="Multi-environment validation experiment")
    parser.add_argument("--model", type=str,
                       default="microsoft/DialoGPT-medium", 
                       help="HuggingFace model name")
    parser.add_argument("--traits", nargs="+",
                       default=["formality", "politeness", "sentiment"],
                       help="Traits to test (use 'all' for all traits). Available: formality, politeness, sentiment, truthfulness, agreeableness")
    parser.add_argument("--environments", nargs="+",
                       default=["in_distribution", "topic_shift", "genre_shift"],
                       help="Environments to test (default: in_distribution topic_shift genre_shift)")
    parser.add_argument("--n_orthogonal", type=int, default=5,
                       help="Number of orthogonal tests per environment")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: src/results)")
    
    args = parser.parse_args()
    
    # Handle "all" traits
    if args.traits == ["all"]:
        traits = ["formality", "politeness", "sentiment", "truthfulness", "agreeableness"]
    else:
        traits = args.traits
    
    # Handle "all" environments
    validator = MultiEnvironmentValidation(args.model)
    environments = args.environments
    # Validate environment names
    invalid_envs = [env for env in environments if env not in validator.environments]
    if invalid_envs:
        print(f"Error: Invalid environments: {invalid_envs}")
        print(f"Available environments: {list(validator.environments.keys())}")
        return
    
    # Set output directory to src/results if not specified
    if args.output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.output_dir = os.path.join(project_root, 'src', 'results')
    
    print(f"Starting multi-environment validation...")
    print(f"Model: {args.model}")
    print(f"Traits: {traits}")
    print(f"Environments: {environments}")
    print(f"Output directory: {args.output_dir}")
    
    # Run validation
    results = validator.run_full_validation(traits, environments, args.n_orthogonal)
    
    # Save results
    validator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()