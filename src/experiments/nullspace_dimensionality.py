"""
Experiment A1 — Direct Null-Space Dimensionality Measurement

Measures the effective dimensionality of the null space (ker(J)) for models.
Converts theoretical claims into empirical measurements by:
1. Computing approximate Jacobian using finite differences
2. Performing SVD to find effective rank
3. Calculating null-space fraction (NF)

HOW TO RUN:
-----------
    cd src/experiments
    python nullspace_dimensionality.py

Results saved to: ../results/nullspace/ (singular value spectra PNGs + JSON files)
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


class NullSpaceDimensionalityExperiment:
    """
    Directly measures the effective dimensionality of the null space.
    
    For each model at layer positions (L/4, L/2, 3L/4):
    - Approximate Jacobian using finite differences (500 probe directions)
    - Compute SVD and effective rank (singular values > 1% of sigma_max)
    - Calculate null-space fraction NF(l)
    """
    
    def __init__(self, model_name: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the nullspace dimensionality experiment.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (default: 'cuda:0' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use device_map with specific GPU to avoid multi-GPU splitting
        if device.startswith('cuda'):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Get model dimensions
        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = len(self.model.model.layers)
        
        # Finite difference parameters
        self.epsilon = 0.01
        self.d_probe = 1000  # Number of random probe directions (increased to 1000 for better null space coverage)
        
        # Effective rank threshold (1% of max singular value)
        self.threshold_percentile = 0.01
        
    def get_output(self, prompt: str) -> torch.Tensor:
        """
        Forward pass and get final logits.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Logits tensor of shape (vocab_size,)
        """
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        return logits
    
    def compute_jacobian_finite_diff(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Compute approximate Jacobian using finite differences.
        
        For each of d_probe random unit directions e_i:
        J_approx[:,i] = (o(h + ε·e_i) − o(h − ε·e_i)) / 2ε
        
        Args:
            prompt: Input prompt
            layer_idx: Layer to perturb (at its output)
            
        Returns:
            Jacobian approximation of shape (vocab_size, d_probe)
        """
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        J_approx = []
        
        print(f"  Computing Jacobian with {self.d_probe} probe directions...")
        for i in tqdm(range(self.d_probe), desc="Probe directions", leave=False):
            # Random unit direction - create on CPU then move to device
            e_i = torch.randn(self.hidden_dim, dtype=torch.float32)
            e_i = e_i / (torch.norm(e_i) + 1e-8)
            e_i = e_i.to(device=self.device, dtype=torch.float16)
            
            # Perturbed forward pass: h + ε·e_i
            logits_plus = self._forward_with_perturbation(inputs, layer_idx, e_i, self.epsilon)
            
            # Perturbed forward pass: h - ε·e_i
            logits_minus = self._forward_with_perturbation(inputs, layer_idx, e_i, -self.epsilon)
            
            # Finite difference: (logits+ - logits-) / 2ε
            jacobian_col = (logits_plus - logits_minus) / (2 * self.epsilon)
            J_approx.append(jacobian_col)
            
            # Clear CUDA cache periodically
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        J_approx = torch.stack(J_approx, dim=1)  # Shape: (vocab_size, d_probe)
        return J_approx
    
    def _forward_with_perturbation(self, inputs: Dict, layer_idx: int, 
                                   direction: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Forward pass with perturbation applied at layer output.
        
        Args:
            inputs: Tokenized inputs
            layer_idx: Layer index to perturb
            direction: Direction vector to perturb in
            scale: Scaling factor
            
        Returns:
            Logits from perturbed forward pass
        """
        perturbed_hidden_states = {}
        
        def hook_fn(module, input, output):
            """Hook to capture and perturb hidden states."""
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()
            
            # Store the hidden state before perturbation
            perturbed_hidden_states['hidden'] = hidden_states.clone()
            
            # Perturb the last token's hidden state
            hidden_states = hidden_states.clone()
            
            # Ensure direction is on the same device and dtype as hidden_states
            perturbation = (scale * direction).to(
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            )
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + perturbation
            
            if rest:
                return (hidden_states,) + rest
            else:
                return hidden_states
        
        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits[0, -1, :]
        finally:
            hook_handle.remove()
        
        return logits
    
    def compute_effective_rank(self, jacobian: torch.Tensor) -> Tuple[np.ndarray, int, float]:
        """
        Compute effective rank using 5% threshold.
        
        Effective rank = number of singular values > 5% of sigma_max.
        
        Args:
            jacobian: Jacobian approximation (vocab_size, d_probe)
            
        Returns:
            Tuple of (singular_values, effective_rank, null_space_fraction)
        """
        # Move to CPU for SVD computation
        J_cpu = jacobian.cpu().detach().numpy().astype(np.float32)
        
        # SVD
        U, S, Vt = np.linalg.svd(J_cpu, full_matrices=False)
        
        # Use 5% threshold
        threshold = 0.05 * S[0]
        effective_rank = np.sum(S > threshold)
        
        # Null-space fraction
        null_space_fraction = (self.d_probe - effective_rank) / self.d_probe
        
        # Debug info
        print(f"      Singular values (first 10): {S[:10]}")
        print(f"      Singular values (last 10): {S[-10:]}")
        print(f"      Threshold (5%): {threshold:.3f}")
        print(f"      Min singular value: {S[-1]:.6e}")
        print(f"      Max singular value: {S[0]:.6f}")
        
        return S, effective_rank, null_space_fraction
    
    def run_experiment(self, prompts: List[str], layer_positions: List[str] = None) -> Dict:
        """
        Run the null-space dimensionality measurement experiment.
        
        Args:
            prompts: List of input prompts
            layer_positions: Which layers to measure ['L/4', 'L/2', '3L/4']
                           Default: all three cardinal positions
            
        Returns:
            Dictionary with results per layer
        """
        if layer_positions is None:
            layer_positions = ['L/4', 'L/2', '3L/4']
        
        results = {
            'model': self.model_name,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'd_probe': self.d_probe,
            'epsilon': self.epsilon,
            'by_layer': {}
        }
        
        # Map position names to layer indices
        position_to_idx = {
            'L/4': self.num_layers // 4,
            'L/2': self.num_layers // 2,
            '3L/4': 3 * self.num_layers // 4,
        }
        
        for position in layer_positions:
            print(f"\nProcessing layer position: {position}")
            layer_idx = position_to_idx[position]
            
            # Use first prompt for dimensionality measurement
            prompt = prompts[0] if prompts else "Hello, this is a test prompt."
            
            # Compute Jacobian
            print(f"  Computing Jacobian at layer {layer_idx}/{self.num_layers}")
            jacobian = self.compute_jacobian_finite_diff(prompt, layer_idx)
            
            # Compute effective rank and null-space fraction
            print(f"  Computing SVD and effective rank...")
            singular_values, eff_rank, nf = self.compute_effective_rank(jacobian)
            
            # Store results
            results['by_layer'][position] = {
                'layer_index': layer_idx,
                'effective_rank': int(eff_rank),
                'null_space_fraction': float(nf),
                'singular_values': singular_values.tolist(),
                'num_singular_values': len(singular_values),
                'threshold': float(self.threshold_percentile * singular_values[0]),
                'max_singular_value': float(singular_values[0]),
            }
            
            print(f"    Effective rank: {eff_rank}/{self.d_probe}")
            print(f"    Null-space fraction: {nf:.4f}")
        
        return results
    
    def visualize_singular_values(self, results: Dict, output_dir: str = "../results/nullspace"):
        """
        Create visualization of singular value spectra.
        
        Args:
            results: Results dictionary from run_experiment
            output_dir: Directory to save figures
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        positions = ['L/4', 'L/2', '3L/4']
        for ax, position in zip(axes, positions):
            if position in results['by_layer']:
                layer_data = results['by_layer'][position]
                singular_values = np.array(layer_data['singular_values'])
                threshold = layer_data['threshold']
                
                # Plot singular values
                ax.semilogy(range(len(singular_values)), singular_values, 'b-', linewidth=2)
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (1%)')
                
                ax.set_xlabel('Index')
                ax.set_ylabel('Singular Value (log scale)')
                ax.set_title(f'{position} (Eff. Rank: {layer_data["effective_rank"]})')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        model_short = self.model_name.split('/')[-1]
        fig.suptitle(f'Singular Value Spectrum - {model_short}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'nullspace_spectrum_{model_short}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
        plt.close()
    
    def save_results(self, results: Dict, output_dir: str = "../results/nullspace"):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model_short = self.model_name.split('/')[-1]
        output_path = os.path.join(output_dir, f'nullspace_dimensionality_{model_short}.json')
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


def main():
    """
    Example: Run Experiment A1 on two models with simple prompts.
    """
    models = [
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    
    # Simple test prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "Tell me a joke about artificial intelligence."
    ]
    
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        try:
            experiment = NullSpaceDimensionalityExperiment(model_name)
            
            # Run experiment
            results = experiment.run_experiment(prompts)
            all_results[model_name] = results
            
            # Visualize and save
            experiment.visualize_singular_values(results)
            experiment.save_results(results, output_dir="../results/nullspace")
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        torch.cuda.empty_cache()
    
    # Save combined results
    with open("../results/nullspace/nullspace_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
