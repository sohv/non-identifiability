#!/usr/bin/env python3
"""
Figure Generation from Experiment Results

Generate plots from JSON experiment outputs. Dynamically discovers and loads 
result files from src/results directory.

Usage:
    python generate_figures.py --experiment orthogonal
    python generate_figures.py --experiment alpha_sweep
    python generate_figures.py --experiment multi_environment
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


class FigureGenerator:
    """Figure generator that loads data from JSON files and generates publication-ready plots"""
    
    def __init__(self):
        """Initialize generator with style config and discover result files"""
        self.root = Path(__file__).parent.parent.parent
        
        config_path = self.root / "config" / "style.yaml"
        with open(config_path) as f:
            self.style = yaml.safe_load(f)
        plt.rcParams.update(self.style['fonts']['rcParams'])
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = self.style['fonts']['sans_serif_fonts']
        
        self.model_colors = {
            'Qwen': '#0066FF',
            'Llama': '#FF3333'
        }
        
        self.results_dir = self.root / "src" / "results"
        self.results_dir = self.root / "src" / "results"
        self.figures_dir = self.root / "src" / "results" / "figures"
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Dynamically discover JSON files
        self._discover_result_files()
    
    def _discover_result_files(self):
        """Discover available JSON result files using glob patterns"""
        self.json_files = {}
        
        # Find all JSON files in results directory (non-recursive from root)
        all_jsons = list(self.results_dir.glob("*.json"))
        
        # Categorize by experiment type
        for json_file in all_jsons:
            name = json_file.stem
            if 'orthogonal' in name:
                if 'orthogonal' not in self.json_files:
                    self.json_files['orthogonal'] = []
                self.json_files['orthogonal'].append(json_file)
            elif 'alpha_sweep' in name:
                if 'alpha_sweep' not in self.json_files:
                    self.json_files['alpha_sweep'] = []
                self.json_files['alpha_sweep'].append(json_file)
            elif 'multi_environment' in name:
                if 'multi_environment' not in self.json_files:
                    self.json_files['multi_environment'] = []
                self.json_files['multi_environment'].append(json_file)
            elif 'logit_distance' in name:
                if 'logit_distance' not in self.json_files:
                    self.json_files['logit_distance'] = []
                self.json_files['logit_distance'].append(json_file)
            elif 'vector_equivalence' in name:
                if 'vector_equivalence' not in self.json_files:
                    self.json_files['vector_equivalence'] = []
                self.json_files['vector_equivalence'].append(json_file)
            elif 'nullspace_spanning' in name:
                if 'nullspace_spanning' not in self.json_files:
                    self.json_files['nullspace_spanning'] = []
                self.json_files['nullspace_spanning'].append(json_file)
            elif 'nullspace_dimensionality' in name:
                if 'nullspace_dimensionality' not in self.json_files:
                    self.json_files['nullspace_dimensionality'] = []
                self.json_files['nullspace_dimensionality'].append(json_file)
    
    def _load_json_data(self, experiment):
        """Load and merge JSON data from result files"""
        data = {}
        for json_file in self.json_files.get(experiment, []):
            try:
                with open(json_file) as f:
                    file_data = json.load(f)
                    data.update(file_data)
                print(f"Loaded: {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
        return data
    
    def _save_figure(self, base_name):
        """Save figure in both PNG and PDF formats"""
        png_path = self.figures_dir / f"{base_name}.png"
        pdf_path = self.figures_dir / f"{base_name}.pdf"
        
        plt.savefig(png_path, dpi=self.style['paper']['dpi'], bbox_inches='tight', pad_inches=0.05)
        print(f" Saved: {png_path}")
        
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05, format='pdf')
        print(f" Saved: {pdf_path}")
        
        plt.close()
    
    def _apply_plot_style(self, ax, plot_type):
        """Apply consistent styling from style.yaml to a plot axis"""
        if plot_type not in self.style['plot_types']:
            return
        
        config = self.style['plot_types'][plot_type]
        
        # Apply spines
        if 'spines' in config:
            ax.spines['left'].set_visible(config['spines'].get('left', True))
            ax.spines['bottom'].set_visible(config['spines'].get('bottom', True))
            ax.spines['right'].set_visible(config['spines'].get('right', False))
            ax.spines['top'].set_visible(config['spines'].get('top', False))
        
        # Apply spine width
        if 'spine_width' in config:
            for spine in ax.spines.values():
                spine.set_linewidth(config['spine_width'])
        
        # Apply grid
        if 'grid' in config and config['grid'].get('enabled', False):
            grid_cfg = config['grid']
            ax.grid(True, 
                   axis=grid_cfg.get('axis', 'y'),
                   linestyle=grid_cfg.get('linestyle', ':'),
                   alpha=grid_cfg.get('alpha', 0.2),
                   linewidth=grid_cfg.get('linewidth', 0.5))
            ax.set_axisbelow(True)
    
    def run(self, experiment):
        """Main entry point - route to appropriate plot function"""
        if experiment not in self.json_files or not self.json_files[experiment]:
            print(f"No data files found for experiment: {experiment}")
            print(f"Available experiments: {', '.join([k for k, v in self.json_files.items() if v])}")
            return False
        
        print(f"\n{'='*80}")
        print(f"Generating Figure: {experiment}")
        print(f"Found {len(self.json_files[experiment])} JSON file(s)")
        print(f"{'='*80}\n")
        
        # Route based on experiment
        if experiment == 'orthogonal':
            self._plot_orthogonal_test()
        elif experiment == 'alpha_sweep':
            self._plot_alpha_sweep()
        elif experiment == 'multi_environment':
            self._plot_multi_environment_heatmap()
        elif experiment == 'logit_distance':
            self._plot_logit_distance()
        elif experiment == 'vector_equivalence':
            self._plot_vector_equivalence()
        elif experiment == 'nullspace_spanning':
            self._plot_nullspace_spanning()
        elif experiment == 'nullspace_dimensionality':
            self._plot_nullspace_dimensionality()
        
        return True
    
    # ========================================================================
    # FIGURE 1: ORTHOGONAL TEST (Cohen's d by trait)
    # ========================================================================
    def _plot_orthogonal_test(self):
        """Bar plot of Cohen's d across traits - loaded from JSON files"""
        
        # Load data from JSON files
        data = self._load_json_data('orthogonal')
        if not data:
            print("No orthogonal test data found")
            return
        
        # Extract trait names
        traits = [t.capitalize() for t in list(data.keys())[:5]]
        
        # Organize data by model
        model_data = {}
        for trait_lower, trait_data in data.items():
            model_name = trait_data.get('model', '').split('/')[-1]
            if 'Llama' in model_name:
                model_short = 'Llama'
            elif 'Qwen' in model_name:
                model_short = 'Qwen'
            else:
                continue
            
            if model_short not in model_data:
                model_data[model_short] = {'means': [], 'stds': []}
            
            summary = trait_data.get('summary', {})
            mean_d = summary.get('mean_cohens_d', 0)
            std_d = summary.get('std_cohens_d', 0)
            
            model_data[model_short]['means'].append(mean_d)
            model_data[model_short]['stds'].append(std_d)
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(self.style['paper']['single_column']['width'], 
                     self.style['paper']['single_column']['height']),
            dpi=self.style['paper']['dpi']
        )
        
        x = np.arange(len(traits))
        width = 0.35
        
        # Plot bars for each model
        colors = [self.model_colors.get('Llama', '#FF3333'), 
                  self.model_colors.get('Qwen', '#0066FF')]
        
        for idx, (model, model_values) in enumerate(sorted(model_data.items())):
            offset = (idx - len(model_data)/2 + 0.5) * width
            ax.bar(x + offset, model_values['means'], width, 
                   label=model, color=colors[idx], alpha=0.8,
                   yerr=model_values['stds'],
                   capsize=5, error_kw={'linewidth': 1.0})
        
        # Apply styling
        self._apply_plot_style(ax, 'bar_plots')
        ax.set_ylabel("Cohen's d", fontsize=self.style['fonts']['axis_label'])
        ax.set_xticks(x)
        ax.set_xticklabels(traits, 
                           fontsize=self.style['fonts']['tick_label'], 
                           rotation=45, ha='right')
        
        if model_data:
            ax.legend(fontsize=self.style['fonts']['legend'], loc='upper center',
                     bbox_to_anchor=(0.5, -0.4), ncol=len(model_data), framealpha=0.95)
        
        plt.tight_layout()
        self._save_figure("figure_orthogonal_test")
    
    # ========================================================================
    # FIGURE 2: MULTI-ENVIRONMENT HEATMAP
    # ========================================================================
    def _plot_multi_environment_heatmap(self):
        """Heatmap of Cohen's d across environments and traits - loaded from JSON"""
        
        # Load all JSON files (multi_environment)  
        traits_order = ['agreeableness', 'formality', 'politeness', 'sentiment', 'truthfulness']
        environments = ['in_distribution', 'topic_shift', 'genre_shift']
        heatmap_data = {}
        
        for json_file in self.json_files.get('multi_environment', []):
            with open(json_file) as f:
                data = json.load(f)
            
            model_name = data.get('model_name', '')
            model_short = 'Llama' if 'Llama' in model_name else 'Qwen' if 'Qwen' in model_name else None
            
            if not model_short:
                continue
            
            # Create matrix for this model
            matrix = np.zeros((len(traits_order), len(environments)))
            trait_results = data.get('trait_results', {})
            
            # Fill matrix with mean_effect_size values
            for trait_idx, trait in enumerate(traits_order):
                if trait in trait_results:
                    cross_env = trait_results[trait].get('cross_environment_tests', {})
                    for env_idx, env in enumerate(environments):
                        if env in cross_env:
                            mean_effect = cross_env[env].get('mean_effect_size', 0)
                            matrix[trait_idx, env_idx] = mean_effect
            
            heatmap_data[model_short] = matrix
        
        if not heatmap_data:
            print("Could not extract heatmap data from JSON")
            return
        
        traits = [t.capitalize() for t in traits_order]
        env_display = [e.replace('_', ' ').title() for e in environments]
        
        # Create figure
        fig, axes = plt.subplots(
            1, 2,
            figsize=(self.style['paper']['full_page']['width'], 
                     self.style['paper']['single_column']['height']),
            dpi=self.style['paper']['dpi']
        )
        
        hm_config = self.style['plot_types']['heatmaps']
        
        for idx, (model_short, hm_data) in enumerate(sorted(heatmap_data.items())):
            ax = axes[idx]
            im = ax.imshow(hm_data, cmap=hm_config['cmap'], aspect=hm_config['aspect'], 
                          vmin=hm_config['vmin'], vmax=hm_config['vmax'])
            ax.set_xticks(np.arange(len(env_display)))
            ax.set_yticks(np.arange(len(traits)))
            ax.set_xticklabels(env_display, fontsize=self.style['fonts']['tick_label'])
            ax.set_yticklabels(traits if idx == 0 else [], fontsize=self.style['fonts']['tick_label'])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add values
            for i in range(len(traits)):
                for j in range(len(env_display)):
                    ax.text(j, i, f'{hm_data[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
            
            # Add subtitle
            ax.text(0.5, 1.08, model_short, fontsize=8, ha='center', transform=ax.transAxes)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=axes[1], label='Effect Size', pad=0.02, orientation='vertical')
        cbar.ax.tick_params(labelsize=self.style['fonts']['tick_label'])
        cbar.set_label('Effect Size', fontsize=self.style['fonts']['axis_label'])
        
        plt.tight_layout()
        self._save_figure("figure_multi_environment")
    
    # ========================================================================
    # FIGURE 3: LOGIT DISTANCE COMPARISON (Two models side-by-side)
    # ========================================================================
    
    def _create_logit_distance_subplot(self, ax, model_name, baseline, vperp, random, 
                                       include_ylabel=True, show_legend=True, legend_loc='upper left',
                                       legend_below=False, compact_legend=False):
        """Helper: Create logit distance bars for a single model on given axis.
        
        Args:
            ax: matplotlib axis to plot on
            model_name: 'Llama' or 'Qwen'
            baseline, vperp, random: Lists of 5 values (one per trait)
            include_ylabel: Whether to include y-axis label
            show_legend: Whether to show legend
            legend_loc: Legend location (e.g., 'upper left', 'upper right')
            legend_below: If True, place legend horizontally below plot with 3 columns
            compact_legend: If True, use smaller fontsize and dimmed appearance for individual plots
        """
        traits = ['Sentiment', 'Truthfulness', 'Politeness', 'Agreeableness', 'Formality']
        x = np.arange(5)
        width = 0.25
        
        # Choose color based on model
        bar_color = '#1f77b4' if model_name == 'Llama' else '#d62728'
        
        # Create bars
        ax.bar(x - width, baseline, width, label='Baseline',
               color=bar_color, alpha=0.9)
        ax.bar(x, vperp, width, label=r'$v_\perp$',
               color=bar_color, alpha=0.6)
        ax.bar(x + width, random, width, label='Random',
               color=bar_color, alpha=0.3, hatch='//')
        
        # Add model label at top
        ax.text(2, 175, model_name, fontsize=9, ha='center')
        
        # Y-axis label
        if include_ylabel:
            ax.set_ylabel('L2 Logit Distance', fontsize=self.style['fonts']['axis_label'] + 1, weight='normal')
        
        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(traits, fontsize=self.style['fonts']['tick_label'] + 1, rotation=45, ha='right')
        ax.set_ylim([0, 180])
        
        # Grid and spines
        ax.grid(True, axis='y', alpha=0.2, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Legend
        if show_legend:
            if legend_below:
                # Horizontal legend below plot with 3 columns
                ax.legend(fontsize=6.5, loc='upper center', ncol=3, 
                         bbox_to_anchor=(0.5, -0.22), framealpha=0.95, 
                         frameon=True, fancybox=False, edgecolor='gray')
            elif compact_legend:
                # Compact legend for individual plots: small and dimmed
                ax.legend(fontsize=5, loc='upper right', framealpha=0.7, 
                         frameon=True, fancybox=False, edgecolor='lightgray')
            else:
                # Standard legend for unified plots: original styling
                ax.legend(fontsize=7, loc=legend_loc, framealpha=0.95)
    

    def _save_individual_logit_distance_plots(self):
        """Generate individual plots for Llama and Qwen separately"""
        
        data = self._load_json_data('logit_distance')
        if not data:
            print("No logit distance data found")
            return
        
        traits_order = ['sentiment', 'truthfulness', 'politeness', 'agreeableness', 'formality']
        model_distances = {}
        
        for model_key, model_data in data.items():
            model_short = 'Llama' if 'Llama' in model_key else 'Qwen' if 'Qwen' in model_key else None
            
            if not model_short:
                continue
            
            if model_short not in model_distances:
                model_distances[model_short] = {'baseline': [], 'v_perp': [], 'random': []}
            
            for trait in traits_order:
                if trait in model_data:
                    trait_data = model_data[trait]
                    distances = trait_data.get('distances', {})
                    baseline = distances.get('baseline', {}).get('mean', 0)
                    v_vs_vperp = distances.get('v_vs_vperp', {}).get('mean', 0)
                    v_vs_random = distances.get('v_vs_random', {}).get('mean', 0)
                    
                    model_distances[model_short]['baseline'].append(baseline)
                    model_distances[model_short]['v_perp'].append(v_vs_vperp)
                    model_distances[model_short]['random'].append(v_vs_random)
        
        for model_short in sorted(model_distances.keys()):
            fig, ax = plt.subplots(
                figsize=(self.style['paper']['single_column']['width'], 
                         self.style['paper']['single_column']['height']),
                dpi=self.style['paper']['dpi']
            )
            self._create_logit_distance_subplot(
                ax, model_short,
                model_distances[model_short]['baseline'],
                model_distances[model_short]['v_perp'],
                model_distances[model_short]['random'],
                compact_legend=True
            )
            plt.tight_layout()
            
            base_name = f"figure_logit_distance_{model_short.lower()}"
            png_path = self.figures_dir / f"{base_name}.png"
            pdf_path = self.figures_dir / f"{base_name}.pdf"
            
            plt.savefig(png_path, dpi=self.style['paper']['dpi'], bbox_inches='tight', pad_inches=0.05)
            print(f"Saved: {png_path}")
            
            plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05, format='pdf')
            print(f"Saved: {pdf_path}")
            
            plt.close()
    
    def _plot_logit_distance(self):
        """Two subplots (one per model) with baseline, v⊥, and random bars"""
        
        # Load data from JSON
        data = self._load_json_data('logit_distance')
        if not data:
            print("No logit distance data found")
            return
        
        # Extract baseline, v_vs_vperp, and v_vs_random distances by model and trait
        traits_order = ['sentiment', 'truthfulness', 'politeness', 'agreeableness', 'formality']
        model_distances = {}
        
        # Iterate through models in data
        for model_key, model_data in data.items():
            model_short = 'Llama' if 'Llama' in model_key else 'Qwen' if 'Qwen' in model_key else None
            
            if not model_short:
                continue
            
            if model_short not in model_distances:
                model_distances[model_short] = {'baseline': [], 'v_perp': [], 'random': []}
            
            # Extract distances for each trait
            for trait in traits_order:
                if trait in model_data:
                    trait_data = model_data[trait]
                    distances = trait_data.get('distances', {})
                    baseline = distances.get('baseline', {}).get('mean', 0)
                    v_vs_vperp = distances.get('v_vs_vperp', {}).get('mean', 0)
                    v_vs_random = distances.get('v_vs_random', {}).get('mean', 0)
                    
                    model_distances[model_short]['baseline'].append(baseline)
                    model_distances[model_short]['v_perp'].append(v_vs_vperp)
                    model_distances[model_short]['random'].append(v_vs_random)
        
        if not model_distances:
            print("Could not extract logit distance data")
            return
        
        # Create figure with two subplots
        fig, axes = plt.subplots(
            1, 2,
            figsize=(10, self.style['paper']['single_column']['height']),
            dpi=self.style['paper']['dpi']
        )
        
        for idx, (model_short, distances) in enumerate(sorted(model_distances.items())):
            ax = axes[idx]
            self._create_logit_distance_subplot(
                ax, model_short,
                distances['baseline'],
                distances['v_perp'],
                distances['random'],
                include_ylabel=(idx == 0),
                show_legend=True
            )
        
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        self._save_figure("figure_logit_distance")
        
        self._save_individual_logit_distance_plots()
    
    # ========================================================================
    # FIGURE 4: ALPHA-SWEEP (Multi-trait steering strength)
    # ========================================================================
    def _plot_alpha_sweep(self):
        """Grid plot: steering scores v vs v+v_perp across alphas for all traits"""
        import json
        
        # Load data from alpha_sweep_multi_trait file
        alpha_file = self.results_dir / 'alpha_sweep_multi_trait_agreeableness_formality_politeness_sentiment_truthfulness.json'
        
        if not alpha_file.exists():
            print(f"File not found: {alpha_file}")
            return
        
        with open(alpha_file) as f:
            data = json.load(f)
        
        traits = ['Agreeableness', 'Formality', 'Politeness', 'Sentiment', 'Truthfulness']
        models = ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-3B-Instruct']
        model_names = ['Llama', 'Qwen']
        
        # Extract data
        results_by_model = {}
        for model_key, model_short in zip(models, model_names):
            results_by_model[model_short] = {}
            for trait in traits:
                trait_lower = trait.lower()
                trait_data = data[model_key][trait_lower]
                alphas = trait_data['alphas']
                
                means_v = []
                stds_v = []
                means_perp = []
                stds_perp = []
                
                for alpha_key in sorted(trait_data['alpha_results'].keys()):
                    alpha_res = trait_data['alpha_results'][alpha_key]
                    
                    scores_v = [s['mean_v'] for s in alpha_res['seeds']]
                    scores_perp = [s['mean_v_perp'] for s in alpha_res['seeds']]
                    
                    means_v.append(np.mean(scores_v))
                    stds_v.append(np.std(scores_v))
                    means_perp.append(np.mean(scores_perp))
                    stds_perp.append(np.std(scores_perp))
                
                results_by_model[model_short][trait] = {
                    'alphas': alphas,
                    'means_v': means_v,
                    'stds_v': stds_v,
                    'means_perp': means_perp,
                    'stds_perp': stds_perp,
                }
        
        # Create grid: 2 rows x 5 columns
        fig, axes = plt.subplots(
            2, 5,
            figsize=(14, 5),
            dpi=self.style['paper']['dpi']
        )
        
        for model_idx, (model_name, model_short) in enumerate(zip(models, model_names)):
            for trait_idx, trait in enumerate(traits):
                ax = axes[model_idx, trait_idx]
                
                data_dict = results_by_model[model_short][trait]
                alphas = np.array(data_dict['alphas'])
                means_v = np.array(data_dict['means_v'])
                stds_v = np.array(data_dict['stds_v'])
                means_perp = np.array(data_dict['means_perp'])
                stds_perp = np.array(data_dict['stds_perp'])
                
                # Plot v
                ax.plot(alphas, means_v, 'o-', linewidth=1.5, markersize=4,
                       label='v', color='#0066FF', alpha=0.95)
                ax.fill_between(alphas, 
                               means_v - stds_v,
                               means_v + stds_v,
                               alpha=0.35, color='#0066FF')
                
                # Plot v + v⟂ (perpendicular)
                ax.plot(alphas, means_perp, 's-', linewidth=1.5, markersize=4,
                       label=r'$v + v_\perp$', color='#FF9900', alpha=0.95)
                ax.fill_between(alphas,
                               means_perp - stds_perp,
                               means_perp + stds_perp,
                               alpha=0.35, color='#FF9900')
                
                # Formatting
                ax.set_xticks(alphas)
                ax.set_xticklabels([f'{x:.1f}' for x in alphas], fontsize=self.style['fonts']['tick_label'] + 2, color='black', family='monospace')
                ax.tick_params(axis='y', labelsize=self.style['fonts']['tick_label'] + 1, colors='black')
                plt.setp(ax.get_yticklabels(), family='monospace')
                ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
                
                # Title at top
                if model_idx == 0:
                    ax.set_title(trait, fontsize=self.style['fonts']['axis_label'] + 3)
                
                # Y-axis label on left
                if trait_idx == 0:
                    ax.set_ylabel(f'{model_short}', fontsize=self.style['fonts']['axis_label'] + 4)
                
                # X-axis label at bottom
                if model_idx == 1:
                    ax.set_xlabel('α', fontsize=self.style['fonts']['axis_label'] + 4)
        
        # Legend at bottom - centered, horizontal
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, 
                  bbox_to_anchor=(0.5, -0.02), fontsize=self.style['fonts']['legend'] + 4, framealpha=0.95)
        
        plt.subplots_adjust(hspace=0.35, wspace=0.35, bottom=0.15)
        
        # Save
        self._save_figure("figure_alpha_sweep")
    
    # ========================================================================
    # FIGURE 5: VECTOR EQUIVALENCE (from RESULTS.md)
    # ========================================================================
    def _plot_vector_equivalence(self):
        """Vector equivalence scatter plot - loaded from JSON"""
        
        # Load data from JSON files
        data = self._load_json_data('vector_equivalence')
        if not data:
            print("No vector equivalence data found")
            return
        
        # Extract scatter data organized by model
        model_data = {'Qwen': {'cosines': [], 'cohens': []},
                     'Llama': {'cosines': [], 'cohens': []}}
        
        # data is organized by model keys
        for model_key, model_traits in data.items():
            model_short = 'Llama' if 'Llama' in model_key else 'Qwen' if 'Qwen' in model_key else None
            
            if not model_short:
                continue
            
            # Extract data for each trait
            for trait_key, trait_data in model_traits.items():
                if isinstance(trait_data, dict):
                    cos_val = trait_data.get('cos_similarity', 0)
                    metrics = trait_data.get('metrics', {})
                    cohens_d = metrics.get('cohens_d', 0)
                    
                    model_data[model_short]['cosines'].append(cos_val)
                    model_data[model_short]['cohens'].append(cohens_d)
        
        if not any(model_data[m]['cosines'] for m in model_data):
            print("Could not extract vector equivalence data")
            return
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(self.style['paper']['single_column']['width'], 
                     self.style['paper']['single_column']['height']),
            dpi=self.style['paper']['dpi']
        )
        
        # Scatter plots for each model
        for model_short in ['Qwen', 'Llama']:
            if model_data[model_short]['cosines']:
                marker = 's' if model_short == 'Qwen' else '^'
                ax.scatter(model_data[model_short]['cosines'],
                          model_data[model_short]['cohens'],
                          s=100, alpha=0.7,
                          color=self.model_colors[model_short], label=model_short,
                          edgecolors='black', linewidth=1.0, marker=marker)
        
        # Add horizontal reference line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5, label='No Effect')
        
        # Apply plot styling from style.yaml
        self._apply_plot_style(ax, 'scatter_plots')
        
        # Formatting
        ax.set_xlabel('Cosine Similarity: cos(v₁, v₂)', fontsize=self.style['fonts']['axis_label'], weight='normal')
        ax.set_ylabel("Cohen's d (Effect Size)", fontsize=self.style['fonts']['axis_label'], weight='normal')
        ax.set_xlim([-0.6, 0.4])
        ax.set_ylim([-0.15, 0.3])
        
        # Legend
        ax.legend(fontsize=self.style['fonts']['legend'], loc='upper left', framealpha=0.95)
        
        plt.tight_layout()
        self._save_figure("figure_vector_equivalence")
    
    # ========================================================================
    # FIGURE 6: NULLSPACE DIMENSIONALITY (from JSON)
    # ========================================================================
    def _plot_geometric_semantic_decoupling(self):
        """2D scatter plot: cosine similarity vs Cohen's d effect size from vector equivalence data"""
        
        # Load data from vector equivalence JSON files
        data = self._load_json_data('vector_equivalence')
        if not data:
            print("No vector equivalence data found")
            return
        
        # Extract scatter data by model
        model_data = {'Qwen': {'cosines': [], 'cohens': []},
                     'Llama': {'cosines': [], 'cohens': []}}
        
        for key, value in data.items():
            if isinstance(value, dict):
                model_name = value.get('model', '').split('/')[-1]
                model_short = 'Llama' if 'Llama' in model_name else 'Qwen' if 'Qwen' in model_name else None
                
                if model_short and model_short in model_data:
                    cos_val = value.get('cosine_similarity', 0)
                    cohens_d = abs(value.get('cohens_d', 0))
                    model_data[model_short]['cosines'].append(cos_val)
                    model_data[model_short]['cohens'].append(cohens_d)
        
        if not any(model_data[m]['cosines'] for m in model_data):
    
            return
        
        # Create figure with background regions
        fig, ax = plt.subplots(
            figsize=(self.style['paper']['single_column']['width'], 
                     self.style['paper']['single_column']['height']),
            dpi=self.style['paper']['dpi']
        )
        
        # Add subtle quadrant shading
        ax.axvspan(-1, 0, -0.2, 1.0, alpha=0.12, color='#FF5252', zorder=0)
        ax.axvspan(0, 1, -0.2, 1.0, alpha=0.12, color='#FF5252', zorder=0)
        ax.axhspan(-1, 0.2, alpha=0.14, color='#66D165', zorder=0)
        
        # Threshold lines
        ax.axhline(y=0.2, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.4, zorder=1)
        
        # Scatter plots for each model
        for model_short in ['Qwen', 'Llama']:
            if model_data[model_short]['cosines']:
                marker = 's' if model_short == 'Qwen' else '^'
                ax.scatter(model_data[model_short]['cosines'],
                          model_data[model_short]['cohens'],
                          s=120, alpha=0.75,
                          color=self.model_colors[model_short], label=model_short,
                          edgecolors='black', linewidth=1.2, marker=marker, zorder=3)
        
        # Apply plot styling
        self._apply_plot_style(ax, 'scatter_plots')
        
        # Formatting
        ax.set_xlabel('cos(v₁, v₂)', fontsize=self.style['fonts']['axis_label'], weight='normal')
        ax.set_ylabel("|Cohen's d|", fontsize=self.style['fonts']['axis_label'], weight='normal')
        ax.set_xlim([-0.6, 0.4])
        ax.set_ylim([-0.15, 0.3])
        
        # Grid for readability
        ax.grid(True, axis='both', linestyle=':', alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(fontsize=self.style['fonts']['legend'] - 1, loc='upper left', 
                 framealpha=0.95, frameon=True)
        
        plt.tight_layout()
        self._save_figure("figure_geometric_semantic_decoupling")
    
    # ========================================================================
    # FIGURE 6: NULLSPACE SPANNING (Cohen's d distribution - unique scatter plot)
    # ========================================================================
    def _plot_nullspace_spanning(self):
        """Nullspace spanning scatter plot - loaded from JSON"""
        
        # Load data from JSON files
        data = self._load_json_data('nullspace_spanning')
        if not data:
            print("No nullspace spanning data found")
            return
        

        
        # Trait display names (abbreviated)
        trait_abbrev = {
            'agreeableness': 'Agree',
            'formality': 'Form',
            'politeness': 'Polite',
            'sentiment': 'Sent',
            'truthfulness': 'Truth'
        }
        
        traits = list(trait_abbrev.values())
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(self.style['paper']['single_column']['width'],
                     self.style['paper']['single_column']['height']),
            dpi=self.style['paper']['dpi']
        )
        
        # Collect all data points (individual nullspace directions)
        x_positions = []
        y_values = []
        colors = []
        
        trait_to_x = {trait: idx for idx, trait in enumerate(traits)}
        
        for result in data['results']:
            trait_full = result['trait'].lower()
            trait_display = trait_abbrev[trait_full]
            model_name = 'Qwen' if 'Qwen' in result['model'] else 'Llama'
            color = self.model_colors[model_name]
            
            # Get individual Cohen's d values for all nullspace directions
            individual_cohens = [d['cohens_d'] for d in result['subexperiment_a']['individual_directions']]
            
            # Add jitter to x-position
            x_base = trait_to_x[trait_display]
            x_offset = 0.12 if model_name == 'Qwen' else -0.12
            
            for cohen_d in individual_cohens:
                jitter = np.random.normal(0, 0.02)  # Small random jitter
                x_positions.append(x_base + x_offset + jitter)
                y_values.append(cohen_d)
                colors.append(color)
        
        # Scatter plot with transparency
        for model_name, model_color in [('Qwen', self.model_colors['Qwen']), 
                                        ('Llama', self.model_colors['Llama'])]:
            mask = [colors[i] == model_color for i in range(len(colors))]
            x_subset = [x_positions[i] for i in range(len(x_positions)) if mask[i]]
            y_subset = [y_values[i] for i in range(len(y_values)) if mask[i]]
            
            ax.scatter(x_subset, y_subset, s=50, alpha=0.6, color=model_color,
                      label=model_name, edgecolors='black', linewidth=0.5)
        
        # Add mean and quartile lines for each trait
        for trait_idx, trait_display in enumerate(traits):
            trait_values_all = []
            
            # Find the trait in trait_abbrev by matching the value
            trait_full = [key for key, val in trait_abbrev.items() if val == trait_display][0]
            
            for result in data['results']:
                if result['trait'].lower() == trait_full:
                    individual_cohens = [d['cohens_d'] for d in result['subexperiment_a']['individual_directions']]
                    trait_values_all.extend(individual_cohens)
            
            if trait_values_all:
                mean = np.mean(trait_values_all)
                q1 = np.percentile(trait_values_all, 25)
                q3 = np.percentile(trait_values_all, 75)
                
                # Mean line
                ax.hlines(mean, trait_idx - 0.25, trait_idx + 0.25, 
                         colors='black', linewidth=2, zorder=10)
                
                # Quartile range
                ax.vlines(trait_idx, q1, q3, colors='gray', linewidth=2, 
                         alpha=0.4, zorder=9)
        
        # Formatting
        ax.set_xlabel('', fontsize=self.style['fonts']['axis_label'])
        ax.set_ylabel("Cohen's d", 
                     fontsize=self.style['fonts']['axis_label'])
        ax.set_xticks(range(len(traits)))
        ax.set_xticklabels(traits, fontsize=self.style['fonts']['tick_label'])
        ax.tick_params(axis='y', labelsize=self.style['fonts']['tick_label'])
        
        # Add horizontal line at Cohen's d = 0.2 (small effect threshold) - NO legend entry
        ax.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.5, zorder=0)
        
        # Grid
        ax.grid(True, axis='y', alpha=0.2, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Legend
        ax.legend(loc='upper left', 
                 framealpha=0.95, fontsize=7, handlelength=1.5, labelspacing=0.3)
        
        # Y-axis limits
        ax.set_ylim([-0.05, 1.0])
        
        plt.tight_layout()
        
        # Save
        self._save_figure("figure_nullspace_spanning")

    # ========================================================================
    # FIGURE 7: NULLSPACE DIMENSIONALITY (3D surface - separate model plots)
    # ========================================================================
    def _plot_nullspace_dimensionality(self):
        """Nullspace dimensionality visualization - loaded from JSON"""
        
        # Load data from JSON files
        data = self._load_json_data('nullspace_dimensionality')
        if not data:
            print("No nullspace dimensionality data found")
            return
        

        
        # data is a dict with individual files merged
        # Extract model-specific data from loaded JSON files
        models_data = {}
        for json_file in self.json_files.get('nullspace_dimensionality', []):
            with open(json_file) as f:
                model_data = json.load(f)
                model_name = json_file.stem.replace('nullspace_dimensionality_', '')
                if model_name in model_data or 'by_layer' in model_data:
                    models_data[model_name] = model_data
        
        if not models_data:
            print("Could not extract model data from JSON files")
            return
        
        # Create separate plots for each model
        for json_file in self.json_files.get('nullspace_dimensionality', []):
            try:
                with open(json_file) as f:
                    model_data = json.load(f)
                
                model_name = json_file.stem.replace('nullspace_dimensionality_', '')
                fig = plt.figure(figsize=(8.5, 6.5), dpi=self.style['paper']['dpi'])
                ax = fig.add_subplot(111, projection='3d')
                
                # Determine model color for scatter points
                point_color = '#FF3333' if 'Llama' in model_name else '#0066FF'
                label_text = '(a)' if 'Llama' in model_name else '(b)'
                
                # Extract data for 3 layers
                layers = ['L/4', 'L/2', '3L/4']
                layer_data_all = []
                
                if 'by_layer' in model_data:
                    for layer in layers:
                        if layer in model_data['by_layer']:
                            sv = np.array(model_data['by_layer'][layer]['singular_values'][:100])
                            layer_data_all.append(sv)
                
                if not layer_data_all:
                    print(f"Could not extract layer data for {model_name}")
                    continue
                
                # Create X (dimensions), Y (layers), Z (singular values)
                n_dims = 100
                X = np.arange(n_dims)
                Y = np.array([0, 1, 2])
                X_mesh, Y_mesh = np.meshgrid(X, Y)
                Z_mesh = np.array(layer_data_all)
                
                # Plot surface with gradient coloring
                surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, 
                                      cmap='viridis', alpha=0.85,
                                      linewidth=0, antialiased=True,
                                      vmin=0, vmax=Z_mesh.max())
                
                # Plot wireframe for definition
                ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, 
                                 color='gray', alpha=0.2, linewidth=0.4)
                
                # Add scatter points on the surface (sample every Nth point for clarity)
                for layer_idx, layer_vals in enumerate(layer_data_all):
                    # Sample every 5th dimension to avoid clutter
                    dims_sample = np.arange(0, len(layer_vals), 5)
                    for dim in dims_sample:
                        ax.scatter([dim], [layer_idx], [layer_vals[dim]], 
                                  color=point_color, s=40, alpha=0.9, edgecolors='white', linewidth=0.8, zorder=5)
                
                # Labels with increased font size
                ax.set_xlabel('Dimension', fontsize=11, labelpad=8)
                ax.set_ylabel('Layer', fontsize=11, labelpad=8)
                ax.set_zlabel('Singular Value', fontsize=11, labelpad=8)
                
                # Layer labels
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(['L/4', 'L/2', '3L/4'], fontsize=10)
                
                # Axis tick labels
                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='z', labelsize=10)
                
                # View angle
                ax.view_init(elev=25, azim=45)
                
                # Remove background panes
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                
                # Add label (a) or (b) at top of y-axis, shifted left
                ax.text(-15, 2.3, max(Z_mesh.flatten()), label_text, fontsize=12, weight='bold')
                
                plt.tight_layout(rect=[0, 0, 1, 1])
                
                # Save separate plots
                base_name = f"figure_nullspace_dimensionality_{model_name.lower()}"
                png_path = self.figures_dir / f"{base_name}.png"
                pdf_path = self.figures_dir / f"{base_name}.pdf"
                
                plt.savefig(png_path, dpi=self.style['paper']['dpi'], bbox_inches='tight', pad_inches=0.05)
                print(f" Saved: {png_path}")
                
                plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05, format='pdf')
                print(f" Saved: {pdf_path}")
                
                plt.close()
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate figures from JSON experiment results'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=False,
        choices=['orthogonal', 'multi_environment', 'logit_distance', 'alpha_sweep', 
                 'vector_equivalence', 'nullspace_spanning', 'nullspace_dimensionality', 'all'],
        default='all',
        help='Which experiment figure to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    generator = FigureGenerator()
    
    available_experiments = [k for k, v in generator.json_files.items() if v]
    
    if args.experiment == 'all':
        for exp in available_experiments:
            success = generator.run(exp)
            if not success:
                print(f"Skipped: {exp}")
    else:
        if args.experiment not in available_experiments:
            print(f"No data found for experiment: {args.experiment}")
            print(f"Available: {', '.join(available_experiments)}")
            return
        generator.run(args.experiment)
    
    print(f"\n{'='*80}")
    print("Figure generation complete!")
    print(f"Figures saved to: {generator.figures_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
