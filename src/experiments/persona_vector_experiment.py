import torch
import numpy as np
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
import json
import yaml
import os
import sys


class PersonaVectorExperiment:
    """
    Implements the non-identifiability experiment from Section 6 of the paper.
    Tests two models (Qwen2.5, Llama-3.1) and two traits (formality, sentiment).
    """

    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        
        # Load configs
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
        with open(os.path.join(config_dir, 'prompts.json'), 'r') as f:
            self.config = json.load(f)
        with open(os.path.join(config_dir, 'config.yml'), 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Initialize semantic probes
        self._init_semantic_probes()

    def _init_semantic_probes(self):
        """
        Initialize semantic probes for computing φ(o).
        These are lightweight models that measure semantic properties.
        """
        print("Loading semantic probes...")

        # Sentiment probe: RoBERTa-based, outputs score in [0, 1]
        try:
            self.sentiment_probe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            print("Loaded RoBERTa sentiment probe")
        except Exception as e:
            print(f"Warning: Could not load RoBERTa sentiment probe: {e}")
            self.sentiment_probe = None

        # Formality probe: try neural classifier first, fall back to heuristic
        try:
            self.formality_probe = pipeline(
                "text-classification",
                model="s-nlp/roberta-base-formality-ranker",
                device=0 if self.device == "cuda" else -1
            )
            self.formality_is_neural = True
            print("Loaded neural formality probe")
        except Exception as e:
            print(f"Could not load neural formality probe: {e}")
            print("Falling back to heuristic formality probe")
            self.formality_probe = self._formality_heuristic
            self.formality_is_neural = False

        # Politeness probe: BART zero-shot (Yin et al. 2019, MNLI-based)
        # Labels: "polite and respectful" vs "rude and disrespectful"
        try:
            self.politeness_probe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            self.politeness_is_neural = True
            print("Loaded BART zero-shot politeness probe")
        except Exception as e:
            print(f"Could not load BART politeness probe: {e}")
            print("Falling back to heuristic politeness probe")
            self.politeness_probe = self._politeness_heuristic
            self.politeness_is_neural = False

        # Agreeableness probe: BART zero-shot (Yin et al. 2019, MNLI-based)
        # Labels: "agreeable and cooperative" vs "disagreeable and contentious"
        try:
            self.agreeableness_probe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            self.agreeableness_is_neural = True
            print("Loaded BART zero-shot agreeableness probe")
        except Exception as e:
            print(f"Could not load BART agreeableness probe: {e}")
            print("Falling back to heuristic agreeableness probe")
            self.agreeableness_is_neural = False

        # Truthfulness probe: BART zero-shot (Yin et al. 2019, MNLI-based)
        try:
            self.truthfulness_probe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            print("Loaded BART zero-shot truthfulness probe")
        except Exception as e:
            print(f"Warning: Could not load BART zero-shot probe: {e}")
            print("Falling back to heuristic truthfulness probe")
            self.truthfulness_probe = None

        print("Semantic probes loaded.")


    def _formality_heuristic(self, text: str) -> float:
        """
        Simple formality score based on linguistic features.
        Returns value in [0, 1] where 1 is most formal.
        """
        # Informal markers
        informal_markers = ['gonna', 'wanna', 'yeah', 'nah', 'hey', 'cool', 'dude',
                          'stuff', 'lots', 'kinda', 'sorta', '!', 'lol', 'omg']

        # Formal markers
        formal_markers = ['therefore', 'furthermore', 'consequently', 'moreover',
                         'nevertheless', 'thus', 'hence', 'regarding', 'pursuant']

        text_lower = text.lower()

        informal_count = sum(1 for marker in informal_markers if marker in text_lower)
        formal_count = sum(1 for marker in formal_markers if marker in text_lower)

        # Sentence length (longer sentences tend to be more formal)
        avg_sentence_len = len(text.split()) / max(text.count('.') + text.count('!') + text.count('?'), 1)

        # Combine features
        formality_score = (formal_count - informal_count + avg_sentence_len / 20.0)

        # Normalize to [0, 1]
        formality_score = 1 / (1 + np.exp(-formality_score))

        return formality_score

    def _politeness_bart_zeroshot(self, text: str) -> float:
        """
        Score politeness using BART zero-shot classification (Yin et al. 2019).
        Classifies between "polite and respectful" vs "rude and disrespectful".
        Returns value in [0, 1] where 1 is most polite.
        """
        try:
            result = self.politeness_probe(
                text[:512],
                ["polite and respectful", "rude and disrespectful"],
                multi_class=False
            )
            
            # result = {'labels': [...], 'scores': [...], ...}
            # Find the score for "polite and respectful"
            polite_score = None
            for label, score in zip(result['labels'], result['scores']):
                if 'polite' in label.lower():
                    polite_score = score
                    break
            
            if polite_score is None:
                polite_score = result['scores'][0] if result['labels'][0] in ["polite and respectful"] else 1.0 - result['scores'][0]
            
            # Clamp to [0, 1]
            polite_score = np.maximum(np.minimum(float(polite_score), 1.0), 0.0)
            return polite_score
        except Exception as e:
            print(f"Warning: BART politeness scoring failed: {e}. Falling back to heuristic.")
            return self._politeness_heuristic(text)


    def _politeness_heuristic(self, text: str) -> float:
        """
        Simple politeness score based on linguistic features.
        Returns value in [0, 1] where 1 is most polite.
        """
        polite_markers = ['please', 'thank you', 'thanks', 'appreciate', 'would you mind',
                         'could you', 'if possible', 'kindly', 'respectfully', 'courtesy',
                         'sincerely', 'regards', 'apologize', 'excuse me', 'pardon']

        rude_markers = ['damn', 'hell', 'shut up', 'idiot', 'stupid', 'useless', 'hate',
                       'disgusting', 'terrible', 'awful', 'horrible']

        text_lower = text.lower()

        polite_count = sum(1 for marker in polite_markers if marker in text_lower)
        rude_count = sum(1 for marker in rude_markers if marker in text_lower)

        # Sentence length (shorter, more concise = polite)
        total_words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_sentence_len = total_words / max(sentences, 1)

        # Combine features
        politeness_score = (polite_count - rude_count + (20.0 / max(avg_sentence_len, 1)))

        # Normalize to [0, 1]
        politeness_score = 1 / (1 + np.exp(-politeness_score))

        return politeness_score


    def _formality_heuristic_updated(self, text: str) -> float:
        """
        Updated formality score based on vocabulary sophistication.
        Focuses on rare vs common words (simpler than comprehensive formality index).
        Returns value in [0, 1] where 1 is most formal (ornate/sophisticated).
        """
        # Sophisticated/rare words (formal)
        sophisticated = ['elaborate', 'ornate', 'sophisticated', 'meticulous', 'prudent',
                        'aesthetic', 'paradigm', 'ephemeral', 'nuanced', 'eloquent',
                        'erudite', 'pensive', 'arcane', 'intricate']
        
        # Common/simple words (informal)
        simple = ['thing', 'stuff', 'really', 'very', 'kind of', 'sort of', 'like',
                 'gonna', 'wanna', 'kinda', 'awesome', 'cool', 'nice']
        
        text_lower = text.lower()
        
        sophisticated_count = sum(1 for w in sophisticated if w in text_lower)
        simple_count = sum(1 for w in simple if w in text_lower)
        
        # Simple scoring: just marker counts
        score = (sophisticated_count - simple_count) / max(len(text.split()) / 10, 1)
        return 1 / (1 + np.exp(-score))  # Normalize to [0, 1]
    
    def _politeness_heuristic_updated(self, text: str) -> float:
        """
        Updated politeness score based on hedging/softening only.
        Focuses narrowly on softeners (not comprehensive politeness markers).
        Returns value in [0, 1] where 1 is most polite (heavily softened).
        """
        # Hedging/softening words (polite)
        hedges = ['perhaps', 'maybe', 'might', 'could', 'somewhat', 'fairly', 'rather',
                 'supposedly', 'arguably', 'if you will', 'so to speak']
        
        # Direct/blunt markers (impolite)
        direct = ['obviously', 'clearly', 'must', 'should', 'definitely', 'absolutely',
                 'never', 'always', 'don\'t', 'won\'t', 'can\'t']
        
        text_lower = text.lower()
        
        hedge_count = sum(1 for h in hedges if h in text_lower)
        direct_count = sum(1 for d in direct if d in text_lower)
        
        # Simple scoring: hedges boost, direct markers reduce
        score = (hedge_count - direct_count) / max(len(text.split()) / 10, 1)
        return 1 / (1 + np.exp(-score))  # Normalize to [0, 1]

    def _sentiment_heuristic(self, text: str) -> float:
        """
        Simple sentiment score based on linguistic features.
        Returns value in [0, 1] where 1 is most positive.
        """
        positive_markers = ['positive', 'good', 'great', 'excellent', 'wonderful', 'amazing',
                           'love', 'perfect', 'fantastic', 'brilliant', 'awesome', 'liked',
                           'happy', 'pleased', 'delighted', 'impressed', 'satisfied',
                           'wonderful', 'outstanding', 'superb', 'brilliant']

        negative_markers = ['negative', 'bad', 'terrible', 'awful', 'horrible', 'dreadful',
                           'hate', 'poor', 'useless', 'disappointing', 'sad', 'angry',
                           'frustrated', 'upset', 'disliked', 'displeased', 'unsatisfied',
                           'weak', 'failed', 'disappointing', 'pathetic']

        text_lower = text.lower()

        positive_count = sum(1 for marker in positive_markers if marker in text_lower)
        negative_count = sum(1 for marker in negative_markers if marker in text_lower)

        # Exclamation marks indicate stronger sentiment
        exclamation_ratio = text.count('!') / max(len(text.split()), 1)

        # Combine features
        sentiment_score = (positive_count - negative_count + (exclamation_ratio * 10.0))

        # Normalize to [0, 1]
        sentiment_score = 1 / (1 + np.exp(-sentiment_score))

        return sentiment_score

    def _truthfulness_bart_zeroshot(self, text: str) -> float:
        """
        Score truthfulness using BART zero-shot classification (Yin et al. 2019).
        Classifies between "accurate and factually correct" vs "plausible but containing inaccuracies".
        Returns value in [0, 1] where 1 is most truthful.
        """
        try:
            result = self.truthfulness_probe(
                text[:512],
                candidate_labels=["accurate and factually correct", "plausible but containing inaccuracies"],
                hypothesis_template="This text is {}."
            )
            
            # Extract scores
            scores = {label: score for label, score in zip(result['labels'], result['scores'])}
            
            # Return probability of first label (truthful)
            truthful_score = scores.get("accurate and factually correct", 0.5)
            return truthful_score
        except Exception as e:
            print(f"Warning: BART truthfulness probe failed: {e}")
            # If BART fails, can't fall back to heuristic — return neutral score
            return 0.5



    def _agreeableness_heuristic(self, text: str) -> float:
        """
        Simple agreeableness score based on linguistic features.
        Returns value in [0, 1] where 1 is most agreeable.
        """
        # Agreement/cooperation markers (agreeable)
        agreeable_markers = ['agree', 'support', 'collaboration', 'together', 'cooperative',
                            'harmonious', 'consensus', 'aligned', 'helpful', 'friendly',
                            'understanding', 'empathetic', 'compassionate', 'considerate',
                            'partner', 'synergy', 'mutual', 'respect', 'appreciate',
                            'indeed', 'certainly', 'absolutely', 'well said']

        # Disagreement/contention markers (disagreeable)
        disagreeable_markers = ['disagree', 'oppose', 'conflict', 'against', 'challenge',
                               'criticism', 'flawed', 'wrong', 'bad idea', 'contrary',
                               'hostile', 'aggressive', 'confrontational', 'argumentative',
                               'contentious', 'rigid', 'stubborn', 'dismissive', 'harsh',
                               'but you are wrong', 'that is incorrect', 'not true']

        text_lower = text.lower()

        agreeable_count = sum(1 for marker in agreeable_markers if marker in text_lower)
        disagreeable_count = sum(1 for marker in disagreeable_markers if marker in text_lower)

        # Sentence structure: shorter, direct sentences tend disagreeable; longer, softer tend agreeable
        sentences = len([s for s in text.split('.') if s.strip()])
        total_words = len(text.split())
        avg_sentence_len = total_words / max(sentences, 1)

        # Softer language tends longer (more elaboration)
        elaboration_bonus = (avg_sentence_len / 20.0)

        # Combine features
        agreeableness_score = (agreeable_count - disagreeable_count + elaboration_bonus)

        # Normalize to [0, 1]
        agreeableness_score = 1 / (1 + np.exp(-agreeableness_score))

    def _agreeableness_bart_zeroshot(self, text: str) -> float:
        """
        Score agreeableness using BART zero-shot classification (Yin et al. 2019).
        Classifies between "agreeable and cooperative" vs "disagreeable and contentious".
        Returns value in [0, 1] where 1 is most agreeable.
        """
        try:
            result = self.agreeableness_probe(
                text[:512],
                ["agreeable and cooperative", "disagreeable and contentious"],
                multi_class=False
            )
            
            # result = {'labels': [...], 'scores': [...], ...}
            # Find the score for "agreeable and cooperative"
            agree_score = None
            for label, score in zip(result['labels'], result['scores']):
                if 'agreeable' in label.lower():
                    agree_score = score
                    break
            
            if agree_score is None:
                agree_score = result['scores'][0] if result['labels'][0] in ["agreeable and cooperative"] else 1.0 - result['scores'][0]
            
            # Clamp to [0, 1]
            agree_score = np.maximum(np.minimum(float(agree_score), 1.0), 0.0)
            return agree_score
        except Exception as e:
            print(f"Warning: BART agreeableness scoring failed: {e}. Falling back to heuristic.")
            return self._agreeableness_heuristic(text)

    def compute_semantic_score(self, text: str, trait: str) -> float:
        """
        Compute semantic score φ(o) for generated text.

        Args:
            text: Generated text
            trait: 'formality', 'formality_updated', 'politeness', 'politeness_updated'

        Returns:
            Scalar semantic score
        """
        if trait == "formality":
            if hasattr(self, 'formality_is_neural') and self.formality_is_neural:
                try:
                    result = self.formality_probe(text[:512])[0]
                    # Map to continuous score: formal → positive, informal → negative
                    is_formal = 'formal' in result['label'].lower()
                    score = result['score'] if is_formal else -result['score']
                    return score  # Returns value in [-1, 1]
                except Exception as e:
                    print(f"Warning: Neural formality probe failed: {e}")
                    return self._formality_heuristic(text)
            else:
                return self._formality_heuristic(text)

        elif trait == "formality_updated":
            return self._formality_heuristic_updated(text)

        elif trait == "politeness":
            # Politeness uses neural BART zero-shot probe if available, falls back to heuristic
            if hasattr(self, 'politeness_is_neural') and self.politeness_is_neural:
                try:
                    return self._politeness_bart_zeroshot(text)
                except Exception as e:
                    print(f"Warning: Neural politeness probe failed: {e}")
                    return self._politeness_heuristic(text)
            else:
                return self._politeness_heuristic(text)

        elif trait == "politeness_updated":
            return self._politeness_heuristic_updated(text)

        elif trait == "sentiment":
            # Use RoBERTa sentiment probe if available, fall back to heuristic
            if self.sentiment_probe is not None:
                try:
                    result = self.sentiment_probe(text[:512])[0]
                    label = result['label'].upper()
                    score = result['score']
                    # Map RoBERTa output: NEGATIVE → -score, NEUTRAL → 0, POSITIVE → +score
                    if 'POSITIVE' in label:
                        return score
                    elif 'NEGATIVE' in label:
                        return -score
                    else:  # NEUTRAL
                        return 0.0
                except Exception as e:
                    print(f"Warning: RoBERTa sentiment probe failed: {e}")
                    return self._sentiment_heuristic(text)
            else:
                return self._sentiment_heuristic(text)

        elif trait == "truthfulness":
            # Use BART zero-shot if available, fall back to heuristic
            if hasattr(self, 'truthfulness_probe') and self.truthfulness_probe is not None:
                return self._truthfulness_bart_zeroshot(text)
            else:
                return self._truthfulness_heuristic(text)

        elif trait == "agreeableness":
            # Agreeableness uses neural BART zero-shot probe if available, falls back to heuristic
            if hasattr(self, 'agreeableness_is_neural') and self.agreeableness_is_neural:
                try:
                    return self._agreeableness_bart_zeroshot(text)
                except Exception as e:
                    print(f"Warning: Neural agreeableness probe failed: {e}")
                    return self._agreeableness_heuristic(text)
            else:
                return self._agreeableness_heuristic(text)

        else:
            raise ValueError(f"Unknown trait: {trait}")

    def create_contrastive_prompts(self, trait: str, n_pairs: int = 50) -> List[Tuple[str, str]]:
        """
        Create contrastive prompt pairs for extracting steering vectors.
        Following the paper's methodology (Section 6.1).
        """
        prompts = []
        persona_prompts = self.config['persona_prompts']

        if trait == "formality":
            contexts = persona_prompts['formality']['contexts']
            positive_template = persona_prompts['formality']['positive_template']
            negative_template = persona_prompts['formality']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                positive = positive_template.format(context=context)
                negative = negative_template.format(context=context)
                prompts.append((positive, negative))

        elif trait == "sentiment":
            contexts = persona_prompts['sentiment']['contexts']
            positive_template = persona_prompts['sentiment']['positive_template']
            negative_template = persona_prompts['sentiment']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                positive = positive_template.format(context=context)
                negative = negative_template.format(context=context)
                prompts.append((positive, negative))

        elif trait == "politeness":
            contexts = persona_prompts['politeness']['contexts']
            positive_template = persona_prompts['politeness']['positive_template']
            negative_template = persona_prompts['politeness']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                polite = positive_template.format(context=context)
                rude = negative_template.format(context=context)
                prompts.append((polite, rude))


        elif trait == "truthfulness":
            contexts = persona_prompts['truthfulness']['contexts']
            positive_template = persona_prompts['truthfulness']['positive_template']
            negative_template = persona_prompts['truthfulness']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                truthful = positive_template.format(context=context)
                untruthful = negative_template.format(context=context)
                prompts.append((truthful, untruthful))

        elif trait == "agreeableness":
            contexts = persona_prompts['agreeableness']['contexts']
            positive_template = persona_prompts['agreeableness']['positive_template']
            negative_template = persona_prompts['agreeableness']['negative_template']
            
            for i, context in enumerate(contexts[:n_pairs]):
                agreeable = positive_template.format(context=context)
                disagreeable = negative_template.format(context=context)
                prompts.append((agreeable, disagreeable))

        return prompts

    def get_hidden_states(self, prompt: str, layer: int = None) -> torch.Tensor:
        """
        Extract hidden states from the model for a given prompt.
        Returns the hidden state at the final token position from the specified or middle layer.
        
        Args:
            prompt: Input prompt
            layer: Layer index (default: None uses middle layer)
            
        Returns:
            Hidden state tensor
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get activation at final token position
            if layer is None:
                layer_idx = len(self.model.model.layers) // 2  # Default to middle layer
            else:
                layer_idx = layer
            hidden_state = outputs.hidden_states[layer_idx][:, -1, :]

        return hidden_state.squeeze(0)

    def extract_steering_vector(self, trait: str, n_pairs: int = 50, layer: int = None) -> torch.Tensor:
        """
        Extract steering vector using contrastive activation addition.
        v = E[h(x+)] - E[h(x-)]
        Following Section 6.1 of the paper.
        
        Args:
            trait: Semantic trait to extract
            n_pairs: Number of contrastive pairs
            layer: Layer index (default: None uses middle layer)
            
        Returns:
            Steering vector
        """
        prompt_pairs = self.create_contrastive_prompts(trait, n_pairs)

        positive_activations = []
        negative_activations = []

        print(f"Extracting {trait} steering vector from {n_pairs} prompt pairs...")
        for pos, neg in tqdm(prompt_pairs):
            pos_hidden = self.get_hidden_states(pos, layer=layer)
            neg_hidden = self.get_hidden_states(neg, layer=layer)

            positive_activations.append(pos_hidden)
            negative_activations.append(neg_hidden)

        # Average and compute difference
        pos_mean = torch.stack(positive_activations).mean(dim=0)
        neg_mean = torch.stack(negative_activations).mean(dim=0)

        steering_vector = pos_mean - neg_mean

        # Ensure float32 for consistent dtype across operations
        return steering_vector.to(torch.float32)

    def extract_steering_vector_pca(self, trait: str, n_pairs: int = 50, layer: int = None) -> torch.Tensor:
        """
        Extract steering vector using PCA on difference vectors.
        Algorithmically independent from contrast vector method.
        
        Computes PCA on matrix of (h_pos - h_neg) differences and returns
        the first principal component as the steering vector.
        
        Args:
            trait: Semantic trait to extract
            n_pairs: Number of contrastive pairs
            layer: Layer index (default: None uses middle layer)
            
        Returns:
            Steering vector (first principal component)
        """
        prompt_pairs = self.create_contrastive_prompts(trait, n_pairs)
        
        differences = []
        
        print(f"Extracting {trait} steering vector via PCA from {n_pairs} prompt pairs...")
        for pos, neg in tqdm(prompt_pairs):
            pos_hidden = self.get_hidden_states(pos, layer=layer)
            neg_hidden = self.get_hidden_states(neg, layer=layer)
            
            diff = pos_hidden - neg_hidden
            differences.append(diff.cpu().numpy())
        
        # Stack differences: [n_pairs, d_model]
        diffs_matrix = np.stack(differences)
        
        # Run PCA
        pca = PCA(n_components=1)
        pca.fit(diffs_matrix)
        
        # Get first principal component
        v_pca = pca.components_[0]  # Shape: [d_model]
        
        # Convert to torch tensor and normalize
        steering_vector = torch.tensor(v_pca, dtype=torch.float32, device=self.device)
        steering_vector = steering_vector / torch.norm(steering_vector)
        
        # Ensure float32 for consistent dtype across operations
        return steering_vector.to(torch.float32)

    def generate_with_steering(self, prompt: str, steering_vector: torch.Tensor,
                               alpha: float = 1.0, max_new_tokens: int = 50,
                               temperature: float = 0.8, num_return_sequences: int = 1,
                               layer: int = None) -> List[str]:
        """
        Generate text with steering vector applied to activations.
        
        Args:
            prompt: Input prompt
            steering_vector: Steering vector to apply
            alpha: Steering strength multiplier
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of sequences to generate
            layer: Layer index to apply steering (default: None uses middle layer)
            
        Returns:
            Generated text(s)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Choose layer for steering hook
        if layer is None:
            layer_idx = len(self.model.model.layers) // 2  # Default to middle layer
        else:
            layer_idx = layer

        def steering_hook(module, input, output):
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            # Add steering to the last token (ensure device and dtype match)
            steering_gpu = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states[:, -1, :] += alpha * steering_gpu

            if rest:
                return (hidden_states,) + rest
            else:
                return hidden_states

        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(steering_hook)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        hook_handle.remove()

        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts

    def test_orthogonal_component_irrelevance(self, prompts: List[str], v: torch.Tensor,
                                             trait: str, alpha: float = 1.0,
                                             n_orthogonal: int = 5) -> Dict:
        """
        Check if components orthogonal to v matter.
        
        If null space is large, then v_perp (orthogonal to v) should have
        minimal semantic effect compared to v_parallel.
        
        """
        print(f"\n{'='*80}")
        print(f"Testing Orthogonal Component Irrelevance")
        print(f"{'='*80}\n")
        
        v_norm = torch.norm(v).item()
        
        # Test 1: Original vector v
        print("Test 1: Original vector v")
        scores_v = []
        for prompt in tqdm(prompts[:20], desc="Original v"):
            texts = self.generate_with_steering(prompt, v, alpha=alpha,
                                                max_new_tokens=40, num_return_sequences=5)
            scores = [self.compute_semantic_score(text, trait) for text in texts]
            scores_v.extend(scores)
        
        mean_v = np.mean(scores_v)
        std_v = np.std(scores_v)
        print(f"  Scores with v: μ={mean_v:.4f}, σ={std_v:.4f}\n")
        
        # Test 2: v + orthogonal component (should be similar to v if null space exists)
        print("Test 2: v + random orthogonal components")
        results = []
        
        for i in range(n_orthogonal):
            # Create random vector orthogonal to v
            random_vec = torch.randn_like(v)
            v_perp = random_vec - (random_vec @ v) / (v @ v) * v  # Gram-Schmidt
            v_perp = v_perp / torch.norm(v_perp) * v_norm  # Same magnitude as v
            
            v_plus_perp = v + v_perp
            
            print(f"\n  Orthogonal vector {i+1}/{n_orthogonal}:")
            print(f"    ||v_perp|| = {torch.norm(v_perp).item():.4f}")
            print(f"    v · v_perp = {(v @ v_perp).item():.6f} (should be ~0)")
            
            scores_perp = []
            for prompt in tqdm(prompts[:20], desc=f"  v+perp {i+1}", leave=False):
                texts = self.generate_with_steering(prompt, v_plus_perp, alpha=alpha,
                                                    max_new_tokens=40, num_return_sequences=5)
                scores = [self.compute_semantic_score(text, trait) for text in texts]
                scores_perp.extend(scores)
            
            mean_perp = np.mean(scores_perp)
            std_perp = np.std(scores_perp)
            
            mean_diff = abs(mean_perp - mean_v)
            cohens_d = mean_diff / np.sqrt((std_v**2 + std_perp**2) / 2)
            correlation = np.corrcoef(scores_v, scores_perp)[0, 1]
            
            print(f"    Scores: μ={mean_perp:.4f}, σ={std_perp:.4f}")
            print(f"    Cohen's d: {cohens_d:.4f}")
            print(f"    Correlation: {correlation:.4f}")
            
            results.append({
                'cohens_d': cohens_d,
                'correlation': correlation,
                'mean_diff': mean_diff
            })
        
        # Test 3: Just the orthogonal component (should have weak effect)
        print(f"\nTest 3: Pure orthogonal components (without v)")
        orthogonal_effects = []
        
        for i in range(n_orthogonal):
            random_vec = torch.randn_like(v)
            v_perp = random_vec - (random_vec @ v) / (v @ v) * v
            v_perp = v_perp / torch.norm(v_perp) * v_norm
            
            scores_only_perp = []
            for prompt in tqdm(prompts[:20], desc=f"  perp-only {i+1}", leave=False):
                texts = self.generate_with_steering(prompt, v_perp, alpha=alpha,
                                                    max_new_tokens=40, num_return_sequences=5)
                scores = [self.compute_semantic_score(text, trait) for text in texts]
                scores_only_perp.extend(scores)
            
            mean_only_perp = np.mean(scores_only_perp)
            effect_ratio = abs(mean_only_perp) / abs(mean_v) if mean_v != 0 else 0
            orthogonal_effects.append(effect_ratio)
            
            print(f" Orthogonal {i+1}: effect = {abs(mean_only_perp):.4f} ({effect_ratio*100:.1f}% of v)")
        
        cohens_ds = [r['cohens_d'] for r in results]
        correlations = [r['correlation'] for r in results]
        
        print(f"v + orthogonal: Cohen's d = {np.mean(cohens_ds):.4f} ± {np.std(cohens_ds):.4f}")
        print(f"v + orthogonal: Correlation = {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        print(f"Orthogonal-only: Mean effect = {np.mean(orthogonal_effects)*100:.1f}% ± {np.std(orthogonal_effects)*100:.1f}%")
        
        return {
            'v_scores': {'mean': mean_v, 'std': std_v},
            'v_plus_perp': results,
            'perp_only_effects': orthogonal_effects,
            'summary': {
                'mean_cohens_d': np.mean(cohens_ds),
                'mean_correlation': np.mean(correlations),
                'mean_perp_effect': np.mean(orthogonal_effects)
            }
        }