"""
Module for implementing and evaluating debiasing techniques
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import Dict, List

class DebiasingEngine:
    def __init__(self, model_name="gpt-neo-1.3B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback response generation")
    
    def counterfactual_augmentation(self, prompt: str, num_variants: int = 3) -> list:
        """Generate counterfactual versions of a prompt by swapping demographic terms"""
        cf_prompts = []
        
        # Gender swapping pairs
        gendered_pairs = [
            ("he", "she"), ("she", "he"),
            ("his", "her"), ("her", "his"),
            ("him", "her"), ("man", "woman"),
            ("woman", "man"), ("boy", "girl"),
            ("girl", "boy"), ("father", "mother"),
            ("mother", "father"), ("son", "daughter"),
            ("daughter", "son"), ("brother", "sister"),
            ("sister", "brother"), ("male", "female"),
            ("female", "male")
        ]
        
        # Race/ethnicity swapping pairs
        racial_pairs = [
            ("white", "black"), ("black", "white"),
            ("african american", "caucasian"), ("caucasian", "african american"),
            ("asian", "hispanic"), ("hispanic", "asian")
        ]
        
        # Age swapping pairs
        age_pairs = [
            ("young", "old"), ("old", "young"),
            ("elderly", "youth"), ("youth", "elderly"),
            ("teenager", "senior"), ("senior", "teenager")
        ]
        
        # Socioeconomic swapping pairs
        socioeconomic_pairs = [
            ("rich", "poor"), ("poor", "rich"),
            ("wealthy", "low-income"), ("low-income", "wealthy"),
            ("privileged", "underprivileged"), ("underprivileged", "privileged")
        ]
        
        # Create variants using all swap types
        all_pairs = [gendered_pairs, racial_pairs, age_pairs, socioeconomic_pairs]
        
        for pairs in all_pairs:
            modified = prompt.lower()
            for term, replacement in pairs:
                if " " + term + " " in " " + modified + " ":
                    modified = modified.replace(term, replacement)
                    # Break after one replacement to avoid cascading changes
                    if modified != prompt.lower():
                        cf_prompts.append(modified)
                        break
        
        # Deduplicate and limit to requested number
        cf_prompts = list(set(cf_prompts))[:num_variants]
        
        return cf_prompts if cf_prompts else [prompt]
    
    def generate_debiased_response(self, prompt: str, max_length: int = 150, debiasing_method: str = "counterfactual") -> str:
        """Generate response using specified debiasing technique"""
        if debiasing_method == "counterfactual":
            return self._debias_with_counterfactual_augmentation(prompt, max_length)
        elif debiasing_method == "neutralization":
            return self._debias_with_neutralization(prompt, max_length)
        elif debiasing_method == "balanced_examples":
            return self._debias_with_balanced_examples(prompt, max_length)
        else:
            print(f"Unknown debiasing method: {debiasing_method}, using counterfactual")
            return self._debias_with_counterfactual_augmentation(prompt, max_length)
    
    def _debias_with_counterfactual_augmentation(self, prompt: str, max_length: int = 150) -> str:
        """Use counterfactual data augmentation to debias responses"""
        # Generate counterfactual prompts
        cf_prompts = self.counterfactual_augmentation(prompt)
        all_responses = []
        
        # If model is not loaded, use sample responses
        if self.model is None:
            try:
                from sample_responses import get_sample_response
                for cf_prompt in cf_prompts:
                    response = get_sample_response(cf_prompt, self.model_name)
                    all_responses.append(response)
            except ImportError:
                return f"Cannot generate response: model not loaded and sample_responses not available"
        else:
            import torch
            for cf_prompt in cf_prompts:
                inputs = self.tokenizer(cf_prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length + inputs.input_ids.shape[1],
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if response.startswith(cf_prompt):
                    response = response[len(cf_prompt):]
                all_responses.append(response.strip())
        
        # Select the most neutral response based on bias scores
        try:
            from bias_evaluation import evaluate_text_for_all_biases
            import numpy as np
            
            bias_scores = []
            for resp in all_responses:
                scores = evaluate_text_for_all_biases(resp)
                # Use the average of absolute bias scores as a neutrality measure
                bias_scores.append(np.mean([abs(v) for v in scores.values()]))
            
            # Return the response with the lowest bias score
            return all_responses[np.argmin(bias_scores)]
        except ImportError:
            # If bias evaluation not available, return the first response
            return all_responses[0] if all_responses else f"No responses generated for {prompt}"
    
    def _debias_with_neutralization(self, prompt: str, max_length: int = 150) -> str:
        """Neutralize potentially biased language in the prompt and generate a response"""
        # Add prefacing with explicit neutralizing instructions
        neutral_prompt = f"Please provide a fair and neutral response, avoiding any gender, racial, age, or socioeconomic stereotypes: {prompt}"
        
        # If model is not loaded, use sample responses
        if self.model is None:
            try:
                from sample_responses import get_sample_response
                response = get_sample_response(neutral_prompt, self.model_name)
                return response
            except ImportError:
                return f"Cannot generate neutral response: model not loaded and sample_responses not available"
        else:
            import torch
            inputs = self.tokenizer(neutral_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length + inputs.input_ids.shape[1],
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(neutral_prompt):
                response = response[len(neutral_prompt):]
            return response.strip()
    
    def _debias_with_balanced_examples(self, prompt: str, max_length: int = 150) -> str:
        """Use balanced examples to guide the model toward unbiased responses"""
        # Create a few-shot prompt with balanced examples
        balanced_prompt = """
Here are examples of fair descriptions that avoid stereotypes:

Q: Describe a typical doctor.
A: A typical doctor is a medical professional with extensive training who diagnoses and treats health issues. They work in various settings like hospitals or clinics and may specialize in specific areas of medicine. Doctors require strong communication skills to explain complex medical information to patients.

Q: Describe a typical nurse.
A: A typical nurse is a healthcare professional who provides direct patient care and serves as a key coordinator between patients and other medical staff. They administer medications, monitor patient conditions, and educate patients about health management. Nurses work in diverse environments including hospitals, schools, and community health centers.

Q: Describe a typical engineer.
A: A typical engineer applies scientific and mathematical principles to develop solutions to technical problems. They design, test, and improve various systems and products. Engineers often work in teams and may specialize in fields such as civil, electrical, mechanical, or software engineering.

Now, please provide a fair and balanced response to this question:
{prompt}
""".format(prompt=prompt)

        # If model is not loaded, use sample responses
        if self.model is None:
            try:
                from sample_responses import get_sample_response
                response = get_sample_response(balanced_prompt, self.model_name)
                # Extract just the answer part if possible
                if "Now, please provide a fair and balanced response to this question:" in response:
                    response = response.split("Now, please provide a fair and balanced response to this question:")[1].strip()
                return response
            except ImportError:
                return f"Cannot generate balanced response: model not loaded and sample_responses not available"
        else:
            import torch
            inputs = self.tokenizer(balanced_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length + inputs.input_ids.shape[1],
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the answer part
            if "Now, please provide a fair and balanced response to this question:" in response:
                response = response.split("Now, please provide a fair and balanced response to this question:")[1].strip()
            elif response.startswith(balanced_prompt):
                response = response[len(balanced_prompt):]
            return response.strip()
    
    def evaluate_debiasing(self, original_responses: dict, debiasing_methods=None) -> dict:
        """Evaluate effectiveness of different debiasing methods"""
        if debiasing_methods is None:
            debiasing_methods = ["counterfactual", "neutralization", "balanced_examples"]
        
        from bias_evaluation import evaluate_text_for_all_biases
        
        results = {}
        for prompt, original_response in original_responses.items():
            prompt_results = {
                'original': {
                    'response': original_response,
                    'bias_scores': evaluate_text_for_all_biases(original_response)
                }
            }
            
            # Generate and evaluate responses with each debiasing method
            for method in debiasing_methods:
                debiased_response = self.generate_debiased_response(prompt, debiasing_method=method)
                debiased_scores = evaluate_text_for_all_biases(debiased_response)
                
                # Calculate improvement for each bias type
                improvements = {}
                for bias_type, original_score in prompt_results['original']['bias_scores'].items():
                    debiased_score = debiased_scores[bias_type]
                    # Improvement is reduction in absolute bias
                    improvements[bias_type] = abs(original_score) - abs(debiased_score)
                
                prompt_results[method] = {
                    'response': debiased_response,
                    'bias_scores': debiased_scores,
                    'improvements': improvements,
                    'average_improvement': sum(improvements.values()) / len(improvements)
                }
            
            results[prompt] = prompt_results
        
        return results

def compare_debiasing_methods(results: dict) -> dict:
    """Compare the effectiveness of different debiasing methods across prompts"""
    comparison = {
        'methods': {},
        'best_method': {},
        'summary': {}
    }
    
    # Get all debiasing methods used
    all_methods = set()
    for prompt_results in results.values():
        for method in prompt_results.keys():
            if method != 'original':
                all_methods.add(method)
    
    # Initialize method statistics
    for method in all_methods:
        comparison['methods'][method] = {
            'avg_improvement': 0,
            'best_count': 0,
            'improvements_by_bias': {
                'gender_bias': [], 
                'racial_bias': [], 
                'socioeconomic_bias': [], 
                'age_bias': []
            }
        }
    
    # Collect statistics for each method across all prompts
    for prompt, prompt_results in results.items():
        # Find best method for this prompt
        best_method = None
        best_improvement = -float('inf')
        
        for method in all_methods:
            if method in prompt_results and 'average_improvement' in prompt_results[method]:
                avg_improvement = prompt_results[method]['average_improvement']
                
                # Update method statistics
                comparison['methods'][method]['avg_improvement'] += avg_improvement
                
                # Collect bias-specific improvements
                if 'improvements' in prompt_results[method]:
                    for bias_type, improvement in prompt_results[method]['improvements'].items():
                        if bias_type in comparison['methods'][method]['improvements_by_bias']:
                            comparison['methods'][method]['improvements_by_bias'][bias_type].append(improvement)
                
                # Check if this is the best method for this prompt
                if avg_improvement > best_improvement:
                    best_improvement = avg_improvement
                    best_method = method
        
        # Record best method for this prompt
        if best_method:
            comparison['best_method'][prompt] = {
                'method': best_method,
                'improvement': best_improvement
            }
            comparison['methods'][best_method]['best_count'] += 1
    
    # Calculate averages
    num_prompts = len(results)
    for method in all_methods:
        comparison['methods'][method]['avg_improvement'] /= num_prompts
        
        # Calculate average improvement by bias type
        for bias_type, improvements in comparison['methods'][method]['improvements_by_bias'].items():
            if improvements:
                comparison['methods'][method]['improvements_by_bias'][bias_type] = sum(improvements) / len(improvements)
            else:
                comparison['methods'][method]['improvements_by_bias'][bias_type] = 0
    
    # Create summary
    best_overall_method = max(comparison['methods'].items(), 
                            key=lambda x: x[1]['avg_improvement'])
    
    comparison['summary'] = {
        'best_overall_method': best_overall_method[0],
        'best_overall_improvement': best_overall_method[1]['avg_improvement'],
        'method_effectiveness_ranking': sorted(all_methods, 
                                            key=lambda m: comparison['methods'][m]['avg_improvement'], 
                                            reverse=True)
    }
    
    return comparison

def plot_debiasing_comparison(debiasing_results: dict, output_file=None):
    """Plot comparison of debiasing methods"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    
    # Extract methods and bias types
    methods = [m for m in debiasing_results['methods'].keys()]
    bias_types = ['gender_bias', 'racial_bias', 'socioeconomic_bias', 'age_bias']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Subplot 1: Average improvement by method
    avg_improvements = [debiasing_results['methods'][m]['avg_improvement'] for m in methods]
    x = np.arange(len(methods))
    
    bars = ax1.bar(x, avg_improvements, width=0.6, color='skyblue')
    ax1.set_ylabel('Average Bias Reduction')
    ax1.set_title('Overall Effectiveness of Debiasing Methods')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Subplot 2: Bias-specific improvements
    data = []
    for m in methods:
        method_data = []
        for bt in bias_types:
            method_data.append(debiasing_results['methods'][m]['improvements_by_bias'][bt])
        data.append(method_data)
    
    x = np.arange(len(bias_types))
    width = 0.8 / len(methods)
    offsets = np.linspace(-(width * (len(methods)-1)/2), width * (len(methods)-1)/2, len(methods))
    
    for i, (m, d, offset) in enumerate(zip(methods, data, offsets)):
        bars = ax2.bar(x + offset, d, width, label=m)
    
    ax2.set_ylabel('Average Bias Reduction')
    ax2.set_title('Bias-Specific Improvements by Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels([bt.replace('_bias', '') for bt in bias_types], rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure if output file specified
    if output_file:
        plt.savefig(output_file)
    else:
        # Create default output file
        os.makedirs("data/results/figures", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"data/results/figures/debiasing_comparison_{timestamp}.png")
    
    plt.close()

def run_debiasing_evaluation(sample_responses=None, output_dir="data/debiased"):
    """Run comprehensive debiasing evaluation"""
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If no sample responses provided, generate or load some
    if sample_responses is None:
        try:
            from bias_evaluation import generate_profession_prompts
            from sample_responses import batch_generate_sample_responses
            
            prompts = generate_profession_prompts()[:5]  # Use a subset for testing
            sample_responses = batch_generate_sample_responses(prompts, ["gpt2"])["gpt2"]
        except ImportError:
            print("Could not generate sample responses. Please provide sample responses.")
            return
    
    print(f"Running debiasing evaluation on {len(sample_responses)} sample responses...")
    
    # Initialize debiasing engine
    debiaser = DebiasingEngine(model_name="gpt2")
    
    # Evaluate different debiasing methods
    methods = ["counterfactual", "neutralization", "balanced_examples"]
    debiasing_results = debiaser.evaluate_debiasing(sample_responses, methods)
    
    # Compare debiasing methods
    comparison = compare_debiasing_methods(debiasing_results)
    
    # Save detailed results
    with open(f"{output_dir}/debiasing_evaluation_{timestamp}.json", "w") as f:
        json.dump(debiasing_results, f, indent=2)
    
    # Save comparison results
    with open(f"{output_dir}/debiasing_comparison_{timestamp}.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Generate visualization
    plot_debiasing_comparison(comparison, f"{output_dir}/debiasing_comparison_{timestamp}.png")
    
    # Print summary
    print("\nDebiasing Evaluation Summary:")
    print(f"Best overall method: {comparison['summary']['best_overall_method']}")
    print(f"Average improvement: {comparison['summary']['best_overall_improvement']:.4f}")
    print("Method ranking (by effectiveness):")
    for i, method in enumerate(comparison['summary']['method_effectiveness_ranking'], 1):
        avg_imp = comparison['methods'][method]['avg_improvement']
        best_count = comparison['methods'][method]['best_count']
        print(f"  {i}. {method}: avg improvement = {avg_imp:.4f}, best for {best_count} prompts")
    
    return debiasing_results, comparison

if __name__ == "__main__":
    # Example usage
    run_debiasing_evaluation()