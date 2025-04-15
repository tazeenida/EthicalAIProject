import os
import json
from datetime import datetime

# Import from the sample_responses module
from sample_responses import batch_generate_sample_responses, save_sample_responses
# Import analysis functionality
from analysis import load_all_results, analyze_bias_by_category, plot_bias_by_category, create_bias_heatmap, generate_summary_report
# Import evaluation functionality
from bias_evaluation import generate_profession_prompts, generate_socioeconomic_prompts, generate_age_prompts, evaluate_text_for_all_biases
# Import ethical analysis 
from ethical_analysis import EthicalAnalyzer
# Import debiasing functionality
from debiasing import run_debiasing_evaluation

def run_free_bias_analysis():
    """Run bias analysis using only pre-generated sample responses"""
    
    # Create required directories
    os.makedirs("data/responses", exist_ok=True)
    os.makedirs("data/results/figures", exist_ok=True)
    
    print("Starting bias analysis with sample responses...")
    
    # 1. Generate prompts
    print("Generating evaluation prompts...")
    profession_prompts = generate_profession_prompts()
    socioeconomic_prompts = generate_socioeconomic_prompts()
    age_prompts = generate_age_prompts()
    all_prompts = profession_prompts + socioeconomic_prompts + age_prompts
    
    # 2. Get sample responses (no API calls) - now with multiple models
    print("Generating sample responses...")
    models = ["gpt2", "gpt-3", "llama2-7b"]  # Using GPT-2 + simulated advanced models
    responses = batch_generate_sample_responses(all_prompts, models)
    
    # 3. Save responses
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_name, model_responses in responses.items():
        filename = f"data/responses/sample_responses_{model_name}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(model_responses, f, indent=4)
    
    # 4. Run analysis on these responses for each model
    results_by_model = {}
    for model_name, model_responses in responses.items():
        print(f"\nAnalyzing responses from {model_name}...")
        model_data = {
            "profession": {},
            "socioeconomic": {},
            "age": {}
        }
        
        # Process profession prompts
        for prompt in profession_prompts:
            profession = prompt.split("typical ")[1].strip(".")
            if prompt in model_responses:
                response = model_responses[prompt]
                bias_scores = evaluate_text_for_all_biases(response)
                model_data["profession"][profession] = bias_scores
        
        # Process socioeconomic prompts
        for prompt in socioeconomic_prompts:
            context = prompt.split("typical person from ")[1].strip(".")
            if prompt in model_responses:
                response = model_responses[prompt]
                bias_scores = evaluate_text_for_all_biases(response)
                model_data["socioeconomic"][context] = bias_scores
        
        # Process age prompts
        for prompt in age_prompts:
            age_group = prompt.split("typical ")[1].strip(".")
            if prompt in model_responses:
                response = model_responses[prompt]
                bias_scores = evaluate_text_for_all_biases(response)
                model_data["age"][age_group] = bias_scores
        
        results_by_model[model_name] = model_data
        
        # Save analysis results for this model
        with open(f"data/results/evaluation_{model_name}_{timestamp}.json", "w") as f:
            json.dump(model_data, f, indent=4)
    
    # 5. Combined analysis with all models
    all_model_results = []
    for model_name, model_data in results_by_model.items():
        combined_data = {model_name: model_data, "_source_file": f"evaluation_{model_name}_{timestamp}.json"}
        all_model_results.append(combined_data)
    
    # 6. Run visualization and reporting for each model
    print("\nVisualizing results for each model...")
    for model_name, model_data in results_by_model.items():
        print(f"\nGenerating visualizations for {model_name}...")
        model_results = [all_model_results[list(results_by_model.keys()).index(model_name)]]
        
        for bias_type in ["gender", "racial", "socioeconomic", "age"]:
            # Analyze by profession
            avg_scores, std_scores = analyze_bias_by_category(model_results, bias_type, "profession")
            if avg_scores:
                plot_bias_by_category(avg_scores, std_scores, bias_type, "profession", 
                                    f"data/results/figures/{model_name}_{bias_type}_by_profession_{timestamp}.png")
            
            # Analyze by socioeconomic context
            avg_scores, std_scores = analyze_bias_by_category(model_results, bias_type, "socioeconomic")
            if avg_scores:
                plot_bias_by_category(avg_scores, std_scores, bias_type, "socioeconomic",
                                    f"data/results/figures/{model_name}_{bias_type}_by_socioeconomic_{timestamp}.png")
            
            # Analyze by age
            avg_scores, std_scores = analyze_bias_by_category(model_results, bias_type, "age")
            if avg_scores:
                plot_bias_by_category(avg_scores, std_scores, bias_type, "age",
                                    f"data/results/figures/{model_name}_{bias_type}_by_age_{timestamp}.png")
        
        # Create heatmaps for this model
        create_bias_heatmap(model_results, category="profession", 
                          output_file=f"data/results/figures/{model_name}_heatmap_profession_{timestamp}.png")
        create_bias_heatmap(model_results, category="socioeconomic",
                          output_file=f"data/results/figures/{model_name}_heatmap_socioeconomic_{timestamp}.png")
        create_bias_heatmap(model_results, category="age",
                          output_file=f"data/results/figures/{model_name}_heatmap_age_{timestamp}.png")
        
        # Generate summary report for this model
        generate_summary_report(model_results, f"data/results/{model_name}_summary_report_{timestamp}.md")
    
    # 7. Run cross-model comparison
    print("\nGenerating cross-model comparison...")
    # Create comparison visualizations across models
    create_model_comparison_visualizations(results_by_model, timestamp)
    
    # 8. Run debiasing evaluation on one model's responses
    print("\nRunning debiasing evaluation...")
    # Use a subset of the responses to keep runtime reasonable
    sample_subset = {k: responses["gpt2"][k] for k in list(responses["gpt2"].keys())[:5]}
    run_debiasing_evaluation(sample_subset)
    
    print("\nAnalysis complete! Results saved to data/results/")
    print("Check data/results/figures/ for visualizations")
    print("Check data/results/ for the summary reports")

def create_model_comparison_visualizations(results_by_model, timestamp):
    """Create visualizations comparing bias across different models"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Setup
    os.makedirs("data/results/figures", exist_ok=True)
    bias_types = ["gender", "racial", "socioeconomic", "age"]
    model_names = list(results_by_model.keys())
    
    # 1. Compare average bias magnitude by model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, bias_type in enumerate(bias_types):
        avg_bias_by_model = []
        
        for model_name in model_names:
            model_data = results_by_model[model_name]
            all_bias_scores = []
            
            # Collect all bias scores for this type
            for category in ["profession", "socioeconomic", "age"]:
                for item, scores in model_data[category].items():
                    bias_key = f"{bias_type}_bias"
                    if bias_key in scores:
                        all_bias_scores.append(abs(scores[bias_key]))
            
            # Calculate average bias magnitude
            if all_bias_scores:
                avg_bias = np.mean(all_bias_scores)
                avg_bias_by_model.append(avg_bias)
            else:
                avg_bias_by_model.append(0)
        
        # Plot this bias type
        x = np.arange(len(model_names))
        axes[i].bar(x, avg_bias_by_model, color='skyblue')
        axes[i].set_title(f'Average {bias_type.capitalize()} Bias Magnitude by Model')
        axes[i].set_ylabel('Bias Magnitude (absolute value)')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45)
        
        # Add value labels
        for j, v in enumerate(avg_bias_by_model):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"data/results/figures/model_comparison_by_bias_type_{timestamp}.png")
    plt.close()
    
    # 2. Create overall bias comparison chart
    overall_bias_by_model = {}
    for model_name in model_names:
        model_data = results_by_model[model_name]
        all_bias_scores = []
        
        # Collect all bias scores
        for category in ["profession", "socioeconomic", "age"]:
            for item, scores in model_data[category].items():
                for bias_type in bias_types:
                    bias_key = f"{bias_type}_bias"
                    if bias_key in scores:
                        all_bias_scores.append(abs(scores[bias_key]))
        
        # Calculate overall bias metrics
        if all_bias_scores:
            overall_bias_by_model[model_name] = {
                "mean": np.mean(all_bias_scores),
                "median": np.median(all_bias_scores),
                "max": np.max(all_bias_scores),
                "min": np.min(all_bias_scores)
            }
    
    # Plot overall comparison
    metrics = ["mean", "median", "max", "min"]
    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
    
    x = np.arange(len(model_names))
    width = 0.2
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        values = [overall_bias_by_model[model][metric] for model in model_names]
        rects = ax.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    ax.set_title('Overall Bias Metrics by Model')
    ax.set_ylabel('Bias Magnitude')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"data/results/figures/overall_bias_comparison_{timestamp}.png")
    plt.close()
    
    # 3. Save comparison data
    comparison_data = {
        "overall_bias_by_model": overall_bias_by_model,
        "timestamp": timestamp
    }
    
    with open(f"data/results/model_comparison_{timestamp}.json", "w") as f:
        json.dump(comparison_data, f, indent=4)

if __name__ == "__main__":
    run_free_bias_analysis()