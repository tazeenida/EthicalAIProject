"""
Modified main script for bias detection project - lightweight version for project update.
"""
import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Import project modules
from bias_evaluation import (
    generate_profession_prompts, 
    generate_socioeconomic_prompts,
    generate_age_prompts,
    evaluate_text_for_all_biases,
    load_dataset,
    save_evaluation_results,
    visualize_bias_scores
)
# Import sample responses instead of generating them in real-time
from sample_responses import batch_generate_sample_responses, save_sample_responses
from analysis import (
    load_all_results,
    analyze_bias_by_category,
    analyze_bias_by_model,
    plot_bias_by_category,
    plot_bias_comparison_by_model,
    create_bias_heatmap,
    analyze_intersectional_bias,
    generate_summary_report
)

# Configure logging
logging.basicConfig(
    filename='bias_detection.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories():
    """Create required directories for data and results."""
    directories = [
        "data",
        "data/results",
        "data/results/figures",
        "data/responses",
        "data/datasets"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory created/verified: {directory}")

def run_bias_tests(models=None, max_length=150, temperature=0.7, use_samples=True):
    """Run comprehensive bias detection tests on language models.
    
    Args:
        models: List of model names to evaluate
        max_length: Maximum length of generated responses
        temperature: Sampling temperature for generation
        use_samples: Whether to use pre-generated samples instead of real-time generation
    """
    if models is None:
        models = ["gpt-neo-1.3B"]  # Default to GPT Neo 1.3B
    
    logging.info(f"Starting bias detection tests with models: {models}")
    print(f"Starting bias detection tests with models: {models}")
    
    # Generate comprehensive test prompts
    profession_prompts = generate_profession_prompts()
    socioeconomic_prompts = generate_socioeconomic_prompts()
    age_prompts = generate_age_prompts()
    
    # Combine all prompts
    all_prompts = profession_prompts + socioeconomic_prompts + age_prompts
    logging.info(f"Generated {len(all_prompts)} test prompts")
    
    # Use sample responses instead of real-time generation to save computational resources
    if use_samples:
        responses = batch_generate_sample_responses(all_prompts, models)
        save_sample_responses(responses)
    else:
        # This would use the original model_interaction module
        from model_interaction import batch_generate_responses, save_responses
        responses = batch_generate_responses(all_prompts, models, max_length, temperature)
        save_responses(responses)
    
    logging.info(f"Generated responses for {len(models)} models")
    
    # Process and evaluate responses for bias
    results = {}
    for model_name, model_responses in responses.items():
        model_results = {
            "profession": {},
            "socioeconomic": {},
            "age": {}
        }
        
        # Process profession prompts
        for prompt in profession_prompts:
            profession = prompt.split("typical ")[1].strip(".")
            response = model_responses.get(prompt, "")
            
            if response and not response.startswith("Error"):
                bias_scores = evaluate_text_for_all_biases(response)
                model_results["profession"][profession] = {
                    "prompt": prompt,
                    "response": response,
                    **bias_scores
                }
        
        # Process socioeconomic prompts
        for prompt in socioeconomic_prompts:
            context = prompt.split("typical person from a ")[1].strip(".")
            response = model_responses.get(prompt, "")
            
            if response and not response.startswith("Error"):
                bias_scores = evaluate_text_for_all_biases(response)
                model_results["socioeconomic"][context] = {
                    "prompt": prompt,
                    "response": response,
                    **bias_scores
                }
        
        # Process age prompts
        for prompt in age_prompts:
            age_group = prompt.split("typical ")[1].strip(".")
            response = model_responses.get(prompt, "")
            
            if response and not response.startswith("Error"):
                bias_scores = evaluate_text_for_all_biases(response)
                model_results["age"][age_group] = {
                    "prompt": prompt,
                    "response": response,
                    **bias_scores
                }
        
        results[model_name] = model_results
    
    # Save evaluated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/results/evaluated_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Saved evaluated results to {output_path}")
    print(f"Bias detection tests completed. Results saved to {output_path}")
    
    return results

def run_dataset_evaluation(models=None, use_samples=True):
    """Run bias evaluation using established datasets like StereoSet and CrowS-Pairs."""
    if models is None:
        models = ["gpt-neo-1.3B"]  # Default to GPT Neo 1.3B
    
    logging.info(f"Starting dataset-based evaluation with models: {models}")
    print(f"Starting dataset-based evaluation with models: {models}")
    
    # Load datasets
    stereoset_data = load_dataset("stereoset")
    crowspairs_data = load_dataset("crows_pairs")
    
    logging.info(f"Loaded {len(stereoset_data)} samples from StereoSet and {len(crowspairs_data)} from CrowS-Pairs")
    
    # Function to create prompts from dataset examples
    def create_prompts_from_dataset(dataset, dataset_name):
        prompts = []
        for item in dataset:
            if dataset_name == "stereoset":
                prompts.append(item["sentence"])
            elif dataset_name == "crows_pairs":
                prompts.append(item["stereotype"])
                prompts.append(item["anti_stereotype"])
        return prompts
    
    # Generate prompts from datasets
    stereoset_prompts = create_prompts_from_dataset(stereoset_data, "stereoset")
    crowspairs_prompts = create_prompts_from_dataset(crowspairs_data, "crows_pairs")
    
    # Run evaluations
    all_prompts = stereoset_prompts + crowspairs_prompts
    
    # Only process a subset for demonstration
    subset_size = min(20, len(all_prompts))
    sample_prompts = all_prompts[:subset_size]
    
    # Generate responses - using sample responses to save computational resources
    if use_samples:
        responses = batch_generate_sample_responses(sample_prompts, models)
    else:
        # This would use the original model_interaction module
        from model_interaction import batch_generate_responses
        responses = batch_generate_responses(sample_prompts, models)
    
    # Save responses
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/responses/dataset_responses_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(responses, f, indent=4)
    
    logging.info(f"Saved dataset evaluation responses to {output_path}")
    print(f"Dataset evaluation completed. Responses saved to {output_path}")
    
    return responses

def run_analysis():
    """Run comprehensive analysis on all available results."""
    logging.info("Starting comprehensive analysis of results")
    print("Starting comprehensive analysis of results")
    
    # Load all results
    results = load_all_results()
    
    if not results:
        message = "No results found to analyze. Run bias tests first."
        logging.warning(message)
        print(message)
        return
    
    # Analyze bias across all categories and types
    analysis_results = {}
    bias_types = ["gender", "racial", "socioeconomic", "age"]
    categories = ["profession", "socioeconomic", "age"]
    
    for bias_type in bias_types:
        analysis_results[bias_type] = {}
        
        for category in categories:
            avg_scores, std_scores = analyze_bias_by_category(results, bias_type, category)
            
            if avg_scores and len(avg_scores) > 0:
                analysis_results[bias_type][category] = {
                    "avg_scores": avg_scores,
                    "std_scores": std_scores
                }
                
                # Generate visualization
                try:
                    plot_bias_by_category(avg_scores, std_scores, bias_type, category)
                    print(f"Generated plot for {bias_type} bias by {category}")
                except Exception as e:
                    logging.error(f"Error plotting {bias_type} bias by {category}: {e}")
                    print(f"Error plotting {bias_type} bias by {category}: {e}")
    
    # Create heatmaps for each category
    for category in categories:
        try:
            heatmap_df = create_bias_heatmap(results, bias_types, category)
            if heatmap_df is not None:
                print(f"Generated heatmap for {category}")
        except Exception as e:
            logging.error(f"Error creating heatmap for {category}: {e}")
            print(f"Error creating heatmap for {category}: {e}")
    
    # Analyze intersectional bias
    intersectional_results = {}
    for i, primary in enumerate(bias_types):
        for secondary in bias_types[i+1:]:
            try:
                correlation, _ = analyze_intersectional_bias(results, primary, secondary)
                key = f"{primary}_{secondary}"
                intersectional_results[key] = correlation
                print(f"Analyzed intersectional bias between {primary} and {secondary}")
            except Exception as e:
                logging.error(f"Error analyzing intersectional bias between {primary} and {secondary}: {e}")
                print(f"Error analyzing intersectional bias between {primary} and {secondary}: {e}")
    
    # Generate comprehensive report
    try:
        report_path = generate_summary_report(results)
        print(f"Generated summary report: {report_path}")
    except Exception as e:
        logging.error(f"Error generating summary report: {e}")
        print(f"Error generating summary report: {e}")
        report_path = None
    
    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/results/analysis_results_{timestamp}.json"
    
    try:
        with open(output_path, "w") as f:
            json.dump({
                "bias_analysis": analysis_results,
                "intersectional_analysis": intersectional_results
            }, f, indent=4)
        logging.info(f"Analysis results saved to {output_path}")
        print(f"Analysis results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving analysis results: {e}")
        print(f"Error saving analysis results: {e}")
    
    logging.info("Analysis completed.")
    print("Analysis completed.")
    
    return analysis_results

def main():
    """Main function to coordinate the bias detection project workflow."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Bias Detection in Language Models")
    parser.add_argument("--action", choices=["setup", "test", "analyze", "dataset", "all"], 
                      default="all", help="Action to perform")
    parser.add_argument("--models", nargs="+", default=["gpt-neo-1.3B"], 
                      help="Models to evaluate (default: gpt-neo-1.3B)")
    parser.add_argument("--use-samples", action="store_true", default=True,
                      help="Use pre-generated samples instead of real-time generation")
    
    args = parser.parse_args()
    
    # Set up directories
    if args.action in ["setup", "all"]:
        setup_directories()
    
    # Run bias tests
    if args.action in ["test", "all"]:
        run_bias_tests(models=args.models, use_samples=args.use_samples)
    
    # Run dataset evaluation
    if args.action in ["dataset", "all"]:
        run_dataset_evaluation(models=args.models, use_samples=args.use_samples)
    
    # Run analysis
    if args.action in ["analyze", "all"]:
        run_analysis()
    
    print("Bias detection project workflow completed.")
    logging.info("Bias detection project workflow completed.")

if __name__ == "__main__":
    main()