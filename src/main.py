"""
Enhanced main script integrating all new components
"""
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Import all modules
from bias_evaluation import (
    generate_profession_prompts,
    generate_socioeconomic_prompts,
    generate_age_prompts,
    evaluate_text_for_all_biases,
    load_dataset
)
from model_interaction import batch_generate_responses, save_responses
from analysis import (
    load_all_results,
    analyze_bias_by_category,
    plot_bias_by_category,
    create_bias_heatmap,
    generate_summary_report
)
from debiasing import DebiasingEngine
from ethical_analysis import EthicalAnalyzer
from qualitative_analysis import analyze_response_patterns

def setup_directories():
    """Create required directory structure"""
    dirs = [
        "data",
        "data/responses",
        "data/results",
        "data/results/figures",
        "data/datasets",
        "data/debiased"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_comprehensive_evaluation(models: List[str] = None, 
                               use_datasets: bool = True,
                               run_debiasing: bool = True):
    """Run complete evaluation pipeline"""
    if models is None:
        models = ["gpt2", "gpt-neo-1.3B"]
    
    print("\n=== Starting Comprehensive Bias Evaluation ===")
    
    # 1. Generate evaluation prompts
    print("\nGenerating evaluation prompts...")
    prompts = (
        generate_profession_prompts() +
        generate_socioeconomic_prompts() +
        generate_age_prompts()
    )
    print(f"Generated {len(prompts)} evaluation prompts")
    
    # 2. Generate model responses
    print("\nGenerating model responses...")
    responses = batch_generate_responses(prompts, models)
    save_responses(responses)
    
    # 3. Dataset evaluation
    if use_datasets:
        print("\nRunning dataset evaluations...")
        dataset_results = evaluate_with_datasets(models)
        save_responses(dataset_results, "data/responses/datasets")
    
    # 4. Debiasing evaluation
    debiasing_results = {}
    if run_debiasing:
        print("\nRunning debiasing evaluation...")
        debiaser = DebiasingEngine()
        for model_name, model_responses in responses.items():
            debiasing_results[model_name] = debiaser.evaluate_debiasing(model_responses)
        
        # Save debiasing results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"data/debiased/debiasing_results_{timestamp}.json", "w") as f:
            json.dump(debiasing_results, f, indent=4)
    
    # 5. Comprehensive analysis
    print("\nRunning comprehensive analysis...")
    analysis_results = {}
    all_results = load_all_results()
    
    for bias_type in ["gender", "racial", "socioeconomic", "age"]:
        analysis_results[bias_type] = {}
        for category in ["profession", "socioeconomic", "age"]:
            avg_scores, std_scores = analyze_bias_by_category(
                all_results, bias_type, category)
            analysis_results[bias_type][category] = {
                "avg_scores": avg_scores,
                "std_scores": std_scores
            }
            
            # Generate visualizations
            plot_bias_by_category(avg_scores, std_scores, bias_type, category)
    
    # Generate heatmaps
    for category in ["profession", "socioeconomic", "age"]:
        create_bias_heatmap(all_results, category=category)
    
    # 6. Qualitative analysis
    print("\nRunning qualitative analysis...")
    qualitative_results = {}
    for model_name, model_responses in responses.items():
        qualitative_results[model_name] = analyze_response_patterns(model_responses)
    
    # 7. Ethical analysis
    print("\nRunning ethical analysis...")
    ethical_analyzer = EthicalAnalyzer()
    ethical_report = ethical_analyzer.generate_report(
        analysis_results,
        "data/results/ethical_report.json"
    )
    
    # 8. Generate final reports
    print("\nGenerating final reports...")
    generate_summary_report(all_results)
    
    print("\n=== Evaluation Complete ===")
    return {
        "responses": responses,
        "analysis": analysis_results,
        "debiasing": debiasing_results,
        "qualitative": qualitative_results,
        "ethical": ethical_report
    }

def evaluate_with_datasets(models: List[str]) -> Dict[str, Dict[str, str]]:
    """Evaluate models using standard datasets"""
    dataset_prompts = {}
    
    # Load datasets and create prompts
    for dataset_name in ["stereoset", "crows_pairs"]:
        data = load_dataset(dataset_name)
        if dataset_name == "stereoset":
            prompts = [item["sentence"] for item in data]
        else:  # crowspairs
            prompts = []
            for item in data:
                prompts.append(item["stereotype"])
                prompts.append(item["anti_stereotype"])
        
        dataset_prompts[dataset_name] = prompts
    
    # Generate responses for all dataset prompts
    all_prompts = []
    for prompts in dataset_prompts.values():
        all_prompts.extend(prompts)
    
    return batch_generate_responses(all_prompts, models)

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Bias Evaluation for Language Models")
    
    parser.add_argument("--models", nargs="+", default=["gpt-neo-1.3B"],
                      help="Models to evaluate")
    parser.add_argument("--skip-datasets", action="store_true",
                      help="Skip dataset evaluations")
    parser.add_argument("--skip-debiasing", action="store_true",
                      help="Skip debiasing evaluation")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_directories()
    
    # Run evaluation
    results = run_comprehensive_evaluation(
        models=args.models,
        use_datasets=not args.skip_datasets,
        run_debiasing=not args.skip_debiasing
    )
    
    print("\nEvaluation results saved in data/results/")
    print("Ethical report saved to data/results/ethical_report.json")

if __name__ == "__main__":
    main()