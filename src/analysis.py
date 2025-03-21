"""
Functions for analyzing bias detection results.
"""
import os
import json
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    filename='analysis.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_all_results(directory="data/results", pattern=None):
    """Load all evaluation results from the specified directory."""
    all_results = []
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Check if directory is empty
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Warning: No results found in {directory}")
        return all_results
    
    for filename in os.listdir(directory):
        # Look for both patterns: "evaluation_*.json" and "evaluated_results_*.json"
        if (filename.startswith("evaluation_") or filename.startswith("evaluated_results_")) and filename.endswith(".json"):
            try:
                with open(os.path.join(directory, filename), "r") as f:
                    result = json.load(f)
                    # Add filename to track source
                    result["_source_file"] = filename
                    all_results.append(result)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file {filename}. Skipping.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(all_results)} results from {directory}.")
    return all_results

def analyze_bias_by_category(results, bias_type="gender", category="profession"):
    """Analyze bias scores across different categories (professions, socioeconomic contexts, etc.)."""
    category_scores = {}
    
    # Handle both list and dict results
    if isinstance(results, list):
        for result in results:
            # Check for single model results
            for model_name, model_data in result.items():
                if model_name == "_source_file":
                    continue
                
                if isinstance(model_data, dict) and category in model_data:
                    cat_data = model_data[category]
                    if isinstance(cat_data, dict):
                        for item, data in cat_data.items():
                            if item not in category_scores:
                                category_scores[item] = []
                            
                            bias_key = f"{bias_type}_bias"
                            # Check if bias key exists and is a number
                            if isinstance(data, dict) and bias_key in data:
                                try:
                                    bias_score = float(data[bias_key])
                                    category_scores[item].append(bias_score)
                                except (ValueError, TypeError):
                                    # Skip non-numeric values
                                    pass
    else:
        # Handle direct dictionary format
        for model_name, model_data in results.items():
            if model_name == "_source_file":
                continue
            
            if isinstance(model_data, dict) and category in model_data:
                cat_data = model_data[category]
                if isinstance(cat_data, dict):
                    for item, data in cat_data.items():
                        if item not in category_scores:
                            category_scores[item] = []
                        
                        bias_key = f"{bias_type}_bias"
                        # Check if bias key exists and is a number
                        if isinstance(data, dict) and bias_key in data:
                            try:
                                bias_score = float(data[bias_key])
                                category_scores[item].append(bias_score)
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass
    
    # Calculate average score for each category
    avg_scores = {}
    for item, scores in category_scores.items():
        if scores:  # Check if the list is not empty
            avg_scores[item] = sum(scores) / len(scores)
    
    # Calculate standard deviation for each category
    std_scores = {}
    for item, scores in category_scores.items():
        if len(scores) > 1:  # Need at least 2 points for std dev
            std_scores[item] = np.std(scores)
        elif scores:  # If only one score, set std dev to 0
            std_scores[item] = 0
    
    return avg_scores, std_scores

def analyze_bias_by_model(results, bias_type="gender"):
    """Analyze bias scores across different models."""
    model_scores = {}
    
    # If results is a list, process each result separately
    if isinstance(results, list):
        for result in results:
            for model_name, model_data in result.items():
                if model_name == "_source_file":
                    continue
                    
                if model_name not in model_scores:
                    model_scores[model_name] = []
                
                # Iterate through all categories and items to collect bias scores
                for category in ["profession", "socioeconomic", "age"]:
                    if category in model_data and isinstance(model_data[category], dict):
                        for item, data in model_data[category].items():
                            if isinstance(data, dict):
                                bias_key = f"{bias_type}_bias"
                                if bias_key in data:
                                    try:
                                        bias_score = float(data[bias_key])
                                        model_scores[model_name].append(bias_score)
                                    except (ValueError, TypeError):
                                        # Skip non-numeric values
                                        pass
    # If results is a dict, process directly
    elif isinstance(results, dict):
        for model_name, model_data in results.items():
            if model_name == "_source_file":
                continue
                
            if model_name not in model_scores:
                model_scores[model_name] = []
            
            # Iterate through all categories and items to collect bias scores
            for category in ["profession", "socioeconomic", "age"]:
                if category in model_data and isinstance(model_data[category], dict):
                    for item, data in model_data[category].items():
                        if isinstance(data, dict):
                            bias_key = f"{bias_type}_bias"
                            if bias_key in data:
                                try:
                                    bias_score = float(data[bias_key])
                                    model_scores[model_name].append(bias_score)
                                except (ValueError, TypeError):
                                    # Skip non-numeric values
                                    pass
    
    # Calculate average and std dev for each model
    model_stats = {}
    for model, scores in model_scores.items():
        if scores:  # Check if we have scores for this model
            model_stats[model] = {
                "mean": sum(scores) / len(scores),
                "std": np.std(scores) if len(scores) > 1 else 0,
                "count": len(scores),
                "max": max(scores),
                "min": min(scores),
                "abs_mean": sum(abs(score) for score in scores) / len(scores)  # Average magnitude of bias
            }
    
    return model_stats

def plot_bias_by_category(avg_scores, std_scores=None, bias_type="gender", category_type="profession"):
    """Create a bar chart of bias by category with error bars."""
    # Check if we have data to plot
    if not avg_scores:
        print(f"Warning: No data available to plot {bias_type} bias by {category_type}")
        return
    
    categories = list(avg_scores.keys())
    scores = list(avg_scores.values())
    
    # Sort by score magnitude
    sorted_indices = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # Include top 10 only if there are many categories
    if len(sorted_categories) > 10:
        sorted_categories = sorted_categories[:10]
        sorted_scores = sorted_scores[:10]
        title_prefix = f"Top 10 {category_type.capitalize()} by "
    else:
        title_prefix = ""
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot with or without error bars
        if std_scores and all(cat in std_scores for cat in sorted_categories):
            sorted_stds = [std_scores.get(cat, 0) for cat in sorted_categories]
            bars = plt.bar(
                sorted_categories, 
                sorted_scores, 
                yerr=sorted_stds,
                capsize=5,
                color=['blue' if s > 0 else 'red' for s in sorted_scores]
            )
        else:
            bars = plt.bar(
                sorted_categories, 
                sorted_scores, 
                color=['blue' if s > 0 else 'red' for s in sorted_scores]
            )
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Set title and labels based on bias type
        if bias_type == "gender":
            ylabel = "Bias Score (-1: Male-biased, 1: Female-biased)"
        elif bias_type == "racial":
            ylabel = "Bias Score (-1: White-biased, 1: Minority-biased)"
        elif bias_type == "socioeconomic":
            ylabel = "Bias Score (-1: Wealthy-biased, 1: Poor-biased)"
        elif bias_type == "age":
            ylabel = "Bias Score (-1: Youth-biased, 1: Elderly-biased)"
        else:
            ylabel = "Bias Score"
        
        plt.title(f"{title_prefix}{bias_type.capitalize()} Bias by {category_type.capitalize()}")
        plt.xlabel(f"{category_type.capitalize()}")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/results/figures", exist_ok=True)
        plt.savefig(f"data/results/figures/{bias_type}_bias_by_{category_type}_{timestamp}.png")
        plt.close()
        
        print(f"Created plot for {bias_type} bias by {category_type}")
    except Exception as e:
        print(f"Error creating plot for {bias_type} bias by {category_type}: {e}")
        plt.close()

def plot_bias_comparison_by_model(model_stats, bias_type="gender"):
    """Plot comparison of bias across different models."""
    if not model_stats:
        print(f"Warning: No model stats available to plot {bias_type} bias comparison")
        return
    
    models = list(model_stats.keys())
    
    try:
        # Extract statistics
        means = [model_stats[model]["mean"] for model in models]
        stds = [model_stats[model]["std"] for model in models]
        abs_means = [model_stats[model]["abs_mean"] for model in models]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Mean bias with error bars
        ax1.bar(models, means, yerr=stds, capsize=5, color='skyblue')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_title(f"Mean {bias_type.capitalize()} Bias by Model")
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Mean Bias Score")
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Plot 2: Absolute mean bias (magnitude)
        ax2.bar(models, abs_means, color='coral')
        ax2.set_title(f"Absolute {bias_type.capitalize()} Bias Magnitude by Model")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Mean Absolute Bias Score")
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/results/figures", exist_ok=True)
        plt.savefig(f"data/results/figures/{bias_type}_bias_by_model_{timestamp}.png")
        plt.close()
        
        print(f"Created model comparison plot for {bias_type} bias")
    except Exception as e:
        print(f"Error creating model comparison plot: {e}")
        plt.close()

def create_bias_heatmap(results, bias_types=None, category="profession"):
    """Create a heatmap showing multiple bias dimensions for each category."""
    if bias_types is None:
        bias_types = ["gender", "racial", "socioeconomic", "age"]
    
    # Extract data
    data = {}
    for bias_type in bias_types:
        avg_scores, _ = analyze_bias_by_category(results, bias_type, category)
        for item, score in avg_scores.items():
            if item not in data:
                data[item] = {}
            data[item][bias_type] = score
    
    # Check if we have any data to visualize
    if not data:
        logging.warning(f"No data available to create heatmap for {category}")
        print(f"Warning: No data available to create heatmap for {category}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Check if the DataFrame is empty or has only NaN values
    if df.empty or df.isna().all().all():
        logging.warning(f"Empty DataFrame for {category}, cannot create heatmap")
        print(f"Warning: Empty DataFrame for {category}, cannot create heatmap")
        return None
    
    # If there are more than 15 items, select the top 15 with highest absolute bias
    if len(df) > 15:
        # Calculate total absolute bias across all dimensions
        df['total_abs_bias'] = df.abs().sum(axis=1)
        df = df.sort_values('total_abs_bias', ascending=False).head(15)
        df = df.drop(columns=['total_abs_bias'])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    try:
        sns.heatmap(df, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=.5)
        plt.title(f"Multi-dimensional Bias Analysis by {category.capitalize()}")
        plt.ylabel(f"{category.capitalize()}")
        plt.xlabel("Bias Dimension")
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/results/figures", exist_ok=True)
        plt.savefig(f"data/results/figures/bias_heatmap_{category}_{timestamp}.png")
        plt.close()
        
        logging.info(f"Created heatmap for {category}")
        print(f"Created heatmap for {category}")
        return df
    except ValueError as e:
        logging.error(f"Error creating heatmap for {category}: {e}")
        print(f"Error creating heatmap for {category}: {e}")
        plt.close()
        return None
    except Exception as e:
        logging.error(f"Unexpected error creating heatmap for {category}: {e}")
        print(f"Unexpected error creating heatmap for {category}: {e}")
        plt.close()
        return None

def analyze_intersectional_bias(results, primary_bias="gender", secondary_bias="racial"):
    """Analyze intersectional bias (e.g., how gender bias correlates with racial bias)."""
    bias_pairs = []
    
    # Extract paired bias scores from list of results
    if isinstance(results, list):
        for result in results:
            for model_name, model_data in result.items():
                if model_name == "_source_file":
                    continue
                
                for category in ["profession", "socioeconomic", "age"]:
                    if category in model_data and isinstance(model_data[category], dict):
                        for item, data in model_data[category].items():
                            if isinstance(data, dict):
                                primary_key = f"{primary_bias}_bias"
                                secondary_key = f"{secondary_bias}_bias"
                                
                                if primary_key in data and secondary_key in data:
                                    try:
                                        primary_score = float(data[primary_key])
                                        secondary_score = float(data[secondary_key])
                                        bias_pairs.append({
                                            "category": category,
                                            "item": item,
                                            primary_bias: primary_score,
                                            secondary_bias: secondary_score
                                        })
                                    except (ValueError, TypeError):
                                        # Skip non-numeric values
                                        pass
    # Extract paired bias scores from direct dictionary
    elif isinstance(results, dict):
        for model_name, model_data in results.items():
            if model_name == "_source_file":
                continue
            
            for category in ["profession", "socioeconomic", "age"]:
                if category in model_data and isinstance(model_data[category], dict):
                    for item, data in model_data[category].items():
                        if isinstance(data, dict):
                            primary_key = f"{primary_bias}_bias"
                            secondary_key = f"{secondary_bias}_bias"
                            
                            if primary_key in data and secondary_key in data:
                                try:
                                    primary_score = float(data[primary_key])
                                    secondary_score = float(data[secondary_key])
                                    bias_pairs.append({
                                        "category": category,
                                        "item": item,
                                        primary_bias: primary_score,
                                        secondary_bias: secondary_score
                                    })
                                except (ValueError, TypeError):
                                    # Skip non-numeric values
                                    pass
    
    # Check if we have enough data points
    if len(bias_pairs) < 2:
        print(f"Not enough data points to analyze correlation between {primary_bias} and {secondary_bias}")
        return 0, None
    
    # Convert to DataFrame
    df = pd.DataFrame(bias_pairs)
    
    # Create scatter plot
    try:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x=primary_bias, y=secondary_bias, hue="category", alpha=0.7)
        
        # Add trend line if there are enough points
        if len(df) >= 3:
            sns.regplot(data=df, x=primary_bias, y=secondary_bias, scatter=False, line_kws={"color": "red"})
        
        # Calculate correlation if there are enough points
        correlation = df[primary_bias].corr(df[secondary_bias]) if len(df) >= 3 else 0
        
        plt.title(f"Correlation between {primary_bias.capitalize()} and {secondary_bias.capitalize()} Bias\nCorrelation: {correlation:.3f}")
        plt.xlabel(f"{primary_bias.capitalize()} Bias")
        plt.ylabel(f"{secondary_bias.capitalize()} Bias")
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/results/figures", exist_ok=True)
        plt.savefig(f"data/results/figures/intersectional_{primary_bias}_{secondary_bias}_{timestamp}.png")
        plt.close()
        
        print(f"Created intersectional plot for {primary_bias} and {secondary_bias}")
        return correlation, df
    
    except Exception as e:
        print(f"Error creating intersectional bias plot: {e}")
        plt.close()
        return 0, None

def generate_summary_report(results, output_file="data/results/summary_report.md"):
    """Generate a comprehensive summary report of bias analysis."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bias_types = ["gender", "racial", "socioeconomic", "age"]
    
    try:
        with open(output_file, "w") as f:
            # Header
            f.write(f"# Bias Analysis Summary Report\n\n")
            f.write(f"Generated on: {timestamp}\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            num_results = len(results) if isinstance(results, list) else 1
            f.write(f"This report summarizes the analysis of {num_results} bias evaluation results.\n\n")
            
            # Summary by bias type
            f.write("## Bias Summary by Type\n\n")
            f.write("| Bias Type | Mean Score | Std Dev | Min | Max | Abs Mean |\n")
            f.write("|-----------|------------|---------|-----|-----|----------|\n")
            
            for bias_type in bias_types:
                all_scores = []
                # Process list of results
                if isinstance(results, list):
                    for result in results:
                        for model_name, model_data in result.items():
                            if model_name == "_source_file":
                                continue
                            
                            for category in ["profession", "socioeconomic", "age"]:
                                if category in model_data and isinstance(model_data[category], dict):
                                    for item, data in model_data[category].items():
                                        if isinstance(data, dict):
                                            bias_key = f"{bias_type}_bias"
                                            if bias_key in data:
                                                try:
                                                    score = float(data[bias_key])
                                                    all_scores.append(score)
                                                except (ValueError, TypeError):
                                                    pass
                # Process direct dictionary
                elif isinstance(results, dict):
                    for model_name, model_data in results.items():
                        if model_name == "_source_file":
                            continue
                        
                        for category in ["profession", "socioeconomic", "age"]:
                            if category in model_data and isinstance(model_data[category], dict):
                                for item, data in model_data[category].items():
                                    if isinstance(data, dict):
                                        bias_key = f"{bias_type}_bias"
                                        if bias_key in data:
                                            try:
                                                score = float(data[bias_key])
                                                all_scores.append(score)
                                            except (ValueError, TypeError):
                                                pass
                
                if all_scores:
                    mean = sum(all_scores) / len(all_scores)
                    std_dev = np.std(all_scores) if len(all_scores) > 1 else 0
                    abs_mean = sum(abs(score) for score in all_scores) / len(all_scores)
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    
                    f.write(f"| {bias_type.capitalize()} | {mean:.3f} | {std_dev:.3f} | {min_score:.3f} | {max_score:.3f} | {abs_mean:.3f} |\n")
                else:
                    f.write(f"| {bias_type.capitalize()} | N/A | N/A | N/A | N/A | N/A |\n")
            
            f.write("\n")
            
            # Most biased professions
            f.write("## Most Biased Professions\n\n")
            for bias_type in bias_types:
                avg_scores, _ = analyze_bias_by_category(results, bias_type, "profession")
                if avg_scores:
                    # Sort by absolute bias
                    sorted_professions = sorted(avg_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    f.write(f"### {bias_type.capitalize()} Bias\n\n")
                    f.write("| Profession | Bias Score |\n")
                    f.write("|------------|------------|\n")
                    
                    # Show top 5 or all if less than 5
                    for profession, score in sorted_professions[:min(5, len(sorted_professions))]:
                        bias_direction = "unknown"
                        if bias_type == "gender":
                            bias_direction = "male" if score < 0 else "female" if score > 0 else "neutral"
                        elif bias_type == "racial":
                            bias_direction = "white" if score < 0 else "minority" if score > 0 else "neutral"
                        elif bias_type == "socioeconomic":
                            bias_direction = "wealthy" if score < 0 else "poor" if score > 0 else "neutral"
                        elif bias_type == "age":
                            bias_direction = "youth" if score < 0 else "elderly" if score > 0 else "neutral"
                        
                        f.write(f"| {profession} | {score:.3f} ({bias_direction}-biased) |\n")
                    
                    f.write("\n")
                else:
                    f.write(f"### {bias_type.capitalize()} Bias\n\n")
                    f.write("No data available for this bias type.\n\n")
            
            # Intersectional bias
            f.write("## Intersectional Bias Analysis\n\n")
            f.write("| Primary Bias | Secondary Bias | Correlation |\n")
            f.write("|--------------|----------------|-------------|\n")
            
            has_intersectional_data = False
            for i, primary in enumerate(bias_types):
                for secondary in bias_types[i+1:]:
                    correlation, df = analyze_intersectional_bias(results, primary, secondary)
                    if df is not None and not df.empty:
                        has_intersectional_data = True
                        f.write(f"| {primary.capitalize()} | {secondary.capitalize()} | {correlation:.3f} |\n")
            
            if not has_intersectional_data:
                f.write("| N/A | N/A | N/A |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, here are some recommendations for mitigating bias:\n\n")
            
            # Generate recommendations based on findings
            bias_means = {}
            for bias_type in bias_types:
                all_scores = []
                
                # Process list of results
                if isinstance(results, list):
                    for result in results:
                        for model_name, model_data in result.items():
                            if model_name == "_source_file":
                                continue
                            
                            for category in ["profession", "socioeconomic", "age"]:
                                if category in model_data and isinstance(model_data[category], dict):
                                    for item, data in model_data[category].items():
                                        if isinstance(data, dict):
                                            bias_key = f"{bias_type}_bias"
                                            if bias_key in data:
                                                try:
                                                    score = float(data[bias_key])
                                                    all_scores.append(abs(score))  # Use absolute values
                                                except (ValueError, TypeError):
                                                    pass
                # Process direct dictionary
                elif isinstance(results, dict):
                    for model_name, model_data in results.items():
                        if model_name == "_source_file":
                            continue
                        
                        for category in ["profession", "socioeconomic", "age"]:
                            if category in model_data and isinstance(model_data[category], dict):
                                for item, data in model_data[category].items():
                                    if isinstance(data, dict):
                                        bias_key = f"{bias_type}_bias"
                                        if bias_key in data:
                                            try:
                                                score = float(data[bias_key])
                                                all_scores.append(abs(score))  # Use absolute values
                                            except (ValueError, TypeError):
                                                pass
                
                if all_scores:
                    bias_means[bias_type] = sum(all_scores) / len(all_scores)
            
            if bias_means:
                most_biased_type = max(bias_means.items(), key=lambda x: x[1])[0]
                f.write(f"1. Focus on reducing {most_biased_type} bias, which shows the highest magnitude in the results.\n")
            else:
                f.write("1. Collect more data to establish baseline bias measurements across all dimensions.\n")
            
            f.write("2. Consider intersectional approaches to bias mitigation, as biases often correlate across dimensions.\n")
            f.write("3. Review and augment training data to better represent diverse demographics.\n")
            f.write("4. Implement targeted prompt engineering techniques to reduce bias in model outputs.\n")
            f.write("5. Continue monitoring bias across multiple dimensions, especially for high-risk applications.\n")
        
        print(f"Summary report generated: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error generating summary report: {e}")
        return None

def compare_datasets(results_stereoset, results_crowspairs, bias_type="gender"):
    """Compare bias scores across different evaluation datasets."""
    # Placeholder for dataset comparison implementation
    # Would extract bias scores from different dataset results and compare statistics
    pass

if __name__ == "__main__":
    # Example usage
    print("Loading results...")
    results = load_all_results()
    
    if results:
        print("Analyzing results...")
        for bias_type in ["gender", "racial", "socioeconomic", "age"]:
            # Analyze by profession
            avg_scores, std_scores = analyze_bias_by_category(results, bias_type, "profession")
            if avg_scores:
                plot_bias_by_category(avg_scores, std_scores, bias_type, "profession")
            
            # Analyze by socioeconomic context
            avg_scores, std_scores = analyze_bias_by_category(results, bias_type, "socioeconomic")
            if avg_scores:
                plot_bias_by_category(avg_scores, std_scores, bias_type, "socioeconomic")
            
            # Analyze by age
            avg_scores, std_scores = analyze_bias_by_category(results, bias_type, "age")
            if avg_scores:
                plot_bias_by_category(avg_scores, std_scores, bias_type, "age")
        
        # Create heatmap
        create_bias_heatmap(results, category="profession")
        create_bias_heatmap(results, category="socioeconomic")
        create_bias_heatmap(results, category="age")
        
        # Generate comprehensive report
        generate_summary_report(results)
        
        print("Analysis complete. Results saved to data/results/figures/")
    else:
        print("No results found to analyze. Run bias evaluation first.")