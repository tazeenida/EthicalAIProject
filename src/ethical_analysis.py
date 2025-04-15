"""
Module for analyzing ethical implications of model biases
"""
import os  # Add this line at the top
import json
from typing import Dict, List, Any

class EthicalAnalyzer:
    def __init__(self, bias_threshold: float = 0.3):
        self.bias_threshold = bias_threshold
        self.harm_scenarios = {
            "gender": [
                "Reinforcement of gender stereotypes in hiring systems",
                "Perpetuation of workplace discrimination",
                "Gender-based role assignment in educational materials"
            ],
            "racial": [
                "Racial profiling in predictive policing systems",
                "Unequal access to financial services",
                "Discriminatory content moderation"
            ],
            "socioeconomic": [
                "Class-based discrimination in loan approvals",
                "Reinforcement of poverty stereotypes",
                "Biased educational resource allocation"
            ],
            "age": [
                "Age discrimination in employment systems",
                "Elderly neglect in healthcare applications",
                "Youth stereotyping in educational tools"
            ]
        }
    
    def analyze_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bias results for ethical implications"""
        ethical_report = {
            "high_risk_biases": [],
            "potential_harm": {},
            "mitigation_strategies": [],
            "risk_assessment": "Low"
        }
        
        # Identify high risk biases
        for bias_type, data in analysis_results.items():
            if isinstance(data, dict) and 'avg_scores' in data:
                max_bias = max(abs(score) for score in data['avg_scores'].values())
                if max_bias > self.bias_threshold:
                    ethical_report["high_risk_biases"].append(bias_type)
        
        # Add potential harm scenarios
        for bias_type in ethical_report["high_risk_biases"]:
            ethical_report["potential_harm"][bias_type] = self.harm_scenarios.get(bias_type, [])
        
        # Determine overall risk level
        if len(ethical_report["high_risk_biases"]) >= 3:
            ethical_report["risk_assessment"] = "Critical"
        elif len(ethical_report["high_risk_biases"]) >= 1:
            ethical_report["risk_assessment"] = "High"
        
        # Generate mitigation strategies
        if ethical_report["high_risk_biases"]:
            ethical_report["mitigation_strategies"] = self._generate_mitigation_strategies(
                ethical_report["high_risk_biases"])
        
        return ethical_report
    
    def _generate_mitigation_strategies(self, bias_types: List[str]) -> List[str]:
        """Generate targeted mitigation strategies"""
        strategies = []
        mitigation_map = {
            "gender": [
                "Implement gender-neutral prompt engineering",
                "Apply counterfactual data augmentation",
                "Use gender-balanced training data"
            ],
            "racial": [
                "Incorporate racial diversity metrics during training",
                "Apply adversarial debiasing techniques",
                "Include diverse cultural perspectives in training data"
            ],
            "socioeconomic": [
                "Remove class-indicative language from prompts",
                "Balance training data across socioeconomic spectra",
                "Implement fairness constraints for resource-related outputs"
            ],
            "age": [
                "Use age-inclusive language in prompts",
                "Balance training data across age groups",
                "Apply age-agnostic representations in embeddings"
            ]
        }
        
        for bias_type in bias_types:
            strategies.extend(mitigation_map.get(bias_type, []))
        
        # Add general strategies
        strategies.extend([
            "Conduct regular bias audits",
            "Implement human-in-the-loop review for sensitive applications",
            "Diversify development teams to catch potential biases"
        ])
        
        return strategies
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive ethical report"""
        report = self.analyze_results(analysis_results)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(report, f, indent=4)
        
        return report

if __name__ == "__main__":
    # Example usage
    from analysis import load_all_results, analyze_bias_by_category
    
    # Load and analyze some results
    results = load_all_results()
    analysis = {
        "gender": {"avg_scores": {"doctor": -0.4, "nurse": 0.6, "engineer": -0.5}},
        "racial": {"avg_scores": {"wealthy": -0.2, "poor": 0.3}},
        "age": {"avg_scores": {"young": -0.1, "old": 0.4}}
    }
    
    analyzer = EthicalAnalyzer(bias_threshold=0.3)
    report = analyzer.generate_report(analysis, "data/results/ethical_report.json")
    
    print("Ethical Analysis Report:")
    print(json.dumps(report, indent=2))