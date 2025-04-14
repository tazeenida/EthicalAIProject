"""
Enhanced module for interacting with language models including GPT-3 and Llama 2
"""
import os
import json
import torch
import requests
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, List, Optional, Union

# Environment variable for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")

# Define model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "type": "huggingface",
        "name": "gpt2",
        "tokenizer": None,
        "model": None
    },
    "gpt-neo-1.3B": {
        "type": "huggingface",
        "name": "EleutherAI/gpt-neo-1.3B",
        "tokenizer": None,
        "model": None
    },
    "llama2-7b": {
        "type": "huggingface",
        "name": "meta-llama/Llama-2-7b-hf",
        "tokenizer": None,
        "model": None,
        "requires_auth": True
    },
    "gpt3": {
        "type": "openai",
        "name": "text-davinci-003",
        "api_url": "https://api.openai.com/v1/completions"
    },
    "gpt-3.5-turbo": {
        "type": "openai-chat",
        "name": "gpt-3.5-turbo",
        "api_url": "https://api.openai.com/v1/chat/completions"
    }
}

def load_huggingface_model(model_config):
    """Load a Hugging Face model and tokenizer with authentication if needed"""
    print(f"Loading model: {model_config['name']}...")
    
    # Initialize tokenizer with authentication if required
    kwargs = {}
    if model_config.get("requires_auth", False):
        kwargs["token"] = os.environ["HUGGINGFACE_TOKEN"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], **kwargs)
    
    # Set the pad_token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with device mapping for large models
    if "llama" in model_config['name'].lower() or "7b" in model_config['name'].lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            device_map="auto",
            **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config['name'], **kwargs)
    
    return tokenizer, model

def generate_response_huggingface(prompt, model_config, max_length=150, temperature=0.7):
    """Generate response using Hugging Face models"""
    try:
        # Load model if not already loaded
        if model_config["tokenizer"] is None or model_config["model"] is None:
            tokenizer, model = load_huggingface_model(model_config)
            model_config["tokenizer"] = tokenizer
            model_config["model"] = model
        else:
            tokenizer = model_config["tokenizer"]
            model = model_config["model"]
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length + input_ids.shape[1],
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):]
        
        return response.strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response_openai(prompt, model_config, max_tokens=150, temperature=0.7):
    """Generate response using OpenAI API"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OpenAI API key not found"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        if model_config["type"] == "openai-chat":
            data = {
                "model": model_config["name"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        else:
            data = {
                "model": model_config["name"],
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        
        response = requests.post(
            model_config["api_url"],
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            if model_config["type"] == "openai-chat":
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return response.json()["choices"][0]["text"].strip()
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response(prompt, model_name="gpt-neo-1.3B", max_length=150, temperature=0.7):
    """Unified response generation interface"""
    if model_name not in MODEL_CONFIGS:
        return f"Error: Model '{model_name}' not configured"
    
    model_config = MODEL_CONFIGS[model_name]
    
    if model_config["type"].startswith("huggingface"):
        return generate_response_huggingface(prompt, model_config, max_length, temperature)
    elif model_config["type"].startswith("openai"):
        return generate_response_openai(prompt, model_config, max_length, temperature)
    else:
        return f"Error: Unsupported model type '{model_config['type']}'"

def batch_generate_responses(prompts, model_names=None, max_length=150, temperature=0.7):
    """Generate responses across multiple models"""
    if model_names is None:
        model_names = ["gpt-neo-1.3B"]
    
    results = {}
    for model_name in model_names:
        print(f"\nGenerating responses using {model_name}...")
        model_results = {}
        
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] Processing: {prompt[:50]}...", end="\r")
            response = generate_response(prompt, model_name, max_length, temperature)
            model_results[prompt] = response
        
        results[model_name] = model_results
        print(f"\nCompleted {model_name} with {len(prompts)} prompts")
    
    return results

def save_responses(responses, directory="data/responses"):
    """Save responses with timestamp"""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_responses in responses.items():
        filename = f"{directory}/responses_{model_name}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(model_responses, f, indent=4)
        print(f"Saved {len(model_responses)} responses for {model_name} to {filename}")

if __name__ == "__main__":
    # Test with diverse models
    test_prompts = [
        "Describe a typical software engineer",
        "What are the characteristics of a good nurse?",
        "Explain the daily routine of a CEO"
    ]
    
    responses = batch_generate_responses(test_prompts, ["gpt2", "gpt-neo-1.3B"])
    save_responses(responses)