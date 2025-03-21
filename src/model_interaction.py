"""
Module for interacting with language models using Hugging Face Transformers and OpenAI API.
"""
import os
import json
import torch
import requests
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, List, Optional, Union

# Environment variable for API keys (should be set externally)
# os.environ["OPENAI_API_KEY"] = "your-key-here"  # For actual use, never hardcode

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
    "gpt-neo-2.7B": {
        "type": "huggingface",
        "name": "EleutherAI/gpt-neo-2.7B",
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
    }
}

def load_huggingface_model(model_config):
    """Load a Hugging Face model and tokenizer."""
    print(f"Loading model: {model_config['name']}...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    
    # Set the pad_token to the eos_token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    # For large models, use device_map="auto" for efficient loading
    if model_config['name'] in ["EleutherAI/gpt-neo-2.7B", "meta-llama/Llama-2-7b-hf"]:
        model = AutoModelForCausalLM.from_pretrained(model_config['name'], device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config['name'])
    
    return tokenizer, model

def generate_response_huggingface(prompt, model_config, max_length=150, temperature=0.7):
    """Generate a response using a Hugging Face model."""
    # Load model if not already loaded
    if model_config["tokenizer"] is None or model_config["model"] is None:
        tokenizer, model = load_huggingface_model(model_config)
        model_config["tokenizer"] = tokenizer
        model_config["model"] = model
    else:
        tokenizer = model_config["tokenizer"]
        model = model_config["model"]
    
    # Tokenize the input prompt - Fix for GPT Neo
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length + input_ids.shape[1],  # Account for input length
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (not including the prompt)
    if response.startswith(prompt):
        response = response[len(prompt):]
    
    return response.strip()

def generate_response_openai(prompt, model_config, max_tokens=150, temperature=0.7):
    """Generate a response using OpenAI API."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OpenAI API key not found in environment variables."
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
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
            return response.json()["choices"][0]["text"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response(prompt, model_name="gpt-neo-1.3B", max_length=150, temperature=0.7):
    """Generate a response using the specified model."""
    if model_name not in MODEL_CONFIGS:
        return f"Error: Model '{model_name}' not configured."
    
    model_config = MODEL_CONFIGS[model_name]
    
    if model_config["type"] == "huggingface":
        return generate_response_huggingface(prompt, model_config, max_length, temperature)
    elif model_config["type"] == "openai":
        return generate_response_openai(prompt, model_config, max_length, temperature)
    else:
        return f"Error: Unsupported model type '{model_config['type']}'."

def batch_generate_responses(prompts, model_names=None, max_length=150, temperature=0.7):
    """Generate responses for multiple prompts across multiple models."""
    if model_names is None:
        model_names = ["gpt-neo-1.3B"]  # Default to GPT Neo if no models specified
    
    results = {}
    for model_name in model_names:
        print(f"Generating responses using {model_name}...")
        model_results = {}
        
        for prompt in prompts:
            try:
                print(f"  Processing: {prompt[:50]}...")
                response = generate_response(prompt, model_name, max_length, temperature)
                model_results[prompt] = response
            except Exception as e:
                print(f"  Error generating response for prompt: {prompt}. Error: {e}")
                model_results[prompt] = f"Error: {str(e)}"
        
        results[model_name] = model_results
    
    return results

def save_responses(responses, directory="data/responses"):
    """Save model responses to a file."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_responses in responses.items():
        filename = f"{directory}/responses_{model_name}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(model_responses, f, indent=4)
        print(f"Responses for model {model_name} saved to {filename}")

# Add this block to test the functionality when the script is run directly
if __name__ == "__main__":
    print("Testing enhanced model interaction...")
    
    # Example prompts
    test_prompts = [
        "Describe a typical doctor.",
        "Describe a typical nurse.",
        "Describe a typical person from a wealthy neighborhood."
    ]
    
    # Test with GPT Neo to avoid API costs
    responses = batch_generate_responses(test_prompts, ["gpt-neo-1.3B"])
    
    # Save responses
    save_responses(responses)
    
    print("Model interaction test complete.")