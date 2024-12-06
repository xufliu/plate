import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from sklearn.metrics import accuracy_score, f1_score

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for now
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # Set GPUs
torch.cuda.empty_cache()

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

# Helper Functions
def generate_prompt(plate, predicted_legal=None):
    examples = (
        "Examples:\n"
        "Plate: BKWDOW\n"
        "Legal Status: 1 (0 for Illegal, 1 for Legal)\n"
        "Explanation: This plate means 'Black Widow', referring to the spider or perhaps a nickname.\n"
        "---\n"
        "Plate: 2FAST4U\n"
        "Legal Status: 1 (0 for Illegal, 1 for Legal)\n"
        "Explanation: This plate means 'Too Fast for You', a playful boast about speed.\n"
        "###\n"
    )
    query = (
        f"Now, analyze the following plate and provide its intended meaning:\n"
        f"Plate: {plate}\n"
        f"Legal Status: {predicted_legal} (0 for Illegal, 1 for Legal)\n"
        "Explanation:"
    )
    #return examples + query
    return query

def clean_explanation(raw_output, plate):
    print(raw_output)
    lines = raw_output.strip().split("\n")
    target_plate = f"Plate: {plate}"
    capture = False
    explanation = ""

    for line in lines:
        if target_plate in line:
            capture = True
        elif capture and "Explanation:" in line:
            explanation = line.split("Explanation:")[1].strip()
            break

    return explanation if explanation else "No meaningful explanation provided."

def inference(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Inference error with prompt: {prompt}\nError: {e}")
        return "Error in inference"

# Load test datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
NY_test = pd.read_csv(os.path.join(script_dir, "NY_test.csv")).sample(20, random_state=42)
CA_test = pd.read_csv(os.path.join(script_dir, "CA_test.csv")).sample(20, random_state=42)

def classify_and_explain(model, tokenizer, test_data):
    predictions = []
    explanations = []
    start_time = time.time()

    for _, row in test_data.iterrows():
        plate = row["plate"]
        try:
            # Classification prompt
            classification_prompt = (
                f"Plate: {plate}\n"
                f"Legal: Predict whether this plate is legal (0 for Illegal, 1 for Legal).\n"
                f"Personalized: Predict whether this plate is personalized (0 for Not Personalized, 1 for Personalized).\n"
            )
            classification_output = inference(model, tokenizer, classification_prompt)
            predicted_legal = 1 if "Legal: 1" in classification_output else 0
            predicted_personalize = 1 if "Personalized: 1" in classification_output else 0

            predictions.append({
                "plate": plate,
                "predicted_legal": predicted_legal,
                "predicted_personalize": predicted_personalize,
            })

            # Explanation prompt
            explanation_prompt = generate_prompt(plate, predicted_legal)
            explanation_output = inference(model, tokenizer, explanation_prompt)
            explanation = clean_explanation(explanation_output, plate)

            explanations.append(explanation)

        except Exception as e:
            print(f"Error processing plate {plate}: {e}")
            predictions.append({
                "plate": plate,
                "predicted_legal": -1,
                "predicted_personalize": -1,
            })
            explanations.append("Error in generating explanation")

    classification_time = time.time() - start_time
    print(f"Classification and Explanation Time: {classification_time:.2f} seconds")

    results = pd.DataFrame(predictions)
    results["explanation"] = explanations
    return results

# Run classification and explanation for both datasets
NY_results = classify_and_explain(model, tokenizer, NY_test)
CA_results = classify_and_explain(model, tokenizer, CA_test)

# Save results
NY_results.to_csv("NY_test_results.csv", index=False)
CA_results.to_csv("CA_test_results.csv", index=False)

# Metrics
true_legal = NY_test["legal"].tolist() + CA_test["legal"].tolist()
true_personalize = NY_test["personalize"].tolist() + CA_test["personalize"].tolist()

predicted_legal = NY_results["predicted_legal"].tolist() + CA_results["predicted_legal"].tolist()
predicted_personalize = NY_results["predicted_personalize"].tolist() + CA_results["predicted_personalize"].tolist()

legal_accuracy = accuracy_score(true_legal, predicted_legal)
legal_f1 = f1_score(true_legal, predicted_legal, average="macro")

personalize_accuracy = accuracy_score(true_personalize, predicted_personalize)
personalize_f1 = f1_score(true_personalize, predicted_personalize, average="macro")

# Print metrics
print(f"Legal Accuracy: {legal_accuracy:.4f}")
print(f"Legal F1 Score: {legal_f1:.4f}")
print(f"Personalized Accuracy: {personalize_accuracy:.4f}")
print(f"Personalized F1 Score: {personalize_f1:.4f}")
