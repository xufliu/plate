import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import os
import csv
import time
from sklearn.metrics import accuracy_score, f1_score

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # Set GPUs
torch.cuda.empty_cache()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.padding_side = "left"  # Ensure left-padding for decoder-only models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = BitsAndBytesConfig(load_in_8bit=True)
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    quantization_config=config,
    torch_dtype=torch.float16
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.01)
model = get_peft_model(model, lora_config)

# Helper Functions
def robust_read_csv(filepath):
    cleaned_rows = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        cleaned_rows.append(header[:5])

        for row in reader:
            if len(row) > 5:
                row[4] = ', '.join(row[4:])
                row = row[:5]
            elif len(row) < 5:
                row += [''] * (5 - len(row))
            cleaned_rows.append(row)

    temp_file = filepath + '_cleaned.csv'
    with open(temp_file, 'w', newline='', encoding='utf-8') as temp_out:
        writer = csv.writer(temp_out)
        writer.writerows(cleaned_rows)

    return pd.read_csv(temp_file, quotechar='"', skipinitialspace=True, engine="python")

def clean_explanation(raw_output, plate):
    """
    Extract the explanation specifically for the provided plate.
    """
    lines = raw_output.strip().split("\n")
    target_plate = f"Plate: {plate}"
    capture = False
    explanation = ""

    for line in lines:
        if target_plate in line:
            capture = True  # Start capturing when the target plate is found
        elif capture and "Explanation:" in line:
            explanation = line.split("Explanation:")[1].strip()
            break  # Stop after capturing the relevant explanation

    if not explanation:
        explanation = "No meaningful explanation provided."
    print(f"Raw Output for Plate {plate}:\n{raw_output}")
    #print(f"Extracted Explanation for Plate {plate}:\n{explanation}")


    return explanation


def generate_prompt(plate, predicted_legal):
    """
    Generate a prompt for explanation generation.
    """
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
    return examples + query
    #return query






# Load datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
NY_train = pd.read_csv(os.path.join(script_dir, "NY_train.csv")).sample(20, random_state=42)
NY_test = pd.read_csv(os.path.join(script_dir, "NY_test.csv")).sample(20, random_state=42)
CA_train = pd.read_csv(os.path.join(script_dir, "CA_train.csv")).sample(20, random_state=42)
CA_test = pd.read_csv(os.path.join(script_dir, "CA_test.csv")).sample(20, random_state=42)
red_guide = pd.read_csv(os.path.join(script_dir, "red-guide.csv"))
NY_explanations = robust_read_csv(os.path.join(script_dir, "license_plates_with_explanation_ny.csv"))
CA_explanations = robust_read_csv(os.path.join(script_dir, "license_plates_with_explanation_ca.csv"))


# Ensure 'explanation' column exists in the test datasets
NY_test['explanation'] = ""  # Add an empty explanation column
CA_test['explanation'] = ""  # Add an empty explanation column

# Classification Dataset
class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        plate = row['plate']
        legal = row['legal']
        personalize = row['personalize']
        prompt = (
            f"Plate: {plate}\n"
            f"Legal: Predict whether this plate is legal (0 or 1).\n"
            f"Personalized: Predict whether this plate is personalized (0 or 1)."
        )
        tokenize_output = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenize_output["input_ids"].squeeze(0)
        attention_mask = tokenize_output["attention_mask"].squeeze(0)
        labels = torch.tensor([legal, personalize], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Classification Fine-Tuning
classification_train_data = pd.concat([NY_train, CA_train])
classification_dataset = ClassificationDataset(classification_train_data, tokenizer)
classification_dataloader = DataLoader(classification_dataset, batch_size=8, shuffle=True)

classification_head = torch.nn.Linear(model.config.hidden_size, 2).to(model.device)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classification_head.parameters()), lr=3e-4)
model.train()

# Timing for classification fine-tuning
classification_start_time = time.time()
for epoch in range(3):
    for batch in classification_dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, 0, :]
        logits = classification_head(hidden_states)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Classification Epoch {epoch + 1} completed.")
classification_time = time.time() - classification_start_time
print(f"Classification Fine-Tuning Time: {classification_time:.2f} seconds")

# Predict classification labels for test datasets
def predict_classification(model, tokenizer, test_data, classification_head):
    model.eval()
    predictions = []
    for _, row in test_data.iterrows():
        plate = row['plate']
        prompt = (
            f"Plate: {plate}\n"
            f"Legal: Predict whether this plate is legal (0 or 1).\n"
            f"Personalized: Predict whether this plate is personalized (0 or 1)."
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, 0, :]
        logits = classification_head(hidden_states).sigmoid()
        predicted_legal, predicted_personalize = logits.round().tolist()[0]
        predictions.append({"plate": plate, "predicted_legal": int(predicted_legal), "predicted_personalize": int(predicted_personalize)})
    return pd.DataFrame(predictions)

NY_test_predictions = predict_classification(model, tokenizer, NY_test, classification_head)
CA_test_predictions = predict_classification(model, tokenizer, CA_test, classification_head)

# Explanation Dataset
NY_test_predictions['legal'] = NY_test_predictions['predicted_legal']
CA_test_predictions['legal'] = CA_test_predictions['predicted_legal']
explanation_train_data = pd.concat([NY_explanations, CA_explanations])

# Explanation Fine-Tuning
class ExplanationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        plate = row['plate']
        predicted_legal = row.get('predicted_legal', row['legal'])  # Use predicted legal if available
        prompt = generate_prompt(plate, predicted_legal)

        tokenize_output = tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenize_output["input_ids"].squeeze(0)
        attention_mask = tokenize_output["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


plates = pd.concat([NY_test_predictions['plate'], CA_test_predictions['plate']]).tolist()
explanation_dataset = ExplanationDataset(pd.concat([NY_test_predictions, CA_test_predictions]), tokenizer)
explanation_dataloader = DataLoader(explanation_dataset, batch_size=8, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()

# Timing for explanation fine-tuning
explanation_start_time = time.time()
for epoch in range(3):
    for batch in explanation_dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pad_token_id=tokenizer.pad_token_id, temperature=0.2 )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Explanation Epoch {epoch + 1} completed.")
explanation_time = time.time() - explanation_start_time
print(f"Explanation Fine-Tuning Time: {explanation_time:.2f} seconds")

# Generate explanations for test datasets
explanation_dataset = ExplanationDataset(pd.concat([NY_test_predictions, CA_test_predictions]), tokenizer)
explanation_dataloader = DataLoader(explanation_dataset, batch_size=8, shuffle=False)

def generate_explanations(model, tokenizer, dataloader, plates):
    """
    Generate explanations for test plates based on predicted legal status.
    """
    model.eval()
    explanations = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id
            )
            # Use the clean_explanation function to process each output
            decoded_outputs = [
                clean_explanation(tokenizer.decode(output, skip_special_tokens=True), plates[i * len(batch['input_ids']) + j])
                for j, output in enumerate(outputs)
            ]
            explanations.extend(decoded_outputs)
    return explanations



# Call the function with plates
NY_explanations = generate_explanations(model, tokenizer, explanation_dataloader, plates)

# Save results
NY_test_predictions['explanation'] = NY_explanations[:len(NY_test_predictions)]
CA_test_predictions['explanation'] = NY_explanations[len(NY_test_predictions):]

NY_test_predictions.to_csv("NY_test_results.csv", index=False)
CA_test_predictions.to_csv("CA_test_results.csv", index=False)

# Total execution time
total_time = classification_time + explanation_time
print(f"Total Execution Time: {total_time:.2f} seconds")
print("Results saved.")


from sklearn.metrics import accuracy_score, f1_score

# Extract true labels
true_legal = NY_test['legal'].tolist() + CA_test['legal'].tolist()
true_personalize = NY_test['personalize'].tolist() + CA_test['personalize'].tolist()

# Extract predicted labels
predicted_legal = NY_test_predictions['predicted_legal'].tolist() + CA_test_predictions['predicted_legal'].tolist()
predicted_personalize = NY_test_predictions['predicted_personalize'].tolist() + CA_test_predictions['predicted_personalize'].tolist()

# Calculate metrics for legal
legal_accuracy = accuracy_score(true_legal, predicted_legal)
legal_f1 = f1_score(true_legal, predicted_legal)

# Calculate metrics for personalized
personalize_accuracy = accuracy_score(true_personalize, predicted_personalize)
personalize_f1 = f1_score(true_personalize, predicted_personalize)

# Print results
print(f"Legal Accuracy: {legal_accuracy:.4f}")
print(f"Legal F1 Score: {legal_f1:.4f}")
print(f"Personalized Accuracy: {personalize_accuracy:.4f}")
print(f"Personalized F1 Score: {personalize_f1:.4f}")
