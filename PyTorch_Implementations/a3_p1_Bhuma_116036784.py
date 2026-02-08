from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding,AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

output_dir = "fine_tuned_boolq"
os.makedirs(output_dir, exist_ok=True)
boolq_dataset = load_dataset('google/boolq')
valida = boolq_dataset['validation']
train = boolq_dataset['train']

def zeroshot_on_boolq(model, tokenizer, dataset, device):
    """
    Evaluates DistilGPT2 model on the BoolQ validation dataset.
    Prints overall accuracy, macro F1, and per-class precision/recall/F1.
    
    Args:
        val_data: validation split of the BoolQ dataset
    """
    # Load model and tokenizer
    model.eval()
    tokenizer.padding_side = "left"

    # Token IDs for "yes" and "no"
    y_token = tokenizer.encode("yes", add_special_tokens=False)[0]
    n_token = tokenizer.encode("no", add_special_tokens=False)[0]

    predictions = []
    true_labels = []

    # Loop through dataset
    for example in tqdm(dataset, desc="Evaluating DistilGPT2 on BoolQ"):
        passage = example["passage"]
        question = example["question"]
        label = example["answer"]

        prompt = passage.strip() + "\n" + question.strip() + "?\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

            logits = outputs.logits[0, -1, :]  # only last token

            y_prob = torch.softmax(logits, dim=-1)[y_token].item()
            n_prob = torch.softmax(logits, dim=-1)[n_token].item()

            prediction = 1 if y_prob > n_prob else 0

        predictions.append(prediction)
        true_labels.append(1 if label else 0)

    # Calculate metrics
    overall_acc = accuracy_score(true_labels, predictions)
    overall_f1 = f1_score(true_labels, predictions, average='macro')

    # Per-class metrics
    yes_prec = precision_score(true_labels, predictions, pos_label=1)
    yes_rec = recall_score(true_labels, predictions, pos_label=1)
    yes_f1 = f1_score(true_labels, predictions, pos_label=1)

    no_prec = precision_score(true_labels, predictions, pos_label=0)
    no_rec = recall_score(true_labels, predictions, pos_label=0)
    no_f1 = f1_score(true_labels, predictions, pos_label=0)

    # Final Output
    print(f"\n=== DistilGPT2 Evaluation on BoolQ ===")
    print(f"Overall: acc: {overall_acc:.3f}, f1: {overall_f1:.3f}")
    print(f"Yes: prec: {yes_prec:.3f}, rec: {yes_rec:.3f}, f1: {yes_f1:.3f}")
    print(f"No:  prec: {no_prec:.3f}, rec: {no_rec:.3f}, f1: {no_f1:.3f}")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

zeroshot_on_boolq(model,tokenizer, valida, device )


def fine_tune_distilgpt2_on_boolq(train_dataset, model, device,tokenizer, output_dir="fine_tuned_boolq", batch_size=8, epochs=1, lr=1e-5, weight_decay=1e-2):
    """
    Fine-tune DistilGPT2 on BoolQ dataset (passage + question --> yes/no).
    Also saves the model and plots the training loss curve.

    Args:
        train_dataset: The BoolQ training dataset.
        model: Preloaded DistilGPT2 model (AutoModelForCausalLM).
        device: torch.device (cuda or cpu).
        output_dir: Directory to save model/tokenizer after fine-tuning.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        losses: List of training losses.
    """

    model.train()
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    # Preprocessing function
    def preprocess(example):
        passage = example["passage"]
        question = example["question"]
        answer = example["answer"]
        prompt = f"{passage.strip()}\n{question.strip()}?\n{'yes' if answer else 'no'}"
        return tokenizer(prompt, truncation=True, padding=True)

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, "loading....."):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            epoch_loss += loss.item()

            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")
        losses.append(avg_epoch_loss)

    # Plotting Loss Curve
    plt.figure(figsize=(10,6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Curve during Language Model Fine-tuning')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss_curve.png')
    plt.show()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Saved fine‑tuned model and tokenizer to {output_dir}")

    return model, tokenizer, losses
fine_tuned_model, fine_tuned_tok, losses_gpt2 = fine_tune_distilgpt2_on_boolq(
    train, 
    model=model,
    device=device,
    tokenizer= tokenizer,
    output_dir="fine_tuned_boolq",
    batch_size=8,
    epochs=1
)


from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

def evaluate_fine_tuned_gpt2_on_boolq(model,tokenizer, dataset, device):
    """
    Evaluate a fine-tuned DistilGPT2 model on BoolQ validation set.

    Args:
        model_dir: directory where fine-tuned model is saved ("fine_tuned_boolq")
        valida_dataset: validation split (Huggingface Dataset object)
        device: torch.device ("cuda" or "cpu")

    Prints:
        Overall Accuracy, Macro F1, Precision/Recall/F1 for "yes" and "no"
    """

 
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating"):
            passage = example['passage']
            question = example['question']
            label = example['answer']

            prompt = passage.strip() + "\n" + question.strip() + "?\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

            output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_token_id = output[0, -1]
            generated_token = tokenizer.decode(generated_token_id, skip_special_tokens=True).strip().lower()

            pred = 1 if generated_token.startswith("yes") else 0
            preds.append(pred)
            labels.append(1 if label else 0)

    # Calculate overall metrics
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')

    # Per class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[1, 0], zero_division=0
    )

    # Print results
    print(f"Overall: acc: {accuracy:.3f}, f1: {macro_f1:.3f}") 
    print(f"    Yes: prec: {precision[0]:.3f}, rec: {recall[0]:.3f},  f1: {f1[0]:.3f}")
    print(f"     No: prec: {precision[1]:.3f}, rec: {recall[1]:.3f},  f1: {f1[1]:.3f}")


evaluate_fine_tuned_gpt2_on_boolq(
    model = fine_tuned_model,
    tokenizer= fine_tuned_tok,
    dataset=valida,
    device=device
)

def fine_tune_distilroberta_on_boolq(train_dataset, model, tokenizer, device, num_epochs=1, batch_size=8, lr=1e-5, weight_decay=1e-3):
    """
    Fine-tunes DistilRoberta on BoolQ dataset for classification task.

    Args:
        train_dataset: BoolQ "train" split (Huggingface dataset)
        model_name: huggingface model ID (default = "distilroberta-base")
        device: cuda/cpu device
        num_epochs: number of epochs
        batch_size: mini-batch size
        lr: learning rate
        weight_decay: weight decay regularization

    Returns:
        fine-tuned model, tokenizer, losses (list of training losses)
    """

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Preprocessing
    def preprocess(example):
        passage = example["passage"]
        question = example["question"]
        prompt = f"{passage.strip()}\n{question.strip()}?\n"
        encoded = tokenizer(prompt, truncation=True, padding=True)
        encoded["labels"] = 1 if example["answer"] else 0
        return encoded

    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            epoch_loss += loss.item()

            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(8,6))
    plt.plot(losses, label='Training Loss', linewidth=2)
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve for distilroberta on BoolQ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, tokenizer, losses

model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels = 2).to(device)
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
tokenizer.pad_token = tokenizer.eos_token

model_roberta, tokenizer_roberta, losses_roberta = fine_tune_distilroberta_on_boolq(
    train,
    model = model,
    tokenizer = tokenizer,
    device=device,
    num_epochs=1,
    batch_size=8
)


from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def evaluate_distilroberta_on_boolq(model, tokenizer, val_dataset, device):
    """
    Evaluates a fine-tuned DistilRoberta model on BoolQ validation set.
    
    Args:
        model: fine-tuned model
        tokenizer: corresponding tokenizer
        val_dataset: validation dataset
        device: cuda/cpu device

    Returns:
        Prints overall and class-wise precision, recall, F1
    """

    model.eval()
    preds = []
    labels = []

    for example in val_dataset:
        passage = example["passage"]
        question = example["question"]
        label = example["answer"]

        prompt = f"{passage.strip()}\n{question.strip()}?\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        cls = torch.argmax(logits, dim=-1).item()
        preds.append("yes" if cls == 1 else "no")
        labels.append("yes" if label else "no")

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    prec, rec, f1s, _ = precision_recall_fscore_support(
        labels, preds, labels=["yes", "no"], zero_division=0
    )

    print(f"Overall: acc: {acc:.3f}, f1: {macro_f1:.3f}")
    print(f"    Yes: prec: {prec[0]:.3f}, rec: {rec[0]:.3f},  f1: {f1s[0]:.3f}")
    print(f"     No: prec: {prec[1]:.3f}, rec: {rec[1]:.3f},  f1: {f1s[1]:.3f}")

    return acc, macro_f1, prec, rec, f1s

evaluate_distilroberta_on_boolq(
    model=model_roberta,
    tokenizer=tokenizer_roberta,
    val_dataset=valida,
    device=device
)

# Example usage
# from datasets import load_dataset
# boolq_dataset = load_dataset("google/boolq")
# train, validation, test = get_boolq_splits()
# evaluate_distilgpt2_on_boolq(validation)

# # Initialize model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

# # Fine-tune
# losses_gpt2 = fine_tune_distilgpt2_on_boolq(
#     train,
#     model=model_gpt2,
#     device=device,
#     output_dir="fine_tuned_boolq",
#     batch_size=8,
#     epochs=1
# )

# evaluate_fine_tuned_gpt2_on_boolq(
#     model_dir="fine_tuned_boolq",
#     valida_dataset=valida,
#     device=device
# )

# # Fine-tune
# model_roberta, tokenizer_roberta, losses_roberta = fine_tune_distilroberta_on_boolq(
#     train,
#     model_name="distilroberta-base",
#     device=device,
#     num_epochs=1,
#     batch_size=8
# )

# evaluate_distilroberta_on_boolq(
#     model=model_roberta,
#     tokenizer=tokenizer_roberta,
#     val_dataset=valida,
#     device=device
# )



