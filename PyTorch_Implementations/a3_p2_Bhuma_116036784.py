# Used different layout for the code than what i do usually because it was very clusterd 

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaLayer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time




start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------Modifying Transformer-------------------------------------------#
#---------------------------------------------------2.1-----------------------------------------------------------------

def distilRB_rand(model):
    for layer in tqdm(model.roberta.encoder.layer, desc="Randomizing DistilRoBERTa"):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    return model

def distilRB_KQV(model):
    base = model.base_model
    if hasattr(base, 'transformer'):
        layers = base.transformer.layer
    elif hasattr(base, 'encoder'):
        layers = base.encoder.layer
    else:
        raise ValueError("Cannot locate transformer layers.")

    for i in tqdm([-2, -1], desc="Sharing QKV layers (distilRB-KQV)"):
        layer = layers[i]
        attn_proj = layer.attention.self
        q_attr, k_attr, v_attr = 'query', 'key', 'value'
        q_w = getattr(attn_proj, q_attr).weight.data
        k_w = getattr(attn_proj, k_attr).weight.data
        shared_w = (q_w + k_w) / 2
        in_f = getattr(attn_proj, q_attr).in_features
        out_f = getattr(attn_proj, q_attr).out_features

        shared_lin = nn.Linear(in_f, out_f, bias=True).to(model.device)
        shared_lin.weight.data.copy_(shared_w)
        shared_lin.bias.data.zero_()

        setattr(attn_proj, q_attr, shared_lin)
        setattr(attn_proj, k_attr, shared_lin)
        setattr(attn_proj, v_attr, shared_lin)

    return model

class distilRB_nores(RobertaLayer):#------------------------------------Took CHATGPTS help for this creating this class#
    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        self_attn_outputs = self.attention(hidden_states, attention_mask, head_mask=layer_head_mask,
                                           encoder_hidden_states=encoder_hidden_states,
                                           encoder_attention_mask=encoder_attention_mask,
                                           past_key_value=past_key_value,
                                           output_attentions=output_attentions)
        attention_output = self_attn_outputs[0]
        attention_output = self.attention.output.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output.dense(intermediate_output)
        layer_output = self.output.LayerNorm(layer_output)
        outputs = (layer_output,)
        if output_attentions:
            outputs += (self_attn_outputs[1],)
        return outputs

def replace_last_two_layers_with_nores(model):
    print("Replacing last 2 layers with RobertaLayerNoRes...")
    for idx in tqdm([-2, -1], desc="Replacing last layers (no residual)"):
        old_layer = model.roberta.encoder.layer[idx]
        new_layer = distilRB_nores(model.config)
        new_layer.load_state_dict(old_layer.state_dict(), strict=False)
        model.roberta.encoder.layer[idx] = new_layer.to(model.device)
#-----------------------------------------------------------end of 2.1--------------------------------------------------

#---------------------------------------------------------------Dataloader functions-------------------------------------
def get_boolq_dataloaders(batch_size=32):
    dataset = load_dataset("google/boolq")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def preprocess(batch):
        texts = [f"{p}\n{q}?\n" for p, q in zip(batch["passage"], batch["question"])]
        enc = tokenizer(texts, truncation=True, padding=True)
        enc["labels"] = [int(a) for a in batch["answer"]]
        return enc

    dataset = dataset.map(preprocess, batched=True, remove_columns=["passage", "question", "answer"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader

def get_sst_dataloaders(batch_size=16):
    dataset = load_dataset("stanfordnlp/sst", "default", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def preprocess(batch):
        enc = tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = batch["label"]
        return enc

    dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence", "tokens", "tree"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------Fine Tuning----------------------------------------------------

def fine_tune_classification(model, loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    model.train()
    for batch in tqdm(loader, desc="Training Classification"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
    return model

def fine_tune_regression(model, loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    mse_loss = nn.MSELoss()
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training Regression"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = outputs.logits.squeeze(-1)
            loss = mse_loss(preds, batch["labels"].float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


#-------------------------------------------------------------Evaluation-------------------------------------------------

def evaluate_classification(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Classification"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

def evaluate_regression(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Regression"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.squeeze(-1)
            preds.extend(logits.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    mae = mean_absolute_error(labels, preds)
    r, _ = pearsonr(labels, preds)
    return mae, r

#-------------------------------------------------------------------------------------------------------------------------


boolq_train_loader, boolq_val_loader = get_boolq_dataloaders(batch_size=32)

# distilRB-rand
model_rand = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(device)
distilRB_rand(model_rand)
acc_rand, f1_rand = evaluate_classification(model_rand, boolq_val_loader)

# distilroberta(Without any modifications)
model_roberta = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(device)
model_roberta = fine_tune_classification(model_roberta, boolq_train_loader)
acc_roberta, f1_roberta = evaluate_classification(model_roberta, boolq_val_loader)

# distilRB-KQV
model_kqv = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(device)
distilRB_KQV(model_kqv)
model_kqv = fine_tune_classification(model_kqv, boolq_train_loader)
acc_kqv, f1_kqv = evaluate_classification(model_kqv, boolq_val_loader)

# distilRB-nores
model_nores = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(device)
replace_last_two_layers_with_nores(model_nores)
model_nores = fine_tune_classification(model_nores, boolq_train_loader)
acc_nores, f1_nores = evaluate_classification(model_nores, boolq_val_loader)

#---------------------------------------------------------Checkpoint 2.2--------------------------------------------------
print("\nCheckpoint 2.2\n")
print("\nboolq validation set:")
print(f"  distilRB-rand: overall acc: {acc_rand:.3f}, f1: {f1_rand:.3f}")
print(f"  distilroberta: overall acc: {acc_roberta:.3f}, f1: {f1_roberta:.3f}")
print(f"  distilRB-KQV: overall acc: {acc_kqv:.3f}, f1: {f1_kqv:.3f}")
print(f"  distilRB-nores: overall acc: {acc_nores:.3f}, f1: {f1_nores:.3f}")

#-------------------------------------------------------------2.3.1---------------------------------------------------------

# Formating the data to fit perfectly into dataloader
sst_train_loader, sst_val_loader, sst_test_loader = get_sst_dataloaders(batch_size=32) 

# num_labels is 1 here, as sepcified in the instructions
model_roberta_reg = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=1).to(device)

# Fine-tuning roberta on sst data
model_roberta_reg,training_loss = fine_tune_regression(model_roberta_reg, sst_train_loader)

plt.figure(figsize=(10, 4))
plt.plot(range(1,len(training_loss)+1),training_loss)
plt.xlabel('Batch Number')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve - distilroberta Regression on SST')
plt.grid(True)
plt.show()

# Evaluate metrices
val_mae, val_r = evaluate_regression(model_roberta_reg, sst_val_loader)
test_mae, test_r = evaluate_regression(model_roberta_reg, sst_test_loader)

#----------------------------------------------------------Checkpoint 2.3.1---------------------------------------------
print("\nCheckpoint 2.3.1 \n")
print(f" Validation: mae: {val_mae:.3f}, r: {val_r:.3f}")
print(f" Test:       mae: {test_mae:.3f}, r: {test_r:.3f}")

#----------------------------------------------------------2.3.2-------------------------------------------------------

# distilRB-rand
model_rand_reg = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=1).to(device)
distilRB_rand(model_rand_reg)
# model_rand_reg,_ = fine_tune_regression(model_rand_reg, sst_train_loader)
rand_mae, rand_r = evaluate_regression(model_rand_reg, sst_test_loader)

# distilRB-KQV
model_kqv_reg = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=1).to(device)
distilRB_KQV(model_kqv_reg)
model_kqv_reg,_ = fine_tune_regression(model_kqv_reg, sst_train_loader)
kqv_mae, kqv_r = evaluate_regression(model_kqv_reg, sst_test_loader)

# distilRB-nores
model_nores_reg = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=1).to(device)
replace_last_two_layers_with_nores(model_nores_reg)
model_nores_reg,_ = fine_tune_regression(model_nores_reg, sst_train_loader)
nores_mae, nores_r = evaluate_regression(model_nores_reg, sst_test_loader)

#------------------------------------------------------------Checkpoint-2.3.2-------------------------------------------
print("\nCheckpoint 2.3.2\n")
print(f" distilRB-rand:  mae: {rand_mae:.3f}, r: {rand_r:.3f}")
print(f" distilRB-KQV:   mae: {kqv_mae:.3f}, r: {kqv_r:.3f}")
print(f" distilRB-nores: mae: {nores_mae:.3f}, r: {nores_r:.3f}")
#-------------------------------------------------------------------------------------------------------------------------
end_time = time.time()
tot_time = end_time - start_time #---------------------This is to show that time taken to run until extra credits

print(f"\nTotal Runtime up to Checkpoint 2.3.2: {int(tot_time// 60)} minutes {round(tot_time % 60,2)} seconds\n")

#------------------------------------------------------Extra credit--------------------------------------------------------

print("\n\n----------------------Extra Credit: Improved Model Fine-Tuning--------------------------------\n")

improved_boolq_model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
).to(device)

improved_sst_model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=1,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
).to(device)

# Fine-tune on BoolQ
improved_boolq_model = fine_tune_classification(improved_boolq_model, boolq_train_loader)

# Fine-tune on SST
improved_sst_model, _ = fine_tune_regression(improved_sst_model, sst_train_loader)


improved_boolq_acc, improved_boolq_f1 = evaluate_classification(improved_boolq_model, boolq_val_loader)
improved_sst_mae, improved_sst_r = evaluate_regression(improved_sst_model, sst_test_loader)

# Comparing with Best Previous Models
print("\n\n===== Final Extra Credit Results =====\n")
print("BoolQ Validation Set:")
print(f" - Improved DistilRoberta Accuracy: {improved_boolq_acc:.3f}, F1: {improved_boolq_f1:.3f}")
print(f" - Best Previous Accuracy: {max(acc_roberta, acc_kqv, acc_nores):.3f}")

print("\nSST Test Set:")
print(f" - Improved DistilRoberta MAE: {improved_sst_mae:.3f}, r: {improved_sst_r:.3f}")
print(f" - Best Previous MAE: {min(test_mae, kqv_mae, nores_mae):.3f}")

print("\n---------------------------------------End of Extra Credit Section and end of part 2 :) -----------------------------")


