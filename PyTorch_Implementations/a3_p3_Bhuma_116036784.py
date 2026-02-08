import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

#---------------------------------------------------------3.1----------------------------------------------------------
def get_unique_ctx_examples(squad, n=500):

    context2idx = {}
    for i, entry in enumerate(squad['validation']):
      if not entry['context'] in context2idx:
        context2idx[entry['context']] = []
        context2idx[entry['context']].append(i)


    queries, contexts, answers = [], [], []

    for k,v in context2idx.items():
        idx = v[0]
        queries.append(squad['validation'][idx]['question'])
        contexts.append(squad['validation'][idx]['context'])
        answers.append(squad['validation'][idx]['answers'])
        if len(queries) == n:
            break

    return queries, contexts, answers

def retrieve(contexts, embeddings, query):
    query_emb = model.encode(query, convert_to_tensor=True)
    similarities = cosine_similarity(query_emb.unsqueeze(0), embeddings)  # shape (1, 500)
    idx = torch.argmax(similarities).item()
    ret_context = contexts[idx]
    return idx, ret_context


squad = load_dataset("squad")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device = device)
queries, contexts, answers = get_unique_ctx_examples(squad)
context_embeddings = model.encode(contexts, convert_to_tensor=True)

correct = 0
total = len(queries)

for query, true_context in zip(queries, contexts):
    idx, retrieved_context = retrieve(contexts, context_embeddings, query)
    if retrieved_context.strip() == true_context.strip():
        correct += 1

accuracy = correct / total
#---------------------------------------------------------------Checkpoint-3.1-------------------------------------------
print("\n Checkpoint-3.1 \n")
print(f"\nRetrieval Accuracy = {correct}/{total} = {accuracy:.3f}")


#----------------------------------------------------------------------------------------------3.2-------------------------------------------------------------------------------------------------------
def generate_response(model, query, ret_context):

    """
    #input
        model: an instance of LM
        query: the question as a string
        ret_context: context retrieved from the embedded vectors

    #output
        response: a string of tokens obtained from the model
    """

    # Instruction template

    message = (
        "You are a helpful AI assistant. "
        "Provide one Answer ONLY to the following query based on the context provided below. "
        "Do not generate or answer any other questions. "
        "Do not make up or infer any information that is not directly stated in the context. "
        "Provide a concise answer.\n"
        f"Context: {ret_context}\n"
        f"Question: {query}\n"
        "Answer:"
    )


    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    inputs = tokenizer(message, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].splitlines()[0].strip()

    return response

LM_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",torch_dtype=torch.bfloat16,).to(device)

  # Selecting 5 correctly and 5 incorrectly retrieved exmaples
correct_examples = []
incorrect_examples = []
for query, ctx, ans in zip(queries, contexts, answers):
  idx, ret_context = retrieve(contexts, context_embeddings, query)
  if ret_context == ctx and len(correct_examples)<5:
    correct_examples.append((query, ctx, ans))
  elif ret_context != ctx and len(incorrect_examples)<5:
    incorrect_examples.append((query, ctx, ans))
  if len(correct_examples) >= 5 and len(incorrect_examples) >= 5:
    break


print("\n Check point-3.2\n")
print(f"\nSummary:")
print(f"Correctly retrieved examples: {len(correct_examples)}")
print(f"Incorrectly retrieved examples: {len(incorrect_examples)}\n")

# Print Correctly Retrieved Examples
print(f"-------------------------------Correctly Retrieved Examples------------------------------------------------\n")

for query, ctx, ans in correct_examples:
    response = generate_response(LM_model, query, ctx)
    print(f"Question:\n{query}\n")
    print(f"LM's Answer:\n{response}\n")
    print(f"Ground Truth Answer:\n{ans['text'][0]}\n")
    print("-" * 70)

# Print Incorrectly Retrieved Examples
print(f"\n------------------------------------------------Incorrectly Retrieved Examples---------------------------\n")

for query, ctx, ans in incorrect_examples:
    response = generate_response(LM_model, query, ctx)
    print(f"Question:\n{query}\n")
    print(f"LM's Answer:\n{response}\n")
    print(f"Ground Truth Answer:\n{ans['text'][0]}\n")
    print("-" * 70)

