# # functions to generate and evaluate text outputs

# import torch

# def f_generate_batch(model, tokenizer, prompts, text_length, temperature=1.0):
#     device = next(model.parameters()).device
#     inputs = tokenizer(prompts, return_tensors="pt")
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     max_length = inputs["input_ids"].shape[1] + text_length

#     return model.generate(**inputs, max_length=max_length,
#                             pad_token_id=tokenizer.pad_token_id,
#                             temperature=temperature, do_sample=True)[:,-text_length:]



# def f_eval_prob_batch(model, tokenizer, prompts, outputs, temperature=1.):
#     res = [] ###
#     # determine the device the model is on
#     device = next(model.parameters()).device

#     assert len(prompts) == len(outputs), "Mismatch between number of prompts and outputs"

#     # ensure tokenized inputs are on the right device
#     ids_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
#     ids_outputs = outputs.to(device)

#     max_len_output = ids_outputs.size(1)
#     log_prob_sums = torch.zeros(len(prompts)).to(device)

#     with torch.no_grad():
#         for i in range(max_len_output):
#             # concatenate prompts and the current output sequences
#             current_output_tokens = ids_outputs[:, :i]
#             combined_inputs = torch.cat([ids_prompts, current_output_tokens], dim=-1)

#             # use a mask to ignore the padding tokens
#             if i>0: mask = ((ids_outputs[:, i-1] != tokenizer.pad_token_id)
#                              * (ids_outputs[:, i-1] != tokenizer.eos_token_id)) # apply mask if *previous* token is pad or eos
#             else: mask = torch.ones_like(ids_outputs[:, i], dtype=torch.bool) # always evaluate on zeroth output token

#             # get model's predictions for the next token
#             logits = model(combined_inputs).logits[:, -1]
#             if temperature != 1.: logits = logits / temperature

#             # compute log probabilities directly using log_softmax for numerical stability
#             log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

#             # extract the log probability of the actual next token in the output for each prompt-output pair
#             real_tokens = ids_outputs[mask, i]
#             batch_log_probs = log_probs[mask].gather(1, real_tokens.unsqueeze(-1)).squeeze()

#             log_prob_sums[mask] += batch_log_probs
            

#     return log_prob_sums.tolist()

import openai
import numpy as np

# Function to generate a batch of text outputs using the OpenAI API
def f_generate_batch(prompt, n_iter, temperature, api_key):
    '''
    Generates a batch of text outputs using the OpenAI API.
    '''
    responses = []
    for _ in range(n_iter):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=100,
        )
        responses.append(response.choices[0].message.content.strip())
    return responses

# Approximation of log probabilities since OpenAI API doesn't expose log-probs directly
def f_eval_prob_batch(ys):
    '''
    Placeholder for log-probability evaluation function.
    This approximates log-probability evaluation by assuming each output's likelihood based on certain metrics.
    '''
    # Here, you can compute something like n-grams similarity or just return a mock value
    res = []
    for y in ys:
        res.append(len(y))  # Just return a mock log-prob for now
    return res
