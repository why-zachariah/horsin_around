# # wrappers to load models
# from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer

# # https://huggingface.co/docs/transformers/installation#install-with-pip
# # https://huggingface.co/EleutherAI/pythia-6.9b

# def my_mistral(device = 'cpu', dowload = False):
#     '''
#     returns the 7B mistral instruct model and its tokenizer
#     '''

#     if dowload:
#         model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
#         tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

#         model.save_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")
#         tokenizer.save_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")

#     else:
#         model = AutoModelForCausalLM.from_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")
#         tokenizer = AutoTokenizer.from_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")

#     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
#     model.to(device)

#     return model, tokenizer


# def my_pythia(model_params = '70m', STEP = 143000, device = 'cpu'):
#     '''
#     returns a pythia model and its tokenizer
#     '''

#     model = GPTNeoXForCausalLM.from_pretrained(
#       f"EleutherAI/pythia-{model_params}-deduped",
#       revision=f"step{STEP}",
#       cache_dir=f"./pythia-{model_params}-deduped/step{STEP}",
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#       f"EleutherAI/pythia-{model_params}-deduped",
#       revision=f"step{STEP}",
#       cache_dir=f"./pythia-{model_params}-deduped/step{STEP}",
#     )

#     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
#     model.to(device)

#     return model, tokenizer


import openai

# Initialize OpenAI API client
def initialize_openai(api_key):
    openai.api_key = api_key

# Function to generate responses from ChatGPT-4-O using OpenAI API
def my_gpt(prompt, temperature, api_key, n_iter=8):
    '''
    Generates responses from the GPT-4 model using the OpenAI API.
    '''
    openai.api_key = api_key
    responses = []
    for _ in range(n_iter):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=100
        )
        responses.append(response['choices'][0]['message']['content'].strip())
    return responses
