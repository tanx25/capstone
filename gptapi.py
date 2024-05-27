from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import os
import pandas as pd
import torch

load_dotenv()
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKENS')

file_path = "QT_regimenes.xlsx"
df = pd.read_excel(file_path)

results = []

model_name = "openbmb/MiniCPM-Llama3-V-2_5"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

def analyze_text(text):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for index, row in df.iterrows():
    text = " ".join([str(item) for item in row if not pd.isnull(item)])
    result = analyze_text(text)
    results.append(result)

df['Analysis Results'] = results

output_file_path = "QT_regimenes_analyzed.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Analysis complete. Results saved to {output_file_path}")
