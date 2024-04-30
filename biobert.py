# biobert to recognize medical entities in text, biobert has medical vocabularies
# The biobert is a pretrained model, no need to worry about the data privacy
import pandas as pd
from transformers import pipeline
import re
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# # Load the data
# df = pd.read_excel('bd_adherence_NOID_V2.xlsx')
# df['combined_text'] = df[['Subjetivo', 'Objetivo', 'Analisis', 'Plan']].apply(lambda x: ' '.join(x.dropna().values.tolist()), axis=1)
# print(df['combined_text'].head())

# Load BioBERT model specifically for NER(named entity recognition)
nlp_ner = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")
# ner_results = nlp_ner(text)

# This is an example data set,
# we are going to use the biobert for both the guidelines and health records to extract the drug use
# Fine-tune is needed for specific drug recognitions, this is using nltk, I will try to biobert to achieve better performance
# Then use the bert to compare and find out the similarity


text1 = """
"
The patient was treated with a regimen including Doxorubicin 50 mg/m2 IV push on day one,
"
"""

text = """
"
Chemotherapy

"

"
"""

drug_name_pattern = r'\b[A-Z][a-z]*[a-z\-]*\b'


def extract_drug_names(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences]

    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    drug_names = []
    for sentence in tagged_sentences:
        for word, tag in sentence:
            if re.match(drug_name_pattern, word):
                drug_names.append(word)

    return list(set(drug_names))


drug_names = extract_drug_names(text)

print("Drug Names:", drug_names)
