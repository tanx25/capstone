import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Bio_ClinicalBERT models using Hugging Face
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize the Named Entity Recognition (NER) pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# read data from excel
file_path = "QT_regimenes.xlsx"
df = pd.read_excel(file_path)

# Merge all lines into one text for NER processing
text = " ".join(df.astype(str).values.flatten())

# Extract named entities from text
entities = ner_pipeline(text)

# Print extracted entities
print("Raw entities:", entities)

# Filter and extract relevant entities
relevant_entities = [entity for entity in entities if entity['entity_group'] in ['DRUG']]

print("Extracted Entities:")
for entity in relevant_entities:
    print(f"Text: {entity['word']}, Entity: {entity['entity_group']}")

training_data = []
for entity in relevant_entities:
    start = max(0, entity['start'] - 50)
    end = min(len(text), entity['end'] + 50)
    context = text[start:end]
    training_data.append({
        "entity": entity['word'],
        "entity_group": entity['entity_group'],
        "context": context,
        "start": entity['start'],
        "end": entity['end']
    })


print("Training Data:")
for data in training_data:
    print(data)


df_output = pd.DataFrame(training_data)


output_file_path = "training_data.csv"
df_output.to_csv(output_file_path, index=False)

print(f"Entities extraction complete. Results saved to {output_file_path}")
