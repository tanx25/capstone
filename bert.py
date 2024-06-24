import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "alvaroalon2/biobert_chemical_ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)


ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


text_1 = """
The patient, a 56-year-old male with a history of hypertension and type 2 diabetes, was diagnosed with stage IIIB non-small cell lung cancer. He began chemotherapy with the following regimen: Cisplatin 75 mg/m2 IV on day 1 and Etoposide 100 mg/m2 IV on days 1-3, repeated every 21 days for four cycles. In addition, the patient was prescribed Amoxicillin 500 mg orally three times daily for 7 days to treat a concurrent respiratory infection. During the treatment, the patient experienced significant nausea and vomiting, which were managed with Ondansetron 8 mg orally twice daily as needed. Due to the patient's compromised renal function, the Cisplatin dose was reduced by 25% for the last two cycles of chemotherapy. Following the completion of chemotherapy, the patient was started on maintenance therapy with Pembrolizumab 200 mg IV every 3 weeks. Additionally, the patient takes Metformin 500 mg orally twice daily for diabetes management and Lisinopril 20 mg orally once daily for hypertension control.
"""

text_2="""Multiple myeloma IgG lambda// QT: Dara-CyBorD + zoledronic acid, cycle 2, day 22. Cycle 3 is scheduled."""

text=""" 
Chemotherapy
Epirubicin (Ellence) 90 mg/m2 IV once on day 1
Paclitaxel (Taxol) 175 mg/m2 IV once on day 1
21-day cycle for 4 cycles

"""
text_4=""" 
Chemotherapy, AC portion (cycles 1 to 4)
Doxorubicin (Adriamycin) 60 mg/m2 IV once on day 1
Cyclophosphamide (Cytoxan) 600 mg/m2 IV once on day 1
Chemotherapy, D portion (cycles 5 to 8)
Docetaxel (Taxotere) 75 mg/m2 IV once on day 1
21-day cycle for 8 cycles (AC x 4; T x 4)

"""


entities = ner_pipeline(text)

drug_entities = [entity for entity in entities if entity['entity_group'] in ['CHEMICAL']]

def extract_dosage(context):

    dosage_pattern = r'\d+(?:\.\d+)?\s*(?:mg|g|ml|µg|mcg|IU|units|tablets|capsules)/?\s*(?:day|daily|weekly|month|hour|minute|week|tablet|capsule)?'
    dosages = re.findall(dosage_pattern, context, re.IGNORECASE)
    return dosages

training_data = []
for entity in drug_entities:
    start = max(0, entity['start'] - 100)
    end = min(len(text), entity['end'] + 100)
    context = text[start:end]
    dosages = extract_dosage(context)

    training_data.append({
        "drug": entity['word'],
        "dosages": ', '.join(dosages) if dosages else 'Not found'
    })



print("Extracted Drug Information:")
for data in training_data:
    print(f"Drug: {data['drug']}, Dosage: {data['dosages']}")

# print(training_data)
# df_output = pd.DataFrame(training_data)
#
# print("\nDataFrame Output:")
# print(df_output)
