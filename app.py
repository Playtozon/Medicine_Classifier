import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_components():
    tokenizer = AutoTokenizer.from_pretrained("./best_drug_model/")
    model = AutoModelForSequenceClassification.from_pretrained("./best_drug_model/")
    
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
        
    return tokenizer, model, le

tokenizer, model, le = load_components()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

st.title("Drug Action Class Predictor")
st.write("Enter the drug details below to predict its therapeutic action class.")

drug_name = st.text_input("Drug Name (e.g., amoxicillin 500mg capsule)", "")
uses = st.text_area("Uses (e.g., Treatment of bacterial infections)", "")
side_effects = st.text_area("Side Effects (e.g., Nausea, diarrhea)", "")

if st.button("Predict Action Class"):
    if not drug_name and not uses and not side_effects:
        st.warning("Please enter some details to make a prediction.")
    else:
        input_text = f"Drug: {drug_name}. Uses: {uses}. Side Effects: {side_effects}"
        
        encoding = tokenizer(
            input_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with st.spinner("Analyzing..."):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]
                pred_id = torch.argmax(probs).item()
                
            pred_class = le.inverse_transform([pred_id])[0]
            
            st.success(f"**Predicted Action Class:** {pred_class}")