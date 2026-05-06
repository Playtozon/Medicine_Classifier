import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load the model, tokenizer, and LabelEncoder
@st.cache_resource
def load_components():
    # Make sure this matches your Hugging Face repo name!
    hf_repo_name = "playtozon/drug-action-classifier"
    
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo_name)
    
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
        
    return tokenizer, model, le

tokenizer, model, le = load_components()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Build the UI
st.title("💊 Drug Action Class Predictor")
st.write("Enter the drug details below to predict its therapeutic action class.")

drug_name = st.text_input("Drug Name (e.g., amoxicillin 500mg capsule)", "")
uses = st.text_area("Uses (e.g., Treatment of bacterial infections)", "")
side_effects = st.text_area("Side Effects (e.g., Nausea, diarrhea)", "")

# 3. Prediction Logic
if st.button("Predict Action Class"):
    if not drug_name and not uses and not side_effects:
        st.warning("Please enter some details to make a prediction.")
    else:
        # Reconstruct the exact input text format from your training pipeline
        input_text = f"Drug: {drug_name}. Uses: {uses}. Side Effects: {side_effects}"
        
        # Tokenize the input
        encoding = tokenizer(
            input_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Run inference
        with st.spinner("Analyzing..."):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1)[0]
                
                # Get the top 3 predictions and their indices
                top3 = torch.topk(probs, k=3)
                top3_indices = top3.indices.cpu().numpy()
                top3_probs = top3.values.cpu().numpy()
                
            # Inverse transform the indices back to class strings
            top3_classes = le.inverse_transform(top3_indices)
            
            # Display the primary success message
            st.success(f"**Top Prediction:** {top3_classes[0]} ({top3_probs[0]:.1%})")
            
            # Display the Top 3 breakdown with progress bars
            st.markdown("### Top 3 Confidence Breakdown:")
            
            for cls, prob in zip(top3_classes, top3_probs):
                # Create two columns: one for the bar, one for the percentage text
                col1, col2 = st.columns([4, 1]) 
                
                with col1:
                    st.write(f"**{cls}**")
                    # st.progress requires a float between 0.0 and 1.0
                    st.progress(float(prob))
                with col2:
                    # Add vertical spacing to align the text with the progress bar
                    st.write("")
                    st.write(f"**{prob:.1%}**")