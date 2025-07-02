# Import the necessary libraries
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_generator.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Truncate if too long
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Predict
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Convert index to word
    word = next((w for w, i in tokenizer.word_index.items() if i == predicted_word_index), None)
    return word

# Create the Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of words", "To be or not to")

if st.button("Predict Next Word"):
    try:
        # Retrieve max sequence length
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        
        if next_word:
            st.write(f'**Next word:** {next_word}')
        else:
            st.warning("Unable to predict the next word. It may be out of vocabulary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")