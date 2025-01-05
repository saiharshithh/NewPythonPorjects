import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def summarize_text(text, max_length=150, temperature=0.7, num_beams=4, do_sample=True):
    input_ids = tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=num_beams, temperature=temperature,
                                 do_sample=do_sample)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Streamlit UI
st.set_page_config(page_title="Text Summarizer", layout="wide")

# Header Section
st.markdown(
    """
    <style>
    .header {
        text-align: center;
        color: #4CAF50;
        font-size: 32px;
        font-weight: bold;
    }
    .subheader {
        text-align: center;
        color: #007BFF;
        font-size: 20px;
    }
    .container {
        max-width: 800px;
        margin: 0 auto;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        border-radius: 5px;
        font-size: 16px;
        margin-top: 20px;
    }
    .button:hover {
        background-color: #45a049;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.markdown('<div class="header">Text Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Summarize your text with ease</div>', unsafe_allow_html=True)

# Input Text Area
input_text = st.text_area("Enter text to summarize:", height=200)

# Parameters for summarization
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
num_beams = st.slider("Number of beams", 1, 10, 4)

# Summarize Button
if st.button("Summarize"):
    if input_text:
        with st.spinner("Summarizing..."):
            summary = summarize_text(input_text, max_length=150, temperature=temperature, num_beams=num_beams,
                                     do_sample=True)
        st.markdown(f"**Summary:**\n\n{summary}")

        # Copy Button
        copy_button = st.button("Copy Summary")
        if copy_button:
            # JavaScript to copy text to clipboard
            st.markdown(
                f"""
                <script>
                navigator.clipboard.writeText("{summary}");
                alert("Summary copied to clipboard!");
                </script>
                """, unsafe_allow_html=True)

    else:
        st.warning("Please enter some text to summarize.")

# Footer Section
st.markdown("""
    <div class="footer">
        A project by <a href="https://www.linkedin.com/in/saiharshithh/" target="_blank">Sai Harshith</a>
    </div>
""", unsafe_allow_html=True)
