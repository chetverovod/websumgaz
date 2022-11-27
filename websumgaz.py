#!/usr/bin/python3 
# -*- coding: utf-8 -*-

import io
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from io import StringIO

model_name = "IlyaGusev/rugpt3medium_sum_gazeta"

@st.cache(allow_output_mutation=True)
def load_model():
    # return EfficientNetB0(weights='imagenet')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return  AutoModelForCausalLM.from_pretrained(model_name), tokenizer

def load_txt():
    uploaded_file = st.file_uploader(label='Выберите текст для распознавания')
    if uploaded_file is not None:
        txt_data = uploaded_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)
        #st.image(txt_data)
        #return Image.open(io.BytesIO(txt_data))
        #return st.text_area(txt_data)    
        return string_data
    else:
        return None

def summarize(model, tokenizer, article_text):
    text_tokens = tokenizer(
    article_text,
    max_length=600,
    add_special_tokens=False, 
    padding=False,
    truncation=True)["input_ids"]
    input_ids = text_tokens + [tokenizer.sep_token_id]
    input_ids = torch.LongTensor([input_ids])

    output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4,
    max_new_tokens = 100
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    summary = summary.split(tokenizer.sep_token)[1]
    summary = summary.split(tokenizer.eos_token)[0]
    return summary  
#-----------------------------------------------------------------------

def print_summary(s):
        st.write(s)


model, tokenizer = load_model()


st.title('Генерация кратких выводов из газетной статьи')
txt = load_txt()

#st.write('-------------------------')
#st.write(txt)
#st.write('-------------------------')
result = st.button('Анализировать')
if result:
    sum = summarize(model, tokenizer, txt)
    st.write('**Выводы:**')
    print_summary(sum)






















