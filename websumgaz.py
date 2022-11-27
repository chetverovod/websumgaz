#!/usr/bin/python3 

import io
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# from PIL import Image
# import numpy as np
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

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
        #st.image(txt_data)
        #return Image.open(io.BytesIO(txt_data))
        return st.text_area(txt_data)
    else:
        return None
# ----------------------------------------------------------------------
#article_text = "По информации издания, для того, чтобы продолжить работу в Twitter, сотрудникам нужно перейти по ссылке в письме и нажать «да» до 17:00 по местному времени (01:00 по московскому времени 17 ноября). В противном случае работников уволят с выплатой трёхмесячного выходного пособия.  В своём письме Маск не обещает работникам лёгкой жизни. По его словам, оставшийся персонал будет работать «долгими часами и с высокой интенсивностью». «Только исключительная продуктивность будет проходным баллом», — отметил Маск.  По данным издания Daily Mail, Маск, уже уволил не менее десяти сотрудников, которые критиковали его в корпоративном мессенджере Slack. Это сделало его «Тем-Кого-Нельзя-Называть», подобно Волан-де-Морту из саги о Гарри Поттере. Сотрудники боятся, что новые коллеги, которых Маск привёл с собой из Tesla, будут просматривать все сообщения в Slack и по поисковым запросам «Илон» и «Маск» находить людей, которые грубо высказываются о новом начальнике."

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
    no_repeat_ngram_size=4
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    summary = summary.split(tokenizer.sep_token)[1]
    summary = summary.split(tokenizer.eos_token)[0]
    return summary  
#-----------------------------------------------------------------------

def print_summary(s):
        st.write(s)


model, tokenizer = load_model()


st.title('Генерация кратких выводов  из газетной статьи')
txt = load_txt()
result = st.button('Анализировать')
if result:
    sum = summarize(model, tokenizer, txt)
    st.write('**Выводы:**')
    print_summary(sum)






















