# websumgaz
Web-applicatation for newspaper summarization,
based on Ilya Gusev's model 
[link to model](https://huggingface.co/IlyaGusev/rugpt3medium_sum_gazeta)

## Create virtual environment
```
$ conda create --name py37 python=3.7
```

### To activate this environment, use
```
$ conda activate py37
```

### To deactivate an active environment, use
```
$ conda deactivate
```
## Install torch
```
$ conda activate py37
does not works: $ sudo pip install torch
does not works: $ conda install torch
$ pip install torchvision 
```

## Install transformers
```
$ pip install transformers
```

## Install streamlit
```
pip install streamlit
```

## Run web-application
```
$ streamlit run websumgaz.py 
```
Web-page will be open automaticaly. 
