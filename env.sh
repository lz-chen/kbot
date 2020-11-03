#!/bin/bash
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/extras/CUPTI/lib64
cur=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${cur}/chatbot_model"
export PYTHONPATH="${PYTHONPATH}:${cur}/chatbot_model/gpt_model/src"
pipenv install --skip-lock
git clone https://github.com/CODAIT/graph_def_editor.git
pipenv install --skip-lock ./graph_def_editor
# problem: the lock fails with tensor2tensor
pipenv lock --clear