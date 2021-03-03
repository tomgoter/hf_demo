# Text Classification with Multiple GPUs
## TP Goter
### March 2021

This repo contains an example script that is setup to run text classification using various transformer-based models from HuggingFace transformers library. The data set used is the AGNews dataset which consists of 120K training examples and 7600 validation examples. Also contained in this repository is a dockerfile that can be used to setup a controlled environment. This process has been tested locally and on the IBM Cloud.

Process for Training in Cloud:

1. Requisition Required Resources (in this case a 2xV100 GPU node)
2. Clone this repository `git clone https://github.com/tomgoter/hf_demo.git`
3. Create docker image - in this case we will call it `nlp' `docker build -t nlp -f hf_demo/build.docker .`
4. Start docker container `docker run -it --rm -v ~/hf_demo:/hf_demo nlp bash`
5. Execute the training script ```python multigpu_example.py \\
6.                                              --model MODEL \\
7.                                              --BATCH_SIZE BATCH_SIZE \\
8.                                              --SEQUENCE_LENGTH SEQUENCE_LENGTH```



