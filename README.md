# TransTopicXAI
Exploration of topic modelling for transformers 📚🤖

## What is this repo for? 🤔
This repo is for exploration and thinking about my b.sc. thesis on creating end-user explainable embeddings using transformers and [BERTopic](https://github.com/MaartenGr/BERTopic). It will be include both code and notes+thoughts. 

### Why End-User Explainability? 👩‍🏫
Many AI-solutions ultimately end up as decision support for non-technical end users. While experts in their domain, these users might have very limited (if any) knowledge of SOTA machine learning algorithms. Therefore it is important to provide *intuitive* and *trustworthy* algorithms and interfaces. This project will probably mainly deal with the former but hopefully also attempt the later. 

## How to reproduce the results
The structure of the repo is very much under development. However as of now it looks something like this: 
0. Set up the correct environment(s) - see `embenv.yml` and `topicmodel.yml`
1. Create the data using `src/tweeteval_data.py` (This requires, you have cloned the [tweeval repo](https://github.com/cardiffnlp/tweeteval/))
2. Create embeddings for the full data using `src/embed_document_df` (requires following folders: models and data)
3. Fit the topic model using `src/create_topic.py` 
4. Train topic-based model with `src/topic-based-embeddings.py`
5. Get pysentimiento prediction with `src/predict_bertweet.py`
6. Evaluate with `src/evaluation.py`
7. Create interview plots with `src/manual_pred.py`
8. Create LIME plots with `src/BERT_lime.py`

all of this can be accomplished by running `reproduce_bsc.sh` with conda installed. 

Any feedback is much appreciated! 😁

