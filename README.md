# TransTopicXAI
Exploration of topic modelling for transformers 📚🤖

## What is this repo for? 🤔
This repo is for exploration and thinking about my b.sc. thesis on creating end-user explainable embeddings using transformers and [BERTopic](https://github.com/MaartenGr/BERTopic). It will be include both code and notes+thoughts. 

### Why End-User Explainability? 👩‍🏫
Many AI-solutions ultimately end up as decision support for non-technical end users. While experts in their domain, these users might have very limited (if any) knowledge of SOTA machine learning algorithms. Therefore it is important to provide *intuitive* and *trustworthy* algorithms and interfaces. This project will probably mainly deal with the former but hopefully also attempt the later. 

## How to reproduce the results
The structure of the repo is very much under development. However as of now it looks something like this: 
0. Set up the correct environment(s) (TODO)
1. Create the data using `src/tweeteval_data.py` (This requires, you have cloned the [tweeval repo](https://github.com/cardiffnlp/tweeteval/))
2. Create embeddings for the full data using `src/embed_document_df` (requires you to have some specific folders for saving stuff; TODO)
3. Fit the topic model using `src/create_topic.py` (also requires certain folders; TODO)
4. Transform the training set corpus with `src/transform_topics.py`
5. Create topic-embeddings using `src/create_topic_embeds.py`
6. Train classifier with `src/fit_explainlp.py`
7. Get pysentimiento prediction with `src/predict_bertweet.py`
8. Evaluate with `src/evaluation.py`
9. Create interview plots with `src/manual_pred.py`
10. Create LIME plots with `src/BERT_lime.py`

Any feedback is much appreciated! 😁

