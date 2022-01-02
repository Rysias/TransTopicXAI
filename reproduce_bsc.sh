eval "$(conda shell.bash hook)"
echo 'Setting up embedding environment...'
conda env create -f embenv.yml
echo 'setting up twitter data'
cd ..
git clone https://github.com/cardiffnlp/tweeteval.git
cd TransTopicXAI
mkdir data models data/explains
cd src
conda activate embenv
python tweeteval_data.py
echo 'embedding data'
python embed_documents_df.py --data-path "../data/tweeteval_text.csv" --embedding-path "../data/" --lazy
echo 'creating BERTweet predictions'
python predict_bertweet.py --data-dir "../data"
conda deactivate
echo 'done embedding data!'
echo 'Setting up topic-model environment'
cd ..
conda env create -f topicmodel.yml
conda activate topicenv
cd ..
git clone https://github.com/Rysias/clearformers.git
cd clearformers 
git pull
pip install .
echo "done!"
echo 'Training topic-model'
cd ../TransTopicXAI/src
python create_topic.py --data-path "../data/tweeteval_text.csv" --embedding-path "../data/bertweet-base-sentiment-analysis_embs.npy" --data-size 40000 --save-path "../models"
echo 'done!'
echo 'Training topic-based embeddings'
python topic-based-embeddings.py --data-dir "../data" --topic-dir "../models"
echo 'done!'
echo 'Creating topic info file'
python get_topic_info.py
echo 'done!'
echo "run evaluations..."
python evaluation.py
echo "done!"
echo "creating explanations"
python BERT_lime.py --data-dir "../data/explains"
python manual_pred.py
