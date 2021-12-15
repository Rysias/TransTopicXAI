eval "$(conda shell.bash hook)"
echo 'Setting up embedding environment...'
conda env create -f embenv.yml || conda activate embenv
echo 'setting up twitter data'
cd ..
git clone https://github.com/cardiffnlp/tweeteval.git
cd TransTopicXAI
mkdir data models data/explains
cd src
conda activate embenv
python tweeteval_data.py
echo 'embedding data'
python embed_documents_df.py --data-path "../data/tweeteval_text.csv" --embedding-path "../data/"
conda deactivate
echo 'done embedding data!'
echo 'Setting up topic-model environment'
cd ..
conda env create -f topicmodel.yml
conda activate rapids-21.12
cd ..
git clone https://github.com/Rysias/clearformers.git
cd clearformers 
pip install .
echo "done!"
echo 'Training topic-model'
cd ../TransTopciXAI/src
python create_topic.py --data-path "../data/tweeteval_text.csv" --embedding-path "../data/bertweet-base-sentiment-analysis_embs.npy" --data-size 45000 --save-path "../models"
echo 'done!'
echo 'Training topic-based embeddings'
python topic-based-embeddings.py --data-dir "../data" --topic-path "../models"
echo 'done!'
echo "run evaluations..."
python evaluation.py

