Something I need to take into account for the [[Data Pipeline]]. Transformer models have a token limit that's pretty strict (around 512?). This makes embedding long documents fairly difficult. Especially since Danish doesn't have any LongFormer models available (yet). Below are possible solutions: 

- **Parse documents as paragraphs:** This approach would split the documents into shorter paragraphs (all below 512). Afterwards the documents could be summed up (or averaged) to produce the complete embedding. That would only work, however, if the paragraphs are logical units (need to look closer at the data)
- **Use a multi-langauge longformer:** This would just substitute the embedding with a multi-language longformer. However, no such thing appears to exist so this method isn't feasible.
- **Train a Danish longformer:** This approach might be the best. However, it is too large a project for my thesis. Luckily, it can later be added as the embedding layer which highlihgts that [[Embeddings has more room for growth than LDA]]. 

