## Overview of data pipeline
How should the data look to facilitate semantic search? Below is a list of the data "flow"
1. Documents (docs): raw texts -->
2. Clean doc: (optionally) cleaned documents (stop words, lower cased etc). This step is not necessary for the pipeline but might improve performance. -->
3. Embedded docs: Documents are embedded using some kind of sentence-transformer solution. Output is saved
4. Topic creation: Following the BERTopic-pipeline (LINK), a list of topics are created. How these exactly are represented is currently unclear, so I will have to read up on HDBScan.
5. Initial topic evaluation: [[Data Scientists can assure minimum quality]] --> iterate if subpar, continue if super
7. User based topic evaluation ([[Topic Evaluations depends on Domain Experts]])
8. Final representation: Each document should have a similarity score to each of the topics (after a fitting amount has been selected). The similarity scores should ideally be normalized. If there are n documents and k topics, this will represent a (n, k) matrix with normalized scores.
