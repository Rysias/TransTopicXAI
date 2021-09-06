The more components there are in a solution, the more complicated it gets. This is in some sense the fundamental curse of adding explanations on top of models (contra using intrinsically interpretable models). For the explanations to be worth it, two things have to be fulfilled: 1. There are no available equally performant end-user interpretable models and 2. The added explainability outweighs the loss in performance. 

For many tasks it is surprisingly difficult to beat the performance of a simple regression with well-chosen features. This is especially true for supervised learning on flat datasets (e.g. churn prediction). However, on high-dimensional unstructured data (such as language and images) the simple models falter; you lose too much of the underlying structure when, say, flattening an image or turning text into bag of words. In these cases, you might have to resort to using some kind of neural network based model. 

Another important point is that it also matters to whom the model is interpretable. As I have [previously discussed](https://becominghuman.ai/can-explainable-ai-be-automated-1b4f09a9a0e7), a model that is interpretable to a data scientist is not necessarily interpretable to a non-technical end-user. 

This leads us to the second criteria, namely, that the increase in explainability should offset the drop in performance. Basically, this means that you have qualitive (i.e. [[Data Scientists can assure minimum quality]] and [[Topic Evaluations depends on Domain Experts]]) and quantitative (i.e. [[Explanatory models should statistically match base models]]) checks in place. 

The tradeoff is of course not all or nothing. Say that you have a huge black box transformer that achieves an AUC of 0.90 on some task. Adding the explainability might yank the AUC down to 0.82 but be understandable by the users. If the intrinsically interpretable model (say, a BoW or TF-IDF) has a performance of 0.75 the trade-off would definitely still be worth it. 



(NOTE: I might have to atomize this into separate points for easier future references)

