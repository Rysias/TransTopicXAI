An important thought on the [[Data Pipeline]]. Distance calculations are a central aspect of which documents are ultimately selected. Therefore, it is important which way these are calculated. Basically, there are two options

1. Calculate the distances on the full embeddings
2. Calculate the distances on the reduced embeddings. 

(1) intuitively uses more information and should therefore probably be better. However, there are two reasons why that might not be the case: Firstly, it is more computationally expensive, because of the additional information. This also holds for storing cost. Secondly (and most importantly), it is not necessarily a just representation of how the topics were created. Therefore, I think (2) is the better solution. 

Of course, (2) also has downsides, namely in that it is an additional component and [[Adding components adds complexity]]. 