An important aspect of my solution is whether the topics make sense. Otherwise, [[Adding components adds complexity]] and we're better off just using the regular model. The question is to whom it should make sense? 

The easiest answer would be that it made sense to the data scientist developing the model. Then you could just judge the quality yourself, which allows for vastly greater iteration speed. 

At the other end of the spectrum, you could optimize so *everyone* understands it. This is also not the right solution. There might be a lot of intricate structures in the domain you're working in that would get lost on laypeople. 

The golden middle way is to optimize for *domain user understanding*. This assures you are as close to the end-user as possible, which ultimately makes your application more likely to be used as a) the user is more likely to understand it and b) the users might feel more buy-in (if they have been involved in the process). 

This does come at a cost, namely that it is slower and more expensive to involve the users. Some of this can be circumvented with [[Data Scientists can assure minimum quality]].

(NOTE: Remember to add this to the [[Data Pipeline]])