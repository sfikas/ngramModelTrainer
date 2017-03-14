# languageModelTrainer

Learns a language model given a corpus.

The learned quantities are:
* Probabilities of unigrams, p(g_i)
* Probabilities of bigrams, p(g_i|g_i-1)
* Probabilities of trigrams, p(g_i|g_i-1,g_i-2)
