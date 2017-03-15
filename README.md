# languageModelTrainer

Learns a language model given a corpus.

The learned quantities are:
* Probabilities of unigrams, p( g<sub>i</sub> )
* Probabilities of bigrams, p( g<sub>i</sub> | g<sub>i-1</sub> )
* Probabilities of trigrams, p( g<sub>i</sub> | g<sub>i-1</sub>, g<sub>i-2</sub> )


## Alphabet

An alphabet of specific acceptable unigrams is required to be defined.
By default, we are using an alphabet of 36 possible letters/digits.
These are held in a python list called 'alphabet', in the following order:

* Positions 0-25: Latin *lowercase* alphabet letters, in standard alphabetical order.
* Positions 26-35: Digits 0-9.