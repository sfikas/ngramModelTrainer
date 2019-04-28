# ngramModelTrainer

[![Build Status](https://travis-ci.org/sfikas/ngramModelTrainer.svg?branch=master)](https://travis-ci.org/sfikas/ngramModelTrainer)

Learns an n-gram language model given a corpus. The corpus should be text file, with a single word per line, containing no inter-word spaces.

The learned quantities are:
* Probabilities of unigrams, p( g<sub>i</sub> )
* Probabilities of bigrams, p( g<sub>i</sub> | g<sub>i-1</sub> )
* Probabilities of trigrams, p( g<sub>i</sub> | g<sub>i-1</sub>, g<sub>i-2</sub> )

## Testing and running

Test the script by running with no argument:
```
python3 ngramModelTrainer
```

Use the -h flag for details on how to use the tool with proper input:
```
python3 ngramModelTrainer -h
```



There are a few example inputs on ```fixtures/```.

The output is saved as four MATLAB matrices.

* unigrams: u(i) stands for p(i).
* bigrams: b(i, j) stands for p(j | i).
* trigrams: t(i, j, k) stands for p(k | j, i).
* quadgrams (tetragrams): q(i, j, k, l) stands for p(l | k, j, i).


## Alphabet

An alphabet of specific acceptable unigrams is required to be defined.
By default, we are using an alphabet of 36 possible letters/digits.
These are held in a python list called 'alphabet', in the following order:

* Positions 0-25: Latin *lowercase* alphabet letters, in standard alphabetical order.
* Positions 26-35: Digits 0-9.

### 'Alternative' alphabets

Non-'standard' versions of the above alphabet may be used.
These include: *dutta_extended*: a number of extra characters (these are notably encodings of the characters and punctuation found in the George Washington handwritten document set). *sophia*: polytonic greek characters. *dummy*: a limited testing set of 3 characters
