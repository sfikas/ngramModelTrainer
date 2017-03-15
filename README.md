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

## Output

The output is saved in three files:

* unigrams.out
* bigrams.out
* trigrams.out

The format of the files is as follows. In each line a single floating-point number is stored. This corresponds to a log probability.
More specifically for each file:

### Unigrams.out

In the i<sup>th</sup> line, the unigram log-probability for the i<sup>th</sup> alphabet letter is stored, i.e. the file looks like
```
log p( a )
log p( b )
log p( c )
...
log p( 9 )
```

### Bigrams.out

In the first 36 lines, the log-probability for the i<sup>th</sup> alphabet letter given a respective alphabet letter is stored. For example, the first 36 lines would be
```
log p( a | a ) 
log p( a | b )
log p( a | c )
...
log p( a | 9 )
```

The next 36 lines would be
```
log p( b | a ) 
log p( b | b )
log p( b | c )
...
log p( b | 9 )
```

And so on.

### Trigrams.out

In the first 36x36 lines, the log-probability for the i<sup>th</sup> alphabet letter given two previous alphabet letters is stored. For example, the first 36 lines would be
```
log p( a | a , a ) 
log p( a | a , b )
log p( a | a , c )
...
log p( a | a , 9 )
```

The next 36 lines would be
```
log p( a | b , a ) 
log p( a | b , b )
log p( a | b , c )
...
log p( a | b , 9 )
```

The last of the first 36x36 lines would be
```
log p( a | 9 , a ) 
log p( a | 9 , b )
log p( a | 9 , c )
...
log p( a | 9 , 9 )
```

And with the immediately next 36 lines, starts the same cycle for the next letter.
```
log p( b | a , a ) 
log p( b | a , b )
log p( b | a , c )
...
log p( b | a , 9 )
```

And so on.