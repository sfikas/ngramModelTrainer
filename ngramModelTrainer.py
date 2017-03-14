import sys
import fileinput
import numpy as np
from collections import Counter

alphabet = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
]

def strip(word):
    # Accepts a string as input, 
    # returns a string where only characters in alphabet remain.
    res = []
    for c in word.lower():
        if c in alphabet:
            res += c
    return ''.join(res)

def normalize(a):
    a 

def countUnigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)):
        res[l[i]] += 1
    return res

def computeUnigramsPdf(unigrams):
    res = np.zeros(len(alphabet))
    for i in range(0, len(alphabet)):
        res[i] = unigrams[alphabet[i]]
    res = res / res.sum() #np.linalg.sum(res)
    return res

def countBigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-1):
        res[(l[i], l[i+1])] += 1
    return res

def computeBigramsPdf(bigrams):
    # This is the (joint) probability of bigrams
    res = np.zeros( (len(alphabet), len(alphabet)) )
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            res[i] = bigrams[alphabet[i], alphabet[i]]
    res = res / res.flatten().sum()
    return res

def countTrigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-2):
        res[(l[i], l[i+1], l[i+2])] += 1
    return res

def computeTrigramsPdf(trigrams):
    # This is the (joint) probability of trigrams
    res = np.zeros( (len(alphabet), len(alphabet), len(alphabet)) )
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            for k in range(0, len(alphabet)):
                res[i] = trigrams[alphabet[i], alphabet[i], alphabet[i]]
    res = res / res.flatten().sum()
    return res
    

if __name__ == "__main__":
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    for word in fileinput.input([sys.argv[1]]):
        strippedword = strip(word)
        unigrams += countUnigrams(strippedword)
        bigrams += countBigrams(strippedword)
        trigrams += countTrigrams(strippedword)
    unigramsPdf = computeUnigramsPdf(unigrams)
    bigramsPdf = computeBigramsPdf(bigrams)
    trigramsPdf = computeTrigramsPdf(trigrams)
    print(unigramsPdf)
    print(bigramsPdf)
    print(trigramsPdf)