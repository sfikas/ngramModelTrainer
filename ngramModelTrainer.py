import sys
import fileinput
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

def countUnigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)):
        res[l[i]] += 1
    return res

def countBigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-1):
        res[(l[i], l[i+1])] += 1
    return res

def countTrigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-2):
        res[(l[i], l[i+1], l[i+2])] += 1
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
    print(unigrams)
    print(bigrams)
    print(trigrams)