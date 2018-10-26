import nltk
import sys
import os.path
import tqdm

if len(sys.argv) == 1:
    print('Syntax: python3 tokenize_corpus.py <corpus raw text> <output file>')
    exit(0)

with open(sys.argv[1],'r') as fin:
    tokens = nltk.word_tokenize(fin.read())
print('Read {} words, writing to output file.'.format(len(tokens)))
if(os.path.isfile(sys.argv[2])):
    raise ValueError('The output file already exists; please erase it manually if you are sure that you want to overwrite it.')
with open(sys.argv[2], 'w') as fout:
    for item in tqdm.tqdm(tokens):
        fout.write("%s\n" % item)