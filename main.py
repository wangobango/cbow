from reader import SetReader
import stop_words as sw
from nltk import ngrams
from ngrammodel import Embeddings
import nltk.tokenize as tokenize

stopList = sw.get_stop_words('polish')
TOKENIZER = tokenize.NLTKWordTokenizer()

"""
@params:
    -data : Array of strings
@returns:
    -dict : vocab dict of word:number of occurances of word,  in descending order of occurances
    -vocab size : size of vocabulary
"""
def createVocab(data):
    word_to_ix = {word: i for i, word in enumerate(data)}
    ix_to_word = {ix: word for ix, word in enumerate(data)}
    return word_to_ix, ix_to_word, len(data)

def removeWordsFromStoplist(arr):
    return [x for x in arr if x not in stopList]

def createNgram(text, n):
    data = []
    if n >= int(len(text) / 2):
        raise Exception("N too huge for n-gram")

    for i in range(n, len(text) - n):
        context = [*[text[x] for x in range(i - n, i)],
                   *[text[x] for x in range(i + 1, i + 1 + n)]]
        target = text[i]
        data.append((context, target))
    return data

"""
@params:
    -array of Strings
@returns:text
    -preprocessed array of Strings
"""
def preprocess(data):
    data = removeWordsFromStoplist(data)
    return data

if __name__ == '__main__':

    # Availavle GPU or CPU
    mode = "CPU"
    train = False
    send_to_device = False

    reader = SetReader()
    trainData = reader.readDev0Train()
    arr = TOKENIZER.tokenize(trainData['data'].str.cat(sep=' '))
    arr = arr[:100]
    word_to_ix, ix_to_word, vocab_size = createVocab(arr)

    """
    params
    """

    n = 2
    embedding_dim = 100
    context_size = 2

    """
    end params
    """


    ngram = createNgram(arr, n)
    embeddings = Embeddings()
    # embeddings.trainEmbeddings(word_to_ix, ix_to_word, vocab_size, embedding_dim, context_size, ngram)
    if(train):
        embeddings.trainEmbeddingCBOW(ngram, embedding_dim, vocab_size, word_to_ix, ix_to_word, mode, send_to_device)
    else:
        embeddings.testModel(ngram, reader.readDev0Expected(), word_to_ix, ix_to_word)
