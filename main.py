# Loading NLTK
# import nltk
# nltk.download()

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import SnowballStemmer
ps = PorterStemmer()
lem = WordNetLemmatizer()
stem = PorterStemmer()

text = """Hola Sr. Smith, ¿cómo está hoy? El clima es genial, y la ciudad es impresionante.
El cielo es azul rosado. No deberías comer cartón"""
tokenized_text = sent_tokenize(text)
print(tokenized_text)

tokenized_word = word_tokenize(text)
print(tokenized_word)

stop_words = set(stopwords.words("spanish"))
print(stop_words)

filtered_sent = []
tokenized_sent = tokenized_word
for w in tokenized_sent:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:", tokenized_sent)
print("Filterd Sentence:", filtered_sent)

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

# print("Filtered Sentence:", filtered_sent)
# print("Stemmed Sentence:", stemmed_words)

import spacy
from spacy_spanish_lemmatizer import SpacyCustomLemmatizer

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("es_core_news_sm")

# Process whole documents
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

def normalize(text):
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if len(t) > 3 and
    t.isalpha()]
    return lexical_tokens
word_list = normalize(text)
print(word_list)

# word = "runing"
# print("Lemmatized Word:", lem.lemmatize(word, "v"))
# print("Stemmed Word:", stem.stem(word))

# import spacy
# nlp = spacy.load("es_core_news_sm")
# text = """Soy un texto. Normalmente soy más largo y más grande. Que no te engañe mi tamaño."""
# doc = nlp(text) # Crea un objeto de spacy tipo nlp
# tokens = [t.orth_ for t in doc] # Crea una lista con las palabras del texto
#
# doc = nlp(text)
# lexical_tokens = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
#
# words = [t.lower() for t in lexical_tokens if len(t) > 3 and t.isalpha()]
#

#
# spanishstemmer = SnowballStemmer("spanish")
# text = """Soy un texto que pide a gritos que lo procesen. Por eso yo canto, tú cantas, ella canta, nosotros cantamos, cantáis, cantan…"""
# tokens = normalize(text) # crear una lista de tokens
# stems = [spanishstemmer.stem(token) for token in tokens]
# print(stems)

# import spacy
# from spacy_spanish_lemmatizer import SpacyCustomLemmatizer
#
# # Load English tokenizer, tagger, parser, NER and word vectors
# nlp = spacy.load("es_core_news_sm")
#
# # Process whole documents
# text = ("Soy un texto. Normalmente soy más largo y más grande. Que no te engañe mi tamaño. Me gusta correr mucho")
# doc = nlp(text)
#
# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
#
# # Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)
#
# nlp = spacy.load("es")
# lemmatizer = SpacyCustomLemmatizer()
# nlp.add_pipe(lemmatizer, name="lemmatizer", after="tagger")
#
# for token in nlp (text): print(token.text, token.lemma_)


