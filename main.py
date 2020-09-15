import spacy
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

lem = WordNetLemmatizer()
stem = PorterStemmer()

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("es_core_news_sm")

text = """Hola Sr. Smith, ¿cómo está hoy? El clima es genial, y la ciudad es impresionante. 
    El cielo es azul rosado. No deberías comer cartón,
    corrió, cantando, bailando, yo, el, ellos, nosotros"""

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

word = "runing"
print("Lemmatized Word:", lem.lemmatize(word, "v"))
print("Stemmed Word:", stem.stem(word))

for token in nlp(text):
    print(token.text, token.lemma_, token.pos_)
