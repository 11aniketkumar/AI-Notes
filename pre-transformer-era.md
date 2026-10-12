[Transformer](./index.md) > Pre-Transformer Era

# The Pre-Transformer Era

## Previous Embedding Approaches

Before Transformers, several techniques were used to represent text numerically so that machine learning models could process it.

### 1. One-Hot Encoding
- Represents each word as a vector of zeros, with exactly one position set to `1`.
- **Limitation:** Produces high-dimensional, sparse vectors and fails to capture semantic relationships between words.

### 2. Bag of Words (BoW)
- Represents text using the frequency of words appearing in a document, ignoring their order.
- **Limitation:** Loses word order and contextual information, making sentences with different meanings potentially appear identical.

### 3. TF-IDF (Term Frequency–Inverse Document Frequency)
- Assigns importance to words based on how frequently they appear in a document relative to their frequency across all documents.
- **Limitation:** Captures word importance but not semantic meaning, word order, or contextual relationships.

### 4. Word Embeddings (Word2Vec, GloVe, FastText)
- Represents words as dense, low-dimensional vectors, where semantically similar words tend to have similar representations.
- Unlike previous approaches, word embeddings can capture semantic relationships between words.
- **Limitation:** Traditional word embeddings are static, meaning each word receives the same vector regardless of its context or the task being performed.

## Limitations of Traditional Word Embeddings

The major limitation of traditional word embeddings is their inability to generate **contextual and task-specific representations**.

Consider these two phrases:
- `money bank grows`
- `river bank flows`

Here, the word `bank` has different meanings:
- In the first phrase, it refers to a financial institution.
- In the second phrase, it refers to the land alongside a river.

However, traditional word embeddings assign the **exact same embedding vector** to `bank` in both phrases, regardless of its meaning.

Ideally, the embedding of a word should depend on the surrounding words, allowing the model to capture its meaning within a particular context.

This limitation motivated the development of **contextual embeddings**, which later became a central capability of Transformer-based models.