# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation, or RAG, combines a search step with a large language model. Instead of relying only on what
the model memorised during training, a RAG system first retrieves relevant passages from a document collection and then
gives those passages to the model as context when it writes the answer.

A typical RAG pipeline has five steps. First, documents are split into chunks of a few hundred words. Second, each chunk is
converted into an embedding, which is a numeric vector that captures its meaning. Third, the vectors are stored in a vector
database or index. Fourth, when a user asks a question, the question is embedded and the most similar chunks are found using
cosine similarity. Fifth, the retrieved chunks are inserted into the prompt and the language model generates an answer
grounded in them.

RAG reduces hallucinations because the model is told to answer only from the supplied context, and it lets answers stay
up to date without retraining: you simply update the documents. It also makes answers verifiable, because the system can cite
which chunks it used. Common weaknesses are poor chunking, retrieval that misses the right passage, and models that ignore
the context. Overlapping chunks help keep sentences that straddle a boundary retrievable.
