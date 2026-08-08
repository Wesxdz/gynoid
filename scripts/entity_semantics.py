# Tradewinds server: ranks entity names by semantic proximity to a query name.
#
# Protocol (msgpack over a ZeroMQ REP socket, same shape as interlocutor.py):
#
#   -> {"type": "rank", "query": "Otter", "corpus": "Aardvark\nAlbatross\n..."}
#   <- {"status": "OK", "order": [37, 12, 4, ...]}
#
# `order` is corpus indices, nearest first, and always a permutation of the
# whole corpus -- the panel reorders rather than filters, so every entity stays
# reachable however unrelated it is.
#
# The corpus rides along with each request rather than being registered up
# front. That keeps the socket strictly stateless request/response, which is
# what REQ/REP wants, and embeddings are cached by name here so resending the
# names costs a dictionary lookup rather than a forward pass.
import os
import sys
import threading

import numpy as np
import zmq
import msgpack
import msgpack_numpy as m

m.patch()

# Overridable so a second instance (a test, a scratch run) can bind its own
# socket instead of colliding with the one the running app already owns.
SOCKET_ADDR = os.environ.get(
    "THORNFIELD_EMBED_SOCKET", "ipc:///tmp/thornfield_entity_semantics_socket")

# Any sentence-transformers model works here; swap via THORNFIELD_EMBED_MODEL.
# Ranking is a dot product over cached unit vectors, so a larger model costs
# startup time and the first encode of each new name, not the per-click ranking.
#
#   Qwen/Qwen3-Embedding-0.6B   595M, 1024d  - Apache-2.0, default
#   BAAI/bge-large-en-v1.5      335M, 1024d  - MIT
#   mixedbread-ai/mxbai-embed-large-v1
#                               335M, 1024d  - Apache-2.0
#   paraphrase-MiniLM-L3-v2      22M,  384d  - fallback, near-instant startup
#
# E5 models (intfloat/e5-*) also work but expect "query: " / "passage: "
# prefixes on the input; without them their ranking quality drops sharply.
MODEL_NAME = os.environ.get("THORNFIELD_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Left unset, sentence-transformers takes the GPU when there is one. This box
# already keeps dinov2 and the ASR models resident, so THORNFIELD_EMBED_DEVICE=cpu
# is the escape hatch when VRAM is contended -- the ranking is cached per name,
# so CPU inference costs a one-off pause per newly seen entity, not per click.
DEVICE = os.environ.get("THORNFIELD_EMBED_DEVICE") or None

# Retrieval instruction used when a search phrase arrives with no entity to
# anchor it to. Only instruction-tuned models read this; for the others the
# prefix is inert text, which costs a little relevance but breaks nothing.
SEARCH_INSTRUCT = os.environ.get(
    "THORNFIELD_SEARCH_INSTRUCT",
    "Given a search phrase, retrieve the entity names it describes",
)

model = None
is_ready = False

# name -> unit-normalised embedding. Bounded by the number of distinct entity
# names the panel has ever asked about, so it does not need eviction yet.
embedding_cache = {}

# (instruct, name) -> unit-normalised embedding. Separate from the document
# cache because the same name embeds differently once an instruction is
# attached to it.
query_cache = {}


def load_model():
    """Loads the embedding model off the main thread so the socket answers immediately."""
    global model, is_ready
    print(f"[entity_semantics] loading {MODEL_NAME}...", file=sys.stderr, flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    is_ready = True
    print(f"[entity_semantics] model loaded on {model.device}", file=sys.stderr, flush=True)


def _encode(texts):
    """Unit-normalised embeddings, so ranking is a plain dot product."""
    vectors = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed(names):
    """Document embeddings for `names`, encoding only the uncached ones."""
    missing = [n for n in names if n not in embedding_cache]
    if missing:
        for name, vector in zip(missing, _encode(missing)):
            embedding_cache[name] = vector
    return np.stack([embedding_cache[n] for n in names])


def embed_query(name, instruct):
    """The vector the corpus is ranked against. Three cases, by what is supplied.

    name only        symmetric name-to-name similarity; the document embedding,
                     so the entity is its own nearest neighbour.
    name + instruct  the relation, anchored on the entity: "lives nearby"
                     against "Giraffe". The instruction is attached in the form
                     instruction-tuned models (Qwen3-Embedding, E5-instruct) are
                     trained on, which ranks by the relation rather than by how
                     much a name resembles "Giraffe lives nearby" as a string.
    instruct only    free-text search, no anchor: the typed phrase is itself the
                     query and gets a generic retrieval instruction.
    """
    if not name and not instruct:
        raise ValueError("rank needs an entity, a query, or both")

    if not name:
        key = ("", instruct)
        if key not in query_cache:
            query_cache[key] = _encode([f"Instruct: {SEARCH_INSTRUCT}\nQuery: {instruct}"])[0]
        return query_cache[key]

    if not instruct:
        return embed([name])[0]

    key = (instruct, name)
    if key not in query_cache:
        query_cache[key] = _encode([f"Instruct: {instruct}\nQuery: {name}"])[0]
    return query_cache[key]


def rank(query, corpus, instruct=""):
    """Corpus indices ordered by relevance, nearest first. See embed_query for
    how `query` (an entity name, possibly empty) and `instruct` (typed text,
    possibly empty) combine."""
    if not corpus or (not query and not instruct):
        return []

    corpus_vectors = embed(corpus)
    query_vector = embed_query(query, instruct)

    similarity = corpus_vectors @ query_vector
    # Descending. argsort is ascending, so negate rather than reverse, which
    # keeps ties in corpus order instead of flipping them.
    return [int(i) for i in np.argsort(-similarity)]


def main():
    threading.Thread(target=load_model, daemon=True).start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(SOCKET_ADDR)
    print(f"[entity_semantics] listening on {SOCKET_ADDR}", file=sys.stderr, flush=True)

    while True:
        req = msgpack.unpackb(socket.recv(), raw=False)

        # Bound before the model exists, so early requests get told to retry
        # rather than blocking the caller's frame.
        if not is_ready:
            socket.send(msgpack.packb({"status": "LOADING"}))
            continue

        if req.get("type") != "rank":
            socket.send(msgpack.packb({"status": "READY_WAITING"}))
            continue

        query = req.get("query", "")
        instruct = req.get("instruct", "").strip()
        corpus = [n for n in req.get("corpus", "").split("\n") if n]

        # Nothing to rank against. Answered rather than ranked, so the caller
        # never receives a partial order it would have to reject.
        if not query and not instruct:
            socket.send(msgpack.packb({"status": "READY_WAITING"}))
            continue

        try:
            order = rank(query, corpus, instruct)
            socket.send(msgpack.packb({"status": "OK", "order": order}))
        except Exception as exc:  # keep the server alive for the next request
            print(f"[entity_semantics] rank failed: {exc}", file=sys.stderr, flush=True)
            socket.send(msgpack.packb({"status": "ERROR", "error": str(exc)}))


if __name__ == "__main__":
    main()
