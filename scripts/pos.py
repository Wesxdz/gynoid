# Tradewinds server: part-of-speech tagging over spaCy.
#
# The POS layer is the gate the morphology rules have been waiting for --
# each synthesized rule carries an empty constraints slot ("applies only to
# nouns") -- but its first job is display: every chat sentence gets a default
# POS tier, one tag per whitespace word, rendered above the sentence in the
# Interlocutor panel as language_2-style tiles. Defaults are proposals: they
# annotate nothing in the world until adopted.
#
# The model is the polylm POS work (en_core_web_trf and siblings) behind the
# standard msgpack/ZeroMQ REP socket. spaCy tokenizes finer than the
# annotator does ("dog." -> dog + .), so replies are re-aligned to whitespace
# chunks: each chunk answers with the POS of its first non-punctuation spaCy
# token, which keeps the reply index-parallel with the annotator's words.
#
#   -> {"type": "tag", "text": "Mice love peanut butter."}
#   <- {"status": "OK", "words": ["Mice", "love", "peanut", "butter."],
#       "tags":  ["NOUN", "VERB", "NOUN", "NOUN"]}
import os
import sys

import zmq
import msgpack
import spacy

SOCKET_ADDR = os.environ.get(
    "THORNFIELD_POS_SOCKET", "ipc:///tmp/thornfield_pos_socket")


def load_model():
    """Most accurate model available wins; the wheel set lives in polylm."""
    candidates = [os.environ.get("THORNFIELD_POS_MODEL"),
                  "en_core_web_trf", "en_core_web_lg", "en_core_web_sm"]
    for name in candidates:
        if not name:
            continue
        try:
            # POS needs the tagger (and, for trf, the transformer feeding it)
            # plus attribute_ruler; the rest is latency.
            nlp = spacy.load(name, disable=["parser", "lemmatizer", "ner"])
            print(f"[pos] loaded {name}", file=sys.stderr, flush=True)
            return nlp
        except Exception:
            continue
    raise RuntimeError("no spaCy model available; pip install en_core_web_sm")


def tag_chunks(nlp, text):
    """One (word, tag) per whitespace chunk of `text`, index-aligned.

    A chunk usually spans several spaCy tokens ("butter." -> butter + .).
    The chunk's tag is its first non-PUNCT token's, so "dog." reads NOUN,
    while a chunk that is only punctuation ("--") keeps PUNCT.
    """
    doc = nlp(text)
    words, tags = [], []
    offset = 0
    for chunk in text.split(" "):
        start, end = offset, offset + len(chunk)
        offset = end + 1
        if not chunk:
            continue
        best = None
        for token in doc:
            if token.idx + len(token.text) <= start or token.idx >= end:
                continue
            if best is None or (best.pos_ == "PUNCT" and token.pos_ != "PUNCT"):
                best = token
        words.append(chunk)
        tags.append(best.pos_ if best is not None else "X")
    return words, tags


def main():
    nlp = load_model()

    context = zmq.Context()
    zsock = context.socket(zmq.REP)
    zsock.bind(SOCKET_ADDR)
    print(f"[pos] listening on {SOCKET_ADDR}", file=sys.stderr, flush=True)

    while True:
        req = msgpack.unpackb(zsock.recv(), raw=False)
        if req.get("type") == "tag":
            text = req.get("text") or ""
            try:
                words, tags = tag_chunks(nlp, text)
                reply = {"status": "OK", "words": words, "tags": tags}
            except Exception as error:
                reply = {"status": "ERROR", "error": str(error)}
            zsock.send(msgpack.packb(reply))
        else:
            zsock.send(msgpack.packb({"status": "READY_WAITING"}))


if __name__ == "__main__":
    main()
