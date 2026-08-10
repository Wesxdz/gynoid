# Tradewinds server: plural morphology learned through demonstration.
#
# Demonstrations live in the headless flecs world as first-class structure:
# Form entities typed Plural / Singular, related by LemmaOf -- written there by
# the Lexicon panel's Enter gesture through the entity_query bridge. This
# server reads those demonstrations and induces a series of regex-like rewrite
# rules (v1 synthesis: longest-common-prefix suffix rewrites), compiled into a
# real OpenFst transducer via pynini.
#
# Nonmonotonicity is architectural, not simulated: the demonstrated pairs
# themselves are the lexicon, and the lexicon PREEMPTS the rule series --
# exact lookup first, synthesized rules only for words no demonstration
# covers. Teaching series->series (an invariant) retracts what the -ies rule
# would have concluded, for that word alone: Panini's elsewhere condition as
# control flow.
#
# Synthesized rules are mirrored back into the world as Rule entities, so the
# rule series is itself browsable and, later, gateable -- each rule carries a
# constraints slot awaiting the POS transducer ("applies only to nouns").
#
#   -> {"type": "analyze", "word": "puppies"}
#   <- {"status": "OK", "lemma": "puppy", "suffix_in": "ies",
#       "suffix_out": "y", "support": 1, "rule": "-ies>y"}
#   <- {"status": "OK", "lemma": "series", "rule": "lexical", ...}
#   <- {"status": "OK"}                      (nothing reaches this word)
#
#   -> {"type": "dump"}
#   <- {"status": "OK", "states": N, "arcs": M, "rules": [...], "lexicon": K}
import json
import os
import socket as pysocket
import sys

import zmq
import msgpack
import pynini
from pynini.lib import pynutil

SOCKET_ADDR = os.environ.get(
    "THORNFIELD_MORPHOLOGY_SOCKET", "ipc:///tmp/thornfield_morphology_socket")
WORLD_HOST = os.environ.get("THORNFIELD_WORLD_HOST", "localhost")
WORLD_PORT = int(os.environ.get("THORNFIELD_WORLD_PORT", "8001"))

MIN_LEMMA = 2
MAX_SUFFIX = 12

# Below this confidence, expansion rules still propose in the Lexicon but do
# NOT auto-rewrite sentences -- an uncertain rule silently mangling a genitive
# is worse than a contraction left folded.
EXPAND_MIN_CONFIDENCE = 0.6


def query_world(expr, timeout=5.0):
    with pysocket.create_connection((WORLD_HOST, WORLD_PORT), timeout=timeout) as sock:
        sock.sendall(expr.encode("utf-8") + b"\n")
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    payload = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not payload:
        raise ValueError("empty response")
    return json.loads(payload)


def entity_text(entity):
    return ((entity.get("components") or {}).get("Text") or {}).get("value")


def fetch_demonstrations():
    """(plural, singular) pairs from the world's Form entities.

    LemmaOf targets arrive as the target's display name -- "#612" for
    anonymous entities -- so the Singular listing doubles as the resolution
    table from those names to text.
    """
    plural = query_world("Plural")
    singular = query_world("Singular")

    singular_text = {}
    for entity in singular.get("results", []):
        text = entity_text(entity)
        if not text:
            continue
        singular_text[f"#{entity.get('id')}"] = text
        singular_text[str(entity.get("name"))] = text

    pairs = []
    plural_texts = set()
    for entity in plural.get("results", []):
        text = entity_text(entity)
        if not text:
            continue
        plural_texts.add(text.lower())
        targets = (entity.get("pairs") or {}).get("LemmaOf")
        if targets is None:
            continue
        for target in (targets if isinstance(targets, list) else [targets]):
            lemma = singular_text.get(str(target))
            if lemma:
                pairs.append((text.lower(), lemma.lower()))
    singular_texts = {t.lower() for t in singular_text.values()}

    # Negative demonstrations: rejected derivations, per layer.
    negatives = set()
    expansion_negatives = set()
    try:
        forms = query_world("Form")
        form_text = {}
        for entity in forms.get("results", []):
            text = entity_text(entity)
            if text:
                form_text[f"#{entity.get('id')}"] = text
                form_text[str(entity.get("name"))] = text
        for entity in forms.get("results", []):
            text = entity_text(entity)
            pairs_map = entity.get("pairs") or {}
            for relation, bucket in (("NotLemmaOf", negatives),
                                     ("NotExpansionOf", expansion_negatives)):
                targets = pairs_map.get(relation)
                if not text or targets is None:
                    continue
                for target in (targets if isinstance(targets, list) else [targets]):
                    rejected = form_text.get(str(target))
                    if rejected:
                        bucket.add((text.lower(), rejected.lower()))
    except Exception:
        pass
    # Expansion demonstrations: Contraction forms related ExpansionOf to
    # their Expansion forms. Same shape as the number pairs; the same
    # machinery learns from them.
    expansions = []
    try:
        contractions = query_world("Contraction")
        expansion_entities = query_world("Expansion")
        expansion_text = {}
        for entity in expansion_entities.get("results", []):
            text = entity_text(entity)
            if text:
                expansion_text[f"#{entity.get('id')}"] = text
                expansion_text[str(entity.get("name"))] = text
        for entity in contractions.get("results", []):
            text = entity_text(entity)
            targets = (entity.get("pairs") or {}).get("ExpansionOf")
            if not text or targets is None:
                continue
            for target in (targets if isinstance(targets, list) else [targets]):
                expanded = expansion_text.get(str(target))
                if expanded:
                    expansions.append((text.lower(), expanded.lower()))
    except Exception:
        pass
    return pairs, plural_texts, singular_texts, negatives, expansions, expansion_negatives


def suffix_rewrite(plural, singular):
    common = 0
    while (common < len(plural) and common < len(singular)
           and plural[common] == singular[common]):
        common += 1
    return plural[common:], singular[common:]


class Morphology:
    """The lexicon of demonstrations, and the rule series induced from them.

    apply() consults them in that order -- lexicon preempts rules -- which is
    the whole nonmonotonic architecture in one line of control flow.
    """

    def __init__(self, pairs, plural_texts=frozenset(), singular_texts=frozenset(),
                 negatives=frozenset()):
        self.negatives = set(negatives)
        # A form typed BOTH ways is an override: number-ambiguous by
        # demonstration (pants). It preempts pairs and rules alike.
        self.ambiguous = set(plural_texts) & set(singular_texts)
        self.plural_only = set(plural_texts) - self.ambiguous
        self.singular_only = set(singular_texts) - self.ambiguous
        self.lexicon = {}              # plural -> singular, exact
        self.by_singular = {}          # singular -> plural: the other tape
        rules = {}                     # suffix_in -> {"out", "support"}
        for plural, singular in pairs:
            self.lexicon[plural] = singular
            self.by_singular.setdefault(singular, plural)
            s_in, s_out = suffix_rewrite(plural, singular)
            if not s_in:
                continue               # invariants teach the lexicon only
            entry = rules.setdefault(s_in, {"out": s_out, "support": 0})
            if entry["out"] == s_out:
                entry["support"] += 1
            elif entry["support"] <= 1:
                rules[s_in] = {"out": s_out, "support": 1}
        self.rules = rules

        # Confidence in a rule = its followers against its observed violators:
        # every demonstrated pair whose plural matches the pattern but does NOT
        # follow the rewrite (an invariant, an irregular) is evidence against
        # the rule's generality. The +1 is a prior against certainty from tiny
        # data. Lexical hits don't consult this -- a demonstration about the
        # exact word is confidence 1 by definition.
        for s_in, entry in rules.items():
            followers = violators = 0
            for plural, singular in pairs:
                if not plural.endswith(s_in):
                    continue
                if plural[:len(plural) - len(s_in)] + entry["out"] == singular:
                    followers += 1
                else:
                    violators += 1
            # A rejected derivation this rule would have produced is direct
            # evidence against it.
            for w, rejected in self.negatives:
                if w.endswith(s_in) and w[:len(w) - len(s_in)] + entry["out"] == rejected:
                    violators += 1
            entry["confidence"] = followers / (followers + violators + 1)

        self.fst = None
        if rules:
            sigma = pynini.union(
                *[pynini.accep(pynini.escape(chr(c))) for c in range(32, 127)]
            ).closure().optimize()
            rewrites = []
            for s_in, entry in rules.items():
                weight = (MAX_SUFFIX - min(len(s_in), MAX_SUFFIX)) * 10 \
                         + max(0, 9 - entry["support"])
                rewrites.append(pynutil.add_weight(
                    pynini.cross(pynini.escape(s_in), pynini.escape(entry["out"])),
                    weight))
            self.fst = (sigma + pynini.union(*rewrites)).optimize()

    def apply(self, word):
        # Lexicon first, BOTH directions: a transducer runs both ways, and so
        # does what it was taught. Inspecting either form of a demonstrated
        # pair shows the whole paradigm.
        if word in self.ambiguous:
            # The override: this exact form is both. No derived variant, no
            # rule application -- the demonstration IS the whole answer.
            return {"lemma": word, "support": 1, "confidence": 1.0,
                    "rule": "lexical", "word_type": "Plural",
                    "variant": word, "variant_type": "Singular"}
        if word in self.lexicon:
            lemma = self.lexicon[word]
            s_in, s_out = suffix_rewrite(word, lemma)
            return {"lemma": lemma, "suffix_in": s_in, "suffix_out": s_out,
                    "support": 1, "confidence": 1.0, "rule": "lexical",
                    "word_type": "Plural", "variant": lemma,
                    "variant_type": "Singular"}
        if word in self.by_singular:
            plural = self.by_singular[word]
            s_in, s_out = suffix_rewrite(word, plural)
            return {"lemma": word, "suffix_in": s_in, "suffix_out": s_out,
                    "support": 1, "confidence": 1.0, "rule": "lexical",
                    "word_type": "Singular", "variant": plural,
                    "variant_type": "Plural"}
        if word in self.plural_only:
            return {"lemma": word, "support": 1, "confidence": 1.0,
                    "rule": "lexical", "word_type": "Plural",
                    "variant": "", "variant_type": ""}
        if word in self.singular_only:
            return {"lemma": word, "support": 1, "confidence": 1.0,
                    "rule": "lexical", "word_type": "Singular",
                    "variant": "", "variant_type": ""}

        if self.fst is None:
            return None
        try:
            lattice = pynini.compose(pynini.escape(word), self.fst)
            if lattice.start() == pynini.NO_STATE_ID:
                return None
            lemma = pynini.shortestpath(lattice).string()
        except pynini.FstOpError:
            return None
        if len(lemma) < MIN_LEMMA or lemma == word:
            return None
        if (word, lemma) in self.negatives:
            return None                # this exact derivation was rejected
        s_in, s_out = suffix_rewrite(word, lemma)
        entry = self.rules.get(s_in)
        return {"lemma": lemma, "suffix_in": s_in, "suffix_out": s_out,
                "support": entry["support"] if entry else 1,
                "confidence": round(entry["confidence"], 3) if entry else 0.5,
                "rule": f"-{s_in}>{s_out}",
                "word_type": "Plural", "variant": lemma,
                "variant_type": "Singular"}

    def dump(self):
        info = {"states": 0, "arcs": 0, "lexicon": len(self.lexicon)}
        if self.fst is not None:
            info["states"] = self.fst.num_states()
            info["arcs"] = sum(self.fst.num_arcs(s) for s in self.fst.states())
        info["rules"] = [{"in": s, "out": e["out"], "support": e["support"],
                          "confidence": round(e.get("confidence", 0), 3),
                          "constraints": []}
                         for s, e in sorted(self.rules.items())]
        return info


def mirror_rules(morph):
    """Synthesized rules become Rule entities, so the series is first-class --
    visible in the Entities panel, and a place for constraints to attach."""
    try:
        existing = set()
        for entity in query_world("Rule").get("results", []):
            text = entity_text(entity)
            if text:
                existing.add(text)
        for s_in, entry in morph.rules.items():
            label = f"-{s_in}>{entry['out']}"
            if label not in existing:
                query_world(f"create Rule {label}")
    except Exception:
        pass                            # world down: rules still apply locally


def main():
    context = zmq.Context()
    zsock = context.socket(zmq.REP)
    zsock.bind(SOCKET_ADDR)
    print(f"[morphology] listening on {SOCKET_ADDR}", file=sys.stderr, flush=True)

    morph = Morphology([])
    expander = Morphology([])
    known = -1

    while True:
        req = msgpack.unpackb(zsock.recv(), raw=False)

        # Demonstrations may have arrived since the last request; relearn when
        # their count moves. The world is the source of truth, not a file.
        try:
            (pairs, plural_texts, singular_texts, negatives,
             expansions, expansion_negatives) = fetch_demonstrations()
            signature = (len(pairs) + len(plural_texts) + len(singular_texts)
                         + len(negatives) + len(expansions)
                         + len(expansion_negatives))
            if signature != known:
                morph = Morphology(pairs, plural_texts, singular_texts, negatives)
                expander = Morphology(expansions, negatives=expansion_negatives)
                known = signature
                mirror_rules(morph)
                print(f"[morphology] {known} demonstrations, "
                      f"{len(morph.rules)} rules", file=sys.stderr, flush=True)
        except Exception:
            pass                        # world down: last machine keeps serving

        kind = req.get("type")
        if kind == "analyze":
            word = (req.get("word") or "").strip().lower()
            result = morph.apply(word) if word else None
            reply = {"status": "OK"}
            if result:
                reply.update(result)
            else:
                expansion = expander.apply(word) if word else None
                if expansion:
                    reply.update(expansion)
                    reply["word_type"] = "Contraction"
                    reply["variant"] = expansion["lemma"]
                    reply["variant_type"] = "Expansion"
            zsock.send(msgpack.packb(reply))
        elif kind == "expand":
            word = (req.get("word") or "").strip().lower()
            result = expander.apply(word) if word else None
            reply = {"status": "OK"}
            if result:
                reply["expansion"] = result["lemma"]
                reply["rule"] = result["rule"]
                reply["confidence"] = result["confidence"]
            zsock.send(msgpack.packb(reply))
        elif kind == "expand_text":
            # The preprocessing layer: each whitespace token is offered to the
            # expander; known contractions unfold in place, everything else
            # passes through untouched. Trailing punctuation is lifted off,
            # expanded, and re-attached, so "Here's," expands cleanly.
            text = req.get("text") or ""
            out_tokens = []
            changed = False
            for token in text.split(" "):
                core = token
                trailing = ""
                while core and not (core[-1].isalnum() or core[-1] == "'"):
                    trailing = core[-1] + trailing
                    core = core[:-1]
                result = expander.apply(core.lower()) if core else None
                if result and (result["rule"] == "lexical"
                               or result["confidence"] >= EXPAND_MIN_CONFIDENCE):
                    replacement = result["lemma"]
                    # Preserve leading capitalisation: Here's -> Here is.
                    if core[:1].isupper():
                        replacement = replacement[:1].upper() + replacement[1:]
                    out_tokens.append(replacement + trailing)
                    changed = True
                else:
                    out_tokens.append(token)
            reply = {"status": "OK", "changed": changed,
                     "text": " ".join(out_tokens)}
            zsock.send(msgpack.packb(reply))
        elif kind == "dump":
            reply = {"status": "OK"}
            number = morph.dump()
            expansion = expander.dump()
            for rule in number["rules"]:
                rule["layer"] = "number"
            for rule in expansion["rules"]:
                rule["layer"] = "expansion"
            reply["states"] = number["states"] + expansion["states"]
            reply["arcs"] = number["arcs"] + expansion["arcs"]
            reply["lexicon"] = number["lexicon"] + expansion["lexicon"]
            reply["rules"] = number["rules"] + expansion["rules"]
            zsock.send(msgpack.packb(reply))
        else:
            zsock.send(msgpack.packb({"status": "READY_WAITING"}))


if __name__ == "__main__":
    main()
