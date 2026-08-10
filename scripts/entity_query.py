# Tradewinds server: bridges the Thornfield editor to a *separate*, headless
# flecs world.
#
# Thornfield speaks msgpack over ZeroMQ (like every other Tradewinds server);
# the headless world speaks newline-terminated query strings over TCP and
# answers with JSON (see paphos/python_query). This sits between them so the
# editor never has to know about the second world's transport, and so the query
# language stays flecs' own -- "Person", "Room", "age > 25, Person",
# "spatial:radius(100,100,75)" all pass straight through.
#
#   -> {"type": "query", "expr": "Person"}
#   <- {"status": "OK",
#       "names":      ["Alice", "Bob", ...],
#       "tags":       [["TestTag"], [], ...],
#       "components": [["Salary", "Person"], ...],
#       "relations":  [["LocatedIn\tRoom101"], ...],
#       "count":      8}
#
# All four lists are index-aligned -- a row is one index into each. Tags,
# components and relations stay apart rather than being flattened into one
# caption because the editor draws each as a different kind of badge, and a
# relation in particular needs its relation and target separable.
import datetime
import json
import os
import socket
import sys

import zmq
import msgpack

SOCKET_ADDR = os.environ.get(
    "THORNFIELD_QUERY_SOCKET", "ipc:///tmp/thornfield_entity_query_socket")

# The headless flecs world. python_query/src/main.cpp binds 8001; Thornfield's
# own in-process query server is on 8005, so pointing at that instead makes the
# editor browse itself, which is a useful thing to be able to do.
WORLD_HOST = os.environ.get("THORNFIELD_WORLD_HOST", "localhost")
WORLD_PORT = int(os.environ.get("THORNFIELD_WORLD_PORT", "8001"))

# The far end truncates its own response at a fixed buffer, so asking for
# everything in one query silently loses the tail. Kept as a ceiling we can
# report against rather than a limit we impose.
RESPONSE_LIMIT = 1 << 20


def query_world(expr, timeout=5.0):
    """Sends `expr` to the headless world, returns its parsed JSON.

    Raises ConnectionError when the world is not up, which is the common case
    -- it is a separate process the user starts independently.
    """
    with socket.create_connection((WORLD_HOST, WORLD_PORT), timeout=timeout) as sock:
        sock.sendall(expr.encode("utf-8") + b"\n")

        chunks = []
        total = 0
        while total < RESPONSE_LIMIT:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)

    payload = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not payload:
        raise ValueError("empty response from world")
    return json.loads(payload)


# Field separator inside a relation or component entry. Tab rather than ':'
# because flecs paths use colons ("core::Position"), and both relation targets
# and component values frequently contain them.
RELATION_SEP = "\t"


def format_value(value):
    """A component member's value as one short display string.

    Reflection gives back whatever the member's type is, and the interface has
    one badge to put it in, so nested structures collapse to compact JSON rather
    than being expanded -- a badge is the wrong shape for a tree.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Trim the trailing zeros reflection produces for whole floats, so a
        # position reads "10" rather than "10.000000".
        return f"{value:g}"
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


# Components whose numeric members are instants rather than quantities. Held as
# a set rather than tested by name suffix, so a member called "time" on some
# unrelated component is not silently reinterpreted as a date.
TIMESTAMP_COMPONENTS = {"CreatedAt"}


def format_timestamp(value):
    """An epoch seconds value as a date a person would want to read.

    The world stores the instant, not this string -- keeping the number as the
    representation is what lets it be compared and sorted; this is only how it
    is shown.
    """
    try:
        moment = datetime.datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError, OverflowError):
        return format_value(value)

    # Day-first without a leading zero, and 24h: unambiguous, and short enough
    # to sit in a badge.
    return moment.strftime("%-d %b %Y, %H:%M")


def component_entry(type_name, value):
    """One component flattened to "Type<TAB>member<TAB>value<TAB>member<TAB>value".

    Flat rather than nested because the editor already parses relations this way,
    and a flat list of strings survives msgpack round-tripping without depending
    on nested-container conversion on the C++ side.
    """
    fields = [str(type_name)]

    is_timestamp = str(type_name) in TIMESTAMP_COMPONENTS

    if isinstance(value, dict):
        for member, member_value in value.items():
            fields.append(str(member))
            fields.append(format_timestamp(member_value) if is_timestamp
                          else format_value(member_value))
    elif value is not None:
        # A component with no reflected members still has a value worth showing
        # (an opaque string, say); give it one unnamed field.
        rendered = format_value(value)
        if rendered:
            fields.append("")
            fields.append(rendered)

    return RELATION_SEP.join(fields)


def describe(entity):
    """Splits an entity into the three things the inspector draws differently.

    Returns (tags, components, relations). A tag is a bare badge, a component is
    a block holding a badge per member, and a relation is an arrow block holding
    its target -- so each has to stay structured rather than being flattened
    into a caption here.
    """
    tags = [str(t) for t in (entity.get("tags") or [])]
    components = [component_entry(name, value)
                  for name, value in (entity.get("components") or {}).items()]

    relations = []
    for relation, target in (entity.get("pairs") or {}).items():
        # A pair may carry one target or several, depending on how many the
        # entity holds for that relation.
        targets = target if isinstance(target, list) else [target]
        for t in targets:
            relations.append(f"{relation}{RELATION_SEP}{t}")

    return tags, components, relations


def entity_name(entity, index):
    """Display label for an entity.

    A named entity uses its name. An anonymous one -- which is what everything
    the editor creates is, since a sentence is not a path-safe name -- falls
    back to its Text component, so a chat message reads as its message rather
    than as "#608". The id is the last resort.
    """
    name = entity.get("name")
    # flecs writes "#<id>" as the name of an anonymous entity; that is an id in
    # disguise, not a label, so it should not win over real content.
    if name and not str(name).startswith("#"):
        return str(name)

    text = ((entity.get("components") or {}).get("Text") or {}).get("value")
    if text:
        return str(text)

    ident = entity.get("id") or name
    return str(ident) if ident else f"<entity {index}>"


def create_in_world(kind, text):
    """Creates one entity of type `kind` carrying `text` in the headless world."""
    try:
        payload = query_world(f"create {kind} {text}")
    except (ConnectionRefusedError, socket.timeout, OSError) as exc:
        return {"status": "OFFLINE", "error": f"{WORLD_HOST}:{WORLD_PORT}: {exc}"}
    except (ValueError, json.JSONDecodeError) as exc:
        return {"status": "ERROR", "error": f"bad response: {exc}"}

    if isinstance(payload, dict) and "error" in payload:
        return {"status": "ERROR", "error": str(payload["error"])}
    return {"status": "OK", "id": str(payload.get("id", ""))}


def handle_query(expr):
    try:
        payload = query_world(expr)
    except (ConnectionRefusedError, socket.timeout, OSError) as exc:
        # Not an error the editor should treat as fatal: the world is simply
        # not running yet, and the panel should say so and stay usable.
        return {"status": "OFFLINE", "error": f"{WORLD_HOST}:{WORLD_PORT}: {exc}"}
    except (ValueError, json.JSONDecodeError) as exc:
        return {"status": "ERROR", "error": f"bad response: {exc}"}

    if isinstance(payload, dict) and "error" in payload:
        return {"status": "ERROR", "error": str(payload["error"])}

    results = payload.get("results") or []
    names = [entity_name(e, i) for i, e in enumerate(results)]
    # The world's own entity ids, so the editor can refer back to an entity
    # rather than to a position in this particular result list.
    ids = [str(e.get("id", "")) for e in results]

    tags, components, relations = [], [], []
    for entity in results:
        t, c, r = describe(entity)
        tags.append(t)
        components.append(c)
        relations.append(r)

    return {
        "status": "OK",
        "names": names,
        "ids": ids,
        # All index-aligned with names; each is a list-per-entity.
        "tags": tags,
        "components": components,
        "relations": relations,
        "count": len(names),
        # Set when a name in the query resolves to nothing in this world. Not an
        # error -- the result is a legitimately empty list -- but worth showing,
        # since it is almost always a typo or a type that has no instances yet.
        "unknown": str(payload.get("unknown", "")),
        # What the world itself claimed, so a truncated response is visible
        # rather than silently short.
        "reported": int(payload.get("count", len(names))),
    }


def main():
    context = zmq.Context()
    socket_rep = context.socket(zmq.REP)
    socket_rep.bind(SOCKET_ADDR)
    print(f"[entity_query] listening on {SOCKET_ADDR}", file=sys.stderr, flush=True)
    print(f"[entity_query] world at {WORLD_HOST}:{WORLD_PORT}", file=sys.stderr, flush=True)

    while True:
        req = msgpack.unpackb(socket_rep.recv(), raw=False)

        if req.get("type") == "create":
            kind = (req.get("kind") or "").strip()
            text = (req.get("text") or "").strip()
            if not kind or not text:
                socket_rep.send(msgpack.packb({"status": "ERROR", "error": "create needs kind and text"}))
                continue
            try:
                # Newlines would terminate the request early, so a multi-line
                # message is flattened rather than truncated.
                reply = create_in_world(kind, " ".join(text.split("\n")))
            except Exception as exc:
                reply = {"status": "ERROR", "error": str(exc)}
            socket_rep.send(msgpack.packb(reply))
            continue

        if req.get("type") == "teach_forms":
            # A committed paradigm: each line is "Type<TAB>text", each becomes
            # a Form entity of that type -- types created on demand, arbitrary.
            # If one row is typed Singular, every other row relates to it by
            # LemmaOf; the lemma convention is the one piece of fixed
            # vocabulary, because the morphology server pivots on it.
            entries = []
            for line in (req.get("forms") or "").split("\n"):
                if "\t" not in line:
                    continue
                kind, text = line.split("\t", 1)
                if kind.strip() and text.strip():
                    entries.append((kind.strip(), text.strip()))
            if not entries:
                socket_rep.send(msgpack.packb(
                    {"status": "ERROR", "error": "teach_forms needs Type\\tText lines"}))
                continue
            try:
                # Rows sharing one text are ONE entity wearing several types --
                # that is what an override like "pants: Plural AND Singular"
                # means. First occurrence creates; later ones add tags.
                created = []          # (type, id)
                id_by_text = {}
                for kind, text in entries:
                    if text in id_by_text:
                        query_world(f"tag {kind} {id_by_text[text]}")
                        created.append((kind, id_by_text[text]))
                        continue
                    result = query_world(f"create {kind} {text}")
                    if result.get("id"):
                        id_by_text[text] = result["id"]
                        created.append((kind, result["id"]))
                # A form that took more than one type is a disjunction; the
                # operator it invokes is a first-class named entity, ensured
                # exactly once however many annotations mention it.
                from collections import Counter
                text_counts = Counter(t for _, t in entries)
                if any(n > 1 for n in text_counts.values()):
                    query_world("ensure Operator Or")

                singular_id = next((i for k, i in created if k == "Singular"), None)
                if singular_id:
                    for kind, ident in created:
                        if ident != singular_id:
                            query_world(f"relate LemmaOf {ident} {singular_id}")
                # Second pivot, same shape: a Contraction row relates to its
                # Expansion row. "Here's" -ExpansionOf-> "Here is".
                contraction_id = next((i for k, i in created if k == "Contraction"), None)
                expansion_id = next((i for k, i in created if k == "Expansion"), None)
                if contraction_id and expansion_id and contraction_id != expansion_id:
                    query_world(f"relate ExpansionOf {contraction_id} {expansion_id}")
                reply = {"status": "OK", "count": len(created)}
            except Exception as exc:
                reply = {"status": "OFFLINE", "error": str(exc)}
            socket_rep.send(msgpack.packb(reply))
            continue

        if req.get("type") == "branch":
            # A committed paraphrase branch: the annotator re-worded a span of
            # a chat message into a more explicit form ("my life" -> "life of
            # Wesley"). The paraphrase is a first-class entity related
            # ExplicationOf to the Chat entity it explicates, so the
            # said/implied distinction is browsable graph structure -- and
            # accruing training data for a future nominalization layer.
            message = " ".join((req.get("message") or "").split())
            span = (req.get("span") or "").strip()
            paraphrase = (req.get("paraphrase") or "").strip()
            if not message or not paraphrase:
                socket_rep.send(msgpack.packb(
                    {"status": "ERROR", "error": "branch needs message and paraphrase"}))
                continue
            try:
                result = query_world(f"create Paraphrase {paraphrase}")
                pid = result.get("id")
                # The chat entity is found by its text; expansion may have
                # rewritten the annotated surface, so an exact miss falls
                # back to leaving the Paraphrase unanchored rather than
                # anchoring it to the wrong message.
                chat_id = None
                for entity in query_world("Chat").get("results", []):
                    text = ((entity.get("components") or {}).get("Text") or {}).get("value")
                    if text and " ".join(text.split()) == message:
                        chat_id = entity.get("id")
                        break
                if pid and chat_id:
                    query_world(f"relate ExplicationOf {pid} {chat_id}")
                reply = {"status": "OK", "id": pid, "chat": chat_id, "span": span}
            except Exception as exc:
                reply = {"status": "OFFLINE", "error": str(exc)}
            socket_rep.send(msgpack.packb(reply))
            continue

        if req.get("type") == "teach_false":
            # A negative demonstration: the machine derived `variant` from
            # `word` and the user rejected it. Both become neutral Form
            # entities and the rejection is a NotLemmaOf relation -- the
            # negativity lives in the relation, not on the form, because
            # "pant" is only wrong as the singular OF PANTS, not wrong per se.
            word = (req.get("word") or "").strip()
            variant = (req.get("variant") or "").strip()
            if not word or not variant:
                socket_rep.send(msgpack.packb(
                    {"status": "ERROR", "error": "teach_false needs word and variant"}))
                continue
            relation = (req.get("relation") or "NotLemmaOf").strip()
            if relation not in ("NotLemmaOf", "NotExpansionOf"):
                relation = "NotLemmaOf"
            try:
                r1 = query_world(f"create Form {word}")
                r2 = query_world(f"create Form {variant}")
                if r1.get("id") and r2.get("id"):
                    query_world(f"relate {relation} {r1['id']} {r2['id']}")
                    reply = {"status": "OK"}
                else:
                    reply = {"status": "ERROR", "error": "create failed"}
            except Exception as exc:
                reply = {"status": "OFFLINE", "error": str(exc)}
            socket_rep.send(msgpack.packb(reply))
            continue

        if req.get("type") != "query":
            socket_rep.send(msgpack.packb({"status": "READY_WAITING"}))
            continue

        # Empty is passed through, not rejected: the world reads it as "every
        # entity". Guarding here would make the panel's empty query bar an error
        # instead of the listing it is meant to be.
        expr = (req.get("expr") or "").strip()

        try:
            reply = handle_query(expr)
        except Exception as exc:  # keep serving the next request
            print(f"[entity_query] query failed: {exc}", file=sys.stderr, flush=True)
            reply = {"status": "ERROR", "error": str(exc)}

        socket_rep.send(msgpack.packb(reply))


if __name__ == "__main__":
    main()
