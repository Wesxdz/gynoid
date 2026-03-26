import contractions
import networkx
import sys


from flashtext import KeywordProcessor
import spacy
nlp = spacy.load("en_core_web_sm")

keyword_processor = KeywordProcessor()

import spacy

nlp = spacy.load("en_core_web_sm")

def get_phrase_subtree(head_noun, exclude_indices):
    """Captures the full phrase belonging to a noun, minus the possessor branch."""
    phrase_tokens = [
        t for t in head_noun.subtree 
        if t.i not in exclude_indices
    ]
    # Filter to ensure we don't grab the whole sentence if the head is the ROOT
    # We only want the noun phrase part
    phrase_tokens = [t for t in phrase_tokens if head_noun.ancestors]
    
    return " ".join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])

def transform_possessive(text):
    doc = nlp(text)
    
    # 1. Find all possessive markers in the sentence
    # We store them in a list so we can process them without mutating the doc yet
    possessive_pairs = []
    for token in doc:
        if token.dep_ == "poss":
            marker = next((c for c in token.children if c.tag_ == "POS"), None)
            if marker:
                # Store (Possessor, Marker, Head Noun)
                possessive_pairs.append((token, marker, token.head))

    # 2. Process them in reverse order (Inside-Out)
    # This ensures "Wesley's dreams" is handled before "Heonae's desire"
    final_text = text
    for possessor, marker, head in reversed(possessive_pairs):
        # Determine the phrase associated with the head (e.g., 'desire to make...')
        # For the inner one: 'dreams'
        # For the outer one: 'desire to make [TRANSFORMED_DREAMS] come true'
        
        # We define the 'original' string to replace
        original_span = doc[possessor.i : head.i + 1].text
        
        # Symbolic Swap
        # Using [DET] as the placeholder for the noun's determiner
        transformed = f"[DET] {head.text} of {possessor.text}"
        
        # This is a simplified string replacement for the example; 
        # in a real library, you'd swap the tokens in a list.
        final_text = final_text.replace(f"{possessor.text}{marker.text} {head.text}", transformed)

    return final_text

# Message should have the unicode string and timestamp
def process_message(text, timestamp):
    # Entry into neurosymbolic processing
    # Expand conjunction
    pass

if __name__ == "__main__":
    message = "Wittgenstein didn't have the technology to create a gynoid"
    if len(sys.argv) > 1:
        message = sys.argv[1]

    message = transform_possessive(message)

    # In order to properly label pronoun entities in sentences, we need to expand contractions
    for contraction in contractions.contractions.items():
        message = message.replace(contraction[0], contraction[1])

    

    print(message)
