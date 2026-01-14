# Given a statement and graph working memory context
# Bind entities and relationships in situ

import sys
import json
import anthropic
from api_keys import ANTHROPIC_API_KEY

def extract_knowledge_graph(text, known_entities=None, previous_sentences=None):
    """
    Extract knowledge graph from text.

    Args:
        text: The sentence to process
        known_entities: Optional list of known entities from previous messages
                       Each entity has: id, label, color
        previous_sentences: Optional list of previous sentences for context
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # 1. Define the 'tool' as your JSON schema
    tools = [
        {
            "name": "natural_language_to_graph_parser",
            "description": 
            """Extract entities with word-index grounding. 
            For entities that match known entities from context, use 'binds_to' to reference them. 
            Most often, use the exact words from the sentence for the label. 
            Do not change the root word in a way that would make the in-situ sentence labeling change in meaning (that includes pronouns, keep them the same word). 
            For new entities, set 'is_new' to true. 
            Assign semantically relevant colors to NEW entities (e.g., green for plants/nature, brown for earth/wood, blue for water/sky, red for danger/fire, grey for stone/metal, etc.).
            
            """,
            
            "input_schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique identifier for this node instance in this sentence"},
                                "label": {"type": "string", "description": "The label/type of this entity"},
                                "is_new": {"type": "boolean", "description": "True if this is a newly introduced entity, false if it binds to an existing entity from context"},
                                "binds_to": {"type": "string", "description": "If is_new is false, this is the id of the known entity this binds to"},
                                "color": {"type": "string", "description": "A hex color code (e.g., '4a7c4e' for green). For new entities, choose semantically relevant colors. For bindings, this can be omitted as we'll use the existing entity's color."},
                                "start_index": {"type": "integer", "description": "The 0-based index of the first word of this entity"},
                                "end_index": {"type": "integer", "description": "The 0-based index of the last word of this entity"}
                            },
                            "required": ["id", "label", "is_new", "start_index", "end_index"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sources": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of source node IDs. Use multiple sources when entities are grouped (e.g., 'The goblin and orc attack' -> sources: ['goblin', 'orc'])"
                                },
                                "targets": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of target node IDs. Use multiple targets when entities are grouped (e.g., 'attacks the knight and wizard' -> targets: ['knight', 'wizard'])"
                                },
                                "relationship": {"type": "string"},
                                "color": {"type": "string", "description": "A hex color code for this relationship. Choose colors that evoke the nature of the relationship (e.g., warm colors for proximity, cool colors for distance)."},
                                "in-situ": {"type": "boolean", "description": "Whether the relationship can be represented directly from contiguous words in the sentence or is implied. If false then do not include start_index and end_index"},
                                "start_index": {"type": "integer", "description": "The 0-based index of the first word of the words representing only the relationship type portion of the edge"},
                                "end_index": {"type": "integer", "description": "The 0-based index of the last word of the words representing only the relationship type portion of the edge"}
                            },
                            "required": ["sources", "targets", "relationship", "color"]
                        }
                    }
                },
                "required": ["nodes", "edges"]
            }
        }
    ]

    words = text.split()
    indexed_sentence = " ".join([f"[{i}]{word}" for i, word in enumerate(words)])

    # Build context from previous sentences
    context_msg = ""
    if previous_sentences and len(previous_sentences) > 0:
        context_msg = "\n\nPrevious sentences in conversation:\n"
        for i, sentence in enumerate(previous_sentences):
            context_msg += f"{i+1}. {sentence}\n"

    # Add known entities context
    if known_entities and len(known_entities) > 0:
        context_msg += "\nKnown entities from these messages:\n"
        for entity in known_entities:
            context_msg += f"- {entity['id']}: {entity['label']} (color: {entity.get('color', 'unknown')})\n"
        context_msg += "\nIf any entities in the new sentence refer to these known entities (including pronouns like 'it', 'he', 'she', 'they'), set is_new=false and binds_to=<known_entity_id>."

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            extra_headers={"anthropic-beta": "structured-outputs-2025-11-13"},
            tools=tools,
            tool_choice={"type": "tool", "name": "natural_language_to_graph_parser"},
            messages=[
                {"role": "user", "content": f"Process this sentence: {indexed_sentence}{context_msg}"}
            ]
        )

        graph_data = message.content[0].input
        return json.dumps(graph_data, indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
        # Check for optional context JSON as second argument
        known_entities = None
        previous_sentences = None
        if len(sys.argv) > 2:
            try:
                known_entities = json.loads(sys.argv[2])
            except json.JSONDecodeError:
                pass
        if len(sys.argv) > 3:
            try:
                previous_sentences = json.loads(sys.argv[3])
            except json.JSONDecodeError:
                pass
        print(extract_knowledge_graph(text, known_entities, previous_sentences))
