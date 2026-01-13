# Given a statement and graph working memory context
# Bind entities and relationships in situ

import sys
import json
import anthropic
from api_keys import ANTHROPIC_API_KEY

def extract_knowledge_graph(text):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # 1. Define the 'tool' as your JSON schema
    # Setting 'strict': True ensures the schema is followed exactly
    tools = [
        {
            "name": "graph_extractor",
            "description": "Extract entities with word-index grounding. Assign semantically relevant colors to entities (e.g., green for plants/nature, brown for earth/wood, blue for water/sky, red for danger/fire, grey for stone/metal, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "color": {"type": "string", "description": "A hex color code (e.g., '4a7c4e' for green) that is semantically relevant to this entity. Choose colors that evoke the nature of the entity."},
                                "start_index": {"type": "integer", "description": "The 0-based index of the first word of this entity"},
                                "end_index": {"type": "integer", "description": "The 0-based index of the last word of this entity"}
                            },
                            "required": ["id", "label", "color", "start_index", "end_index"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relationship": {"type": "string"},
                                "color": {"type": "string", "description": "A hex color code for this relationship. Choose colors that evoke the nature of the relationship (e.g., warm colors for proximity, cool colors for distance)."},
                                "in-situ": {"type": "boolean", "description": "Whether the entity can be represented directly from contiguous words in the sentence or is implied. If false then do not include start_index and end_index"},
                                "start_index": {"type": "integer", "description": "The 0-based index of the first word of the words representing only the relationship type portion of the edge"},
                                "end_index": {"type": "integer", "description": "The 0-based index of the last word of the words representing only the relationship type portion of the edge"}
                            },
                            "required": ["source", "target", "relationship", "color"]
                        }
                    }
                },
                "required": ["nodes", "edges"]
            }
        }
    ]

    words = text.split()
    indexed_sentence = " ".join([f"[{i}]{word}" for i, word in enumerate(words)])

    try:
        # 2. Call the model and FORCE the tool use
        # The 'tool_choice' parameter ensures Claude ONLY responds via the tool
        message = client.messages.create(
            # model="claude-haiku-4-5-20251001",
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            # This header is required for strict/structured outputs in the current beta
            extra_headers={"anthropic-beta": "structured-outputs-2025-11-13"},
            tools=tools,
            tool_choice={"type": "tool", "name": "graph_extractor"},
            messages=[
                {"role": "user", "content": f"Process this: {indexed_sentence}"}
            ]
        )

        # 3. Extract the input from the tool_use block
        # Because we forced tool_choice, we know the response is in content[0]
        graph_data = message.content[0].input
        return json.dumps(graph_data, indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        print(extract_knowledge_graph(sys.argv[1]))
