import json
from typing import Any


def generate_markdown_from_conversation_json(json_data: list[dict[str, Any]]) -> str:
    """
    Generates Markdown from a JSON array of conversation turns.

    Args:
        json_data (list): A list of dictionaries, where each dictionary
                          represents a conversation turn with "speaker" and "content" keys.

    Returns:
        str: A string containing the conversation formatted in Markdown.
    """
    markdown_output = []
    for turn in json_data:
        speaker = turn.get("speaker", "Unknown")
        content = turn.get("content", "").strip()

        # Format based on speaker
        if speaker.lower() == "user":
            markdown_output.append(f"**User:**\n{content}\n")
        elif speaker.lower() == "assistant":
            markdown_output.append(f"**Assistant:**\n{content}\n")
        else:
            # Handle other potential speakers if needed, or default
            markdown_output.append(f"**{speaker}:**\n{content}\n")

    return "\n---\n\n".join(markdown_output)  # Separator between turns


if __name__ == "__main__":
    # Example JSON input (you would load this from a file in a real scenario)
    example_json_input = [
        {
            "speaker": "Assistant",
            "content": "I see the issue - I need to use the full path that includes the user directory.\nL",
            "index": 25,
            "word_count": 18,
            "char_count": 81,
            "token_count": 14,
        },
        {
            "speaker": "User",
            "content": "Great! I can see the emotional-conversation-processor directory.\nL",
            "index": 26,
            "word_count": 8,
            "char_count": 66,
            "token_count": 6,
        },
        {
            "speaker": "Assistant",
            "content": "Excellent! Now that you've confirmed you can see it, "
            "what would you like to do next?\nPerhaps navigate into it, or list its contents?",
            "index": 27,
            "word_count": 25,
            "char_count": 130,
            "token_count": 20,
        },
    ]

    # To load from a file:
    try:
        with open("conversation.json") as f:
            conversation_data = json.load(f)
    except FileNotFoundError:
        print("Error: conversation.json not found.")
        conversation_data = []  # Or handle error appropriately

    markdown_output = generate_markdown_from_conversation_json(conversation_data)
    print(markdown_output)

    # To save to a file:
    with open("conversation.md", "w") as f:
        f.write(markdown_output)
    print("\nMarkdown saved to conversation.md")
