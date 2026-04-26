import json
import re

def parse_json_from_llm(generated_text: str):
    # Regex to find the first JSON-like block (handles markdown ````json)
    match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

test1 = """
I am thinking about this.
```json
{
  "thought_process": "hello",
  "action_type": "send_message",
  "text": "world"
}
```
"""
print(parse_json_from_llm(test1))
