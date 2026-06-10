import anthropic
import json
from copilot.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

AGENT_SYSTEM_PROMPT = """
You are a healthcare claims specialist AI with access to tools.
When a user asks about a specific claim, use the available tools to:
1. Validate ICD-CPT pairing if both codes are provided
2. Run the denial predictor if claim details are available
3. Look up ICD codes if the user describes a condition in plain English

Always explain what tools you are using and why.
After using tools, summarize the findings in plain language.
If no tools are needed (general questions), answer from your knowledge.
"""

class ClaimsAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = 'claude-sonnet-4-5'
        self.messages = []
        self.trace = []  # observability log of all tool calls

    def run(self, user_message: str) -> tuple:
        """
        Run the agent loop for a user message.
        Returns (final_response, trace_log).
        """
        self.messages.append({'role': 'user', 'content': user_message})
        turn_trace = []

        # Agent loop: call Claude, handle tool use, repeat until text response
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.messages,
            )

            if response.stop_reason == 'end_turn':
                # Claude is done — extract text response
                final_text = next(b.text for b in response.content if b.type == 'text')
                self.messages.append({'role': 'assistant', 'content': response.content})
                self.trace.append(turn_trace)
                return final_text, turn_trace

            elif response.stop_reason == 'tool_use':
                # Claude wants to call one or more tools
                self.messages.append({'role': 'assistant', 'content': response.content})
                tool_results = []

                for block in response.content:
                    if block.type == 'tool_use':
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        # Execute the tool
                        tool_fn = TOOL_FUNCTIONS.get(tool_name)
                        if tool_fn:
                            result = tool_fn(**tool_input)
                        else:
                            result = {'error': f'Unknown tool: {tool_name}'}

                        # Log to trace
                        turn_trace.append({
                            'tool': tool_name,
                            'input': tool_input,
                            'output': result,
                        })

                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': json.dumps(result),
                        })

                # Feed all tool results back to Claude
                self.messages.append({'role': 'user', 'content': tool_results})

            else:
                break

        return 'Agent loop ended unexpectedly.', turn_trace

    def reset(self):
        self.messages = []
        self.trace = []