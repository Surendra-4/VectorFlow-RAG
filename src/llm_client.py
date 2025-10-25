import requests
import json
import sys
import time

class OllamaClient:
    def __init__(self, model="tinyllama", base_url="http://localhost:11434"):
        self.model = model
        self.url = f"{base_url}/api/generate"

    def generate(self, prompt: str, context: list = None,
                 max_tokens: int = 512, temperature: float = 0.7,
                 stream: bool = True) -> str:
        """
        Generate an answer using Ollama LLM with optional streaming output.

        Args:
            prompt: The input question or instruction.
            context: List of retrieved context documents.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            stream: Whether to stream output live.

        Returns:
            The complete generated response as a string.
        """
        # Combine context and question
        if context:
            full_prompt = "\n\n".join(context) + f"\n\nQuestion: {prompt}\nAnswer:"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            },
            "stream": True  # enable streaming
        }

        try:
            response = requests.post(self.url, json=payload, stream=True)
            response.raise_for_status()

            final_text = ""

            if stream:
                print("\n[Streaming LLM output...]\n")
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            token = data.get("response", "")
                            if token:
                                final_text += token
                                sys.stdout.write(token)
                                sys.stdout.flush()
                        except json.JSONDecodeError:
                            continue
                print("\n")  # newline after stream ends
            else:
                # Non-streaming fallback
                text_data = response.text.strip().split("\n")
                for chunk in text_data:
                    try:
                        data = json.loads(chunk)
                        if "response" in data:
                            final_text += data["response"]
                    except json.JSONDecodeError:
                        continue

            return final_text.strip() or "No response generated."

        except Exception as e:
            return f"[Error communicating with Ollama: {e}]"


if __name__ == "__main__":
    client = OllamaClient(model="tinyllama")
    ans = client.generate("Explain reinforcement learning in simple terms.")
    print("\n\nFinal Answer:\n", ans)
