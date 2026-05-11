import numpy as np
import requests
import json

from abc import ABC, abstractmethod
from data import make_prompts_dict
from pathlib import Path

BASE_URL = "http://localhost:11434"
TRIALS = 1
RESULT_PATH = Path("./results")
RESULT_PATH.mkdir(exist_ok=True)

thinking_models = [
    "gemma4:e2b",
    "qwen3.5:0.8b",
    "qwen3.5:9b",
]
non_thinking_models = ["llama3.2:3b", "llama3.1:8b"]

# tuple of model_name, do_thinking
# thinking models will be run with thinking off and on
models = list(
    zip(
        map(str, np.repeat(thinking_models, 2)),
        map(bool, np.tile([True, False], len(thinking_models))),
    )
) + [(m, False) for m in non_thinking_models]


class Prompter(ABC):

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        """Prompts an LLM

        Args:
            prompt: The prompt to send.

        Returns:
            The model's response
        """
        pass


class OllamaPrompter(Prompter):

    def __init__(
        self,
        model_name: str,
        api_url: str,
        think=False,
        payload_args: dict | None = None,
    ):
        """Create an Ollama prompting instance.

        Args:
            model_name: The name of the model to use.
            api_url: The base url of the Ollama API (no routes, just url and port)
            payload_args: Additional arguments to pass in the request payload. Streaming is not supported, however. Defaults to None.

        Raises:
            ValueError: If stream=True in self.payload_args
        """
        self._model: str = model_name
        self._url: str = api_url
        self._think: bool | str = think
        self._payload_args: dict = {} if payload_args is None else payload_args
        self._history = []

        if self._payload_args.get("stream", False):
            raise ValueError("Prompter does not support streamed response.")

    @staticmethod
    def _format_msg(role, content):
        return dict(role=role, content=content)

    def prompt(self, prompt) -> str:
        self._history.append(self._format_msg("user", prompt))

        payload = dict(
            model=self._model,
            messages=self._history,
            think=self._think,
            stream=False,
            **self._payload_args,
        )

        response = requests.post(f"{self._url}/api/chat", json=payload)

        if not response.ok:
            print(response.text)
            response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise Exception(result)

        llm_response = result["message"]["content"]

        self._history.append(self._format_msg("assistant", llm_response))

        return llm_response

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history.copy()


def main():
    print("Num Trials:", TRIALS, "\nNum Models:", len(models))

    for model_name, do_thinking in models[:1]:
        print("\n\nModel:", model_name, "   Thinking:", do_thinking)

        prompter = OllamaPrompter(model_name, BASE_URL, do_thinking)
        results = make_prompts_dict()

        # loop through each category
        for category_name, questions in results.items():
            print(f"\nEvaluating category: {category_name}")

            for question in questions:

                print(
                    "\nCurrent question:",
                    question["baseline"],
                    "\nAnswer:",
                    question["answer"],
                    "\n",
                )
                for i, variant in enumerate(question["variants"]):
                    print(f"\tEvaluating variant {i}...")

                    for _ in range(TRIALS):

                        for prompt in variant["prompts"]:
                            prompter.prompt(prompt)

                        variant["responses"].append(prompter.get_history())
                        prompter.clear_history()

        save_fp = RESULT_PATH / f"{model_name.replace(':', '-')}_{do_thinking}.json"
        with open(save_fp, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
