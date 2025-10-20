from typing import Iterable, Mapping, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class ExternalLLMError(RuntimeError):
    pass


class ExternalLLMClient:
    def __init__(self, *, model: str = "gpt-5", api_key: Optional[str] = None) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        **kwargs: object,
    ) -> dict:
        try:
            response = self._create_completion(
                messages=list(messages),
                model=model or self.model,
                **kwargs,
            )
        except Exception as err:
            raise ExternalLLMError(str(err)) from err

        choice = response.choices[0].message
        return {
            "content": choice.content,
            "raw": response.model_dump(mode="json"),
        }

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def _create_completion(self, **payload: object):
        return self._client.chat.completions.create(**payload)
