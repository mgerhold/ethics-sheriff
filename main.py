import json
from enum import auto
from enum import Enum
from typing import NamedTuple
from typing import Optional
from typing import Self

import nltk
from docx import Document
from nltk import sent_tokenize

INPUT_PATH = "example.docx"
OUTPUT_PATH = "results.json"


class GenderCategory(Enum):
    MALE = auto()
    FEMALE = auto()
    NEUTRAL = auto()
    INCLUSIVE = auto()


class LlmSession:
    def __init__(self, *, initial_prompt: Optional[str] = None) -> None:
        # todo: setup a real session connected to a real LLM on the web
        print("Please open https://chat.openai.com now.")
        if initial_prompt is not None:
            _ = self.send_prompt(initial_prompt, ignore_response=True)

    def send_prompt(self, prompt: str, *, ignore_response: bool = False) -> str:
        # todo: send this to the LLM via its API and get back the response
        print(f"Copy this into the ChatGPT prompt: \n{prompt}")
        if ignore_response:
            return ""
        return input("Please enter the response you got back from ChatGPT:\n")


class RecordParseError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Record(NamedTuple):
    sentence: str
    gender_categories: list[GenderCategory]

    @classmethod
    def from_response(cls, llm_session: LlmSession, sentence: str) -> Self:
        response = llm_session.send_prompt(sentence).strip()
        if not response.startswith("[") or not response.endswith("]"):
            raise RecordParseError(response)
        try:
            categories = [
                GenderCategory[category] for parts in response[1:-1].split(",")
                if len(category := parts.strip()) > 0
            ]
            return cls(sentence, categories)
        except KeyError as e:
            raise RecordParseError(f"Error while parsing key: {e}")

    def into_dict(self) -> dict[str, str | list[GenderCategory]]:
        return {
            "sentence": self.sentence,
            "gender_categories": [category.name for category in self.gender_categories],
        }


def main() -> None:
    nltk.download("punkt")
    text = "\n".join(paragraph.text for paragraph in Document(INPUT_PATH).paragraphs)
    sentences = sent_tokenize(text, language="german")
    session = LlmSession(initial_prompt=open("priming_prompt.txt", encoding="utf-8").read())
    records = [Record.from_response(session, sentence) for sentence in sentences[:3]]
    male_count = sum([sum([1 for category in record.gender_categories if category == GenderCategory.MALE]) for record in records])
    result = {
        "total_num_male": male_count,
        "records": [record.into_dict() for record in records],
    }
    json_result = json.dumps(result, indent=4)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(json_result)


if __name__ == "__main__":
    main()
