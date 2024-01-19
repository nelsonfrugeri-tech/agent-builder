from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from faker import Faker
from presidio_anonymizer.entities import OperatorConfig

import langdetect
from langchain.schema import runnable

class TextAnonymizer:
    def __init__(self, entities):
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "pt", "model_name": "pt_core_news_sm"},
            ],
        }

        self.language = "pt"

        self.anonymizer = PresidioReversibleAnonymizer(
            analyzed_fields=entities,
            languages_config=nlp_config,
        )

    def entity_recognition(self, text: str):
        print(self.anonymizer.anonymize(text, language=self.language))

    def detect(self, text: str):
        chain = runnable.RunnableLambda(self._detect_language) | (
            lambda x: self.anonymizer.anonymize(x["text"], language=x["language"])
        )

        print(chain.invoke(text))

    def _detect_language(self, text: str) -> str:
        language = langdetect.detect(text)
        print(f"O idioma é {language}")
        return {"text": text, "language": language}
    
    def advanced_entity_recognition(self, text:str):
        self.anonymizer.add_recognizer(SpacyRecognizer(
            supported_language="pt",
            check_label_groups=[
                ({"LOCATION"}, {"placeName", "geogName"}),
                ({"PERSON"}, {"personName"}),
                ({"DATE_TIME"}, {"date", "time"}),
            ],
        ))

        print(
            self.anonymizer.anonymize(text, language=self.language)
        )

    def own_faker(self, text: str):
        fake = Faker(locale="pt_BR")

        new_operators = {
            "PERSON": OperatorConfig("custom", {"lambda": lambda _: fake.first_name()}),
            "LOCATION": OperatorConfig("custom", {"lambda": lambda _: fake.city()}),
            "DATE_TIME": OperatorConfig("custom", {"lambda": lambda _: fake.date()}),
        }

        self.anonymizer.add_operators(new_operators)

        print(
            self.anonymizer.anonymize(text, language=self.language)
        )


# Testando a classe
if __name__ == '__main__':
    anonymizer = TextAnonymizer(["PERSON", "LOCATION", "DATE_TIME"])
    anonymizer.entity_recognition("Olá eu me chamo João Paulo, nasci em São Paulo no dia 10/05/2001")
    anonymizer.detect("Olá eu me chamo Maria")
    anonymizer.advanced_entity_recognition("Olá eu me chamo João Paulo, nasci em São Paulo no dia 10/05/2001")
    anonymizer.own_faker("Olá eu me chamo Carlos, nasci em São Paulo no dia 10/05/2001")
