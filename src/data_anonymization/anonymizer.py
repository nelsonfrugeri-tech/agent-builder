import langdetect

from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from faker import Faker
from presidio_anonymizer.entities import OperatorConfig

from langchain.schema import runnable

from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    fuzzy_matching_strategy,
)

from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class TextAnonymizer:
    def __init__(self, entities, lang_code, model_name):
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": lang_code, "model_name": model_name},
            ],
        }

        self.language = lang_code

        self.anonymizer = PresidioReversibleAnonymizer(
            analyzed_fields=entities,
            languages_config=nlp_config,
        )

        fake = Faker(locale="pt_BR")

        self.anonymizer.add_operators({
            "PERSON": OperatorConfig("custom", {"lambda": lambda _: fake.first_name()}),
            "LOCATION": OperatorConfig("custom", {"lambda": lambda _: fake.city()}),
            "DATE_TIME": OperatorConfig("custom", {"lambda": lambda _: fake.date()}),
        })

    def _call_openai(self, text:str) -> str:
        llm_chain = LLMChain(
            prompt=PromptTemplate(
                template="""A partir dessas informações: {Information}.
                    Me diga quem eu sou, onde nasci e quando? Para responder a data utilize o formato yyyy-mm-dd""", 
                input_variables=["Information"]
            ), 
            llm=OpenAI(model_name="gpt-3.5-turbo-instruct")
        )

        response = llm_chain.invoke(text)
        
        print(f"Resposta da OpenAI: {response}")
        
        return response["text"]

    def entity_recognition(self, text: str):
        anonymized = self.anonymizer.anonymize(text, language=self.language)
        print(f"Dados PII anonimizados: {anonymized}")
        print(f"Dados PII anonimizados em formato map: {self.anonymizer.deanonymizer_mapping}")
    
        response = self._call_openai(anonymized)
        print(f"Resposta da OpenAI com dados PII desanonimizados: {self.anonymizer.deanonymize(response)}")

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
            supported_language=self.language,
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
    anonymizer = TextAnonymizer(["PERSON", "LOCATION", "DATE_TIME"], "pt", "pt_core_news_sm")
    anonymizer.entity_recognition("Olá eu me chamo João Paulo, nasci em São Paulo no dia 2001-05-10")
    # anonymizer.detect("Olá eu me chamo Maria")
    # anonymizer.advanced_entity_recognition("Olá eu me chamo João Paulo, nasci em São Paulo no dia 10/05/2001")
    # anonymizer.own_faker("Olá eu me chamo Carlos, nasci em São Paulo no dia 10/05/2001")
