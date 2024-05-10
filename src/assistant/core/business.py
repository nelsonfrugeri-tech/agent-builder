import os

from openai import OpenAI

from core.event_handler import EventHandler


class Business:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        
        self.vector_store = self._load_file_batch()
        self.assistant = self._load_assistant()
        self.thread = self._get_thread()

    def instruction(self, instructions):
        try:
            with self.client.beta.threads.runs.stream(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                event_handler=EventHandler(),
                instructions=instructions,
                max_completion_tokens=250,
            ) as stream: stream.until_done()
        except Exception as e:
            print(f"*** EXCEPTION *** {e}")
            return None

    def _get_thread(self):
        return self.client.beta.threads.create()

    def _load_assistant(self):        
        return self.client.beta.assistants.create(
            instructions="You're Python expert",
            model="gpt-4-turbo",
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}},            
        )

    def _load_file_batch(self):
        try:
            # Create a vector store caled "Financial Statements"
            vector_store = self.client.beta.vector_stores.create(name="Python Documentations")
            
            # Ready the files for upload to OpenAI
            base_path = os.path.abspath(os.path.dirname(__file__))
            # Ajusta para subir um diretório antes de acessar o subdiretório 'file'
            file_base = os.path.join(base_path, "..", "file", "doc_python.pdf")
            file_paths = [os.path.abspath(file_base)]  # Normaliza o caminho
            file_streams = [open(path, "rb") for path in file_paths]
            
            # Use the upload and poll SDK helper to upload the files, add them to the vector store,
            # and poll the status of the file batch for completion.
            self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )
            
            return vector_store
        except Exception as exception:
            print(f"*** EXCEPTION *** {exception}")
            