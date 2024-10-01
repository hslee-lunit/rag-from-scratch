from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class LLMManager:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

    def generate_answer(self, context, question):
        template = """Answer the following question based on the context provided.

Context:
{context}

Question:
{question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        formatted_prompt = prompt.format(context=context, question=question)
        response = self.llm(formatted_prompt)
        return StrOutputParser().parse(response)
