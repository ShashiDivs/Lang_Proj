from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()
model = ChatOpenAI(model="gpt-5")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You love facts and you tell facts about {animal}"),
        ("human","Tell me {count} facts")
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}"),
        ("human","Translate the following text to {language}: {text}")
    ]
)

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "telugu"})

chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

response = chain.invoke({"animal": "cat", "count": 2})

print(response)