from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()
model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You love facts and you tell facts about {animal}"),
        ("human","Tell me {count} facts")
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parser_output = RunnableLambda(lambda x: x.content)


chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parser_output)

response = chain.invoke({"animal": "cat", "count": 5})

print(response)


