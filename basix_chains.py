from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



prompt  = ChatPromptTemplate.from_template("Explain the philosophical concepts {topic}") #system behaviour
model = ChatOpenAI(model="gpt-4o") # model
parser = StrOutputParser() #output

chain = prompt | model | parser

result = chain.invoke({"topic":"Plato's Love"})


print(result)

