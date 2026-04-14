from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#model = ChatOpenAI(model="gpt-4o")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=1024
)

# Proper prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert email writer"),
    ("human", "Write a {tone} email to {company} expressing interest in the {position}, mention the {skill} as key strength and keep it 5 lines maximum")
])

# Invoke with variables
prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "Genpact",
    "position": "AI Engineer",
    "skill": "Agentic AI"
})

result = model.invoke(prompt)
print(result.content)