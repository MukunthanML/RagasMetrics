import asyncio
from ragas.dataset_schema import SingleTurnSample  
from ragas.metrics import LLMContextRecall
from ragas.llms import LangchainLLMWrapper 
from langchain_groq import ChatGroq 


# Initialize the ChatGroq LLM with specific parameters
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b", 
    temperature=0.0,  # Temperature for deterministic responses
    max_retries=2  # Maximum retries for API calls
    # set GROQ_API_KEY to your environment variable or uncomment the line below to set it directly
    # groq_api_key="your_groq_api_key"  # Uncomment and set your API key if needed
)

# Wrap the LLM using LangchainLLMWrapper for compatibility with Ragas
evaluator_llm = LangchainLLMWrapper(llm)


sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the capital of France.","There are many famous landmarks in Paris."], 
    # retrieved_contexts=["Paris is the capital of France.","The Eiffel Tower is one of the most famous landmarks in Paris."]
    
)

# Initialize the context recall scorer with the wrapped LLM
context_recall= LLMContextRecall(llm=evaluator_llm)



# Define the main asynchronous function to calculate the Recall score
async def main():
    # Calculate the faithfulness score for the given sample
    context_recall_score = await context_recall.single_turn_ascore(sample)
    # Print the calculated recall score
    print(f"Context Recall : {context_recall_score}")

# Run the main asynchronous function
asyncio.run(main())

