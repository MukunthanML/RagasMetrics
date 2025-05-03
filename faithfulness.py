import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

llm = ChatGroq( model="deepseek-r1-distill-llama-70b", temperature=0.0,max_retries=2)
evaluator_llm = LangchainLLMWrapper(llm)

sample = SingleTurnSample(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)

scorer = Faithfulness(llm=evaluator_llm)

async def main():
    score = await scorer.single_turn_ascore(sample)
    print(score)

asyncio.run(main())