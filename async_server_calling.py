import time
import asyncio
import logging

from openai import AsyncOpenAI


MODEL_PATH = '/home/ai_architect/models/qwen-32b-gptq'
MODEL_BASE_URL = 'http://localhost:8000/v1'


logging.basicConfig(
    level=logging.WARNING,  # Standaard drempelwaarde voor de gehele applicatie
    format='[%(asctime)s %(filename)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %z'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_client():
    """Get the AsyncOpenaAI client which will talk to our locally deployed model."""
    client = AsyncOpenAI(base_url=MODEL_BASE_URL, api_key="token-vllm")
    logger.info('Got client OK')
    return client


async def query_vllm_server(client, system_prompt, user_prompt):
    """Async routing to query the deployed model. Returns the number of generated tokens."""
    response = await client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response.usage.completion_tokens
    

async def main():
    """
    Main routine, does the following:
    1. Get SDK client and define system and user prompt
    2. Collect 50 coroutines that will send the prompts to our model
    3. Run async and wait for all to finish
    4. Print total average of generated tokens per second of runtime
    """
    client = get_client()
    system_prompt = 'You are a helpful AI-assistent.'
    user_prompt = 'Tell me what you know about Albert Einstein in {} words.'

    async_tasks = list()
    for i in range(50):
        task = query_vllm_server(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt.format((i+1)*100))
        async_tasks.append(task)

    logger.info(f'Collected {len(async_tasks)} async tasks.')

    t0 = time.time()
    counts_list = await asyncio.gather(*async_tasks)
    t1 = time.time()

    logger.info(f'Generated {sum(counts_list) / (t1 - t0) :.0f} tokens per second.')


if __name__ == '__main__':
    asyncio.run(main())
    exit(0)
