from aiohttp import ClientSession
from .config import Config
from . import logger
import os
if os.path.exists("config.json"):
    with open("config.json", "r",encoding="utf-8") as f:
        plugin_config = Config.model_validate_json(f.read())
else:
    plugin_config = Config()
CLASSFY_CONTENT=[
    {
        "role": "system",
        "content": "\n    ### Job Description',\n    You are a text classification engine that analyzes text data and assigns categories based on user input or automatically determined categories.\n    ### Task\n    Your task is to assign one categories ONLY to the input text and only one category may be assigned returned in the output. Additionally, you need to extract the key words from the text that are related to the classification.\n    ### Format\n    The input text is in the variable input_text. Categories are specified as a category list with two filed category_id and category_name in the variable categories. Classification instructions may be included to improve the classification accuracy.\n    ### Constraint\n    DO NOT include anything other than the JSON array in your response.\n    ### Memory\n    Here is the chat histories between human and assistant, inside <histories></histories> XML tags.\n    <histories>\n    \n    </histories>\n",
        "images": []
    },
    {
        "role": "user",
        "content": "\n    { \"input_text\": [\"I recently had a great experience with your company. The service was prompt and the staff was very friendly.\"],\n    \"categories\": [{\"category_id\":\"f5660049-284f-41a7-b301-fd24176a711c\",\"category_name\":\"Customer Service\"},{\"category_id\":\"8d007d06-f2c9-4be5-8ff6-cd4381c13c60\",\"category_name\":\"Satisfaction\"},{\"category_id\":\"5fbbbb18-9843-466d-9b8e-b9bfbb9482c8\",\"category_name\":\"Sales\"},{\"category_id\":\"23623c75-7184-4a2e-8226-466c2e4631e4\",\"category_name\":\"Product\"}],\n    \"classification_instructions\": [\"classify the text based on the feedback provided by customer\"]}\n",
        "images": []
    },
    {
        "role": "assistant",
        "content": "\n```json\n    {\"keywords\": [\"recently\", \"great experience\", \"company\", \"service\", \"prompt\", \"staff\", \"friendly\"],\n    \"category_id\": \"f5660049-284f-41a7-b301-fd24176a711c\",\n    \"category_name\": \"Customer Service\"}\n```\n",
        "images": []
    },
    {
        "role": "user",
        "content": "\n    {\"input_text\": [\"bad service, slow to bring the food\"],\n    \"categories\": [{\"category_id\":\"80fb86a0-4454-4bf5-924c-f253fdd83c02\",\"category_name\":\"Food Quality\"},{\"category_id\":\"f6ff5bc3-aca0-4e4a-8627-e760d0aca78f\",\"category_name\":\"Experience\"},{\"category_id\":\"cc771f63-74e7-4c61-882e-3eda9d8ba5d7\",\"category_name\":\"Price\"}],\n    \"classification_instructions\": []}\n",
        "images": []
    },
    {
        "role": "assistant",
        "content": "\n```json\n    {\"keywords\": [\"bad service\", \"slow\", \"food\", \"tip\", \"terrible\", \"waitresses\"],\n    \"category_id\": \"f6ff5bc3-aca0-4e4a-8627-e760d0aca78f\",\n    \"category_name\": \"Experience\"}\n```\n",
        "images": []
    }
]

GENERATE_TITLE=""""
Create a concise, 10 word title title for the chat history, in the given language. RESPOND ONLY WITH THE TITLE TEXT.
Examples of titles:
Stock Market Trends
Perfect Chocolate Chip Recipe
Evolution of Music Streaming
Remote Work Productivity Tips
Artificial Intelligence in Healthcare
Video Game Development Insights"""

SEARCH_SYSTEM ="""现在的日期是{},下面是互联网搜索结果的内容，然后根据用户的查询生成回答。{}"""

RE_GENERATE_TITLE = """您将提出一个后续问题。如果需要，您需要重新表述后续问题，以便它是一个独立问题，不超过30字,可供 AI 模型用来搜索互联网。

示例：

后续问题：心脏病发作的症状是什么？

重新表述的问题：心脏病发作的症状。

后续问题：即将举行的奥运会在哪里举行？

重新表述的问题：即将举行的奥运会的地点。

后续问题：泰勒·斯威夫特的最新专辑？

重新表述的问题：泰勒·斯威夫特最新专辑的名称。

上一个对话：
{}
后续问题：{}
"""

def is_enable() -> bool:
    """
    Check if the AI assistant plugin is enabled.
    
    Returns:
        bool: True if the plugin is enabled, False otherwise.
    """
    return plugin_config.assistant_plugin_enabled

async def http_invoke(url, params=None, headers=None, data=None, method='POST'):
    default_headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8'
    }
    if headers:
        default_headers.update(headers)
    async with ClientSession() as session:
        async with session.request(method, url, headers=default_headers, data=data, params=params, proxy="http://127.0.0.1:8389") as response:
            if not response.ok:
                logger.error('ERROR',f'{url} response {await response.text()}')
                return None
            content_type = response.headers['Content-Type']
            if content_type.startswith('application/json'):
                return await response.json()
            elif content_type.startswith('text/'):
                return await response.text()
            return await response.content.read()