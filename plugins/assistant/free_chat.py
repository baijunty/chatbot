import hashlib
import json
import jieba
from nonebot.rule import Rule
from nonebot import on_message
from nonebot.matcher import Matcher
from nonebot.adapters.onebot.v11 import (
    Bot, GroupMessageEvent, PrivateMessageEvent, Event, Message, MessageSegment)
from .common import *
from chromadb import Client, Collection
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from .html_parser import HtmlParser
from uuid import uuid4
from . import logger
URL_REGEX = r'https?:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}\/?[\w\-@%&=\?\.~:\/?#]*'


async def custom_rule(bot: Bot, event: Event) -> bool:
    user_id = event.get_user_id()
    logger.debug('INFO', f"User {user_id} is checking the {event.get_message()}.")
    if event is PrivateMessageEvent:
        logger.debug('INFO', f"私聊消息：{user_id} in {user_id in bot.config.superusers} or {
               f"{bot.adapter.get_name().split(maxsplit=1)[0].lower()}:{user_id}" in bot.config.superusers}")
        return f"{bot.adapter.get_name().split(maxsplit=1)[0].lower()}:{user_id}" in bot.config.superusers or user_id in bot.config.superusers
    return event.is_tome() or any(msg.type == 'at' and msg.data['qq'] == bot.self_id for msg in event.get_message())

free_chat = on_message(rule=Rule(custom_rule, is_enable),
                       priority=plugin_config.assistant_command_priority**2)
client = Client()
embeding = OllamaEmbeddingFunction(f'{plugin_config.assistant_llm_base_url}/api/embeddings',
                                   plugin_config.assistant_embeddings_model) if plugin_config.assistant_llm_provider == 'ollama' else DefaultEmbeddingFunction()


def create_or_recreate_collection(collection_name: str) -> Collection:
    collection = client.get_or_create_collection(
        collection_name, embedding_function=embeding)
    if collection.count() > 0:
        client.delete_collection(collection_name)
    return client.get_or_create_collection(collection_name, embedding_function=embeding)


async def merge_msg(bot: Bot, matcher: Matcher, message: Message,collection: Collection):
    last = matcher.get_receive('summary_last_receive', default=None)
    images = []
    reference = ''
    request = ''
    urls = []
    if last is not None:
        last_content = await get_content_from_msg(bot, last.original_message,collection)
        images.extend(last_content[0])
        reference += last_content[1]+'\n'
        request += last_content[2]+'\n'
        urls.extend(last_content[3])
    request_msg = await get_content_from_msg(bot, message,collection)
    images.extend(request_msg[0])
    reference += request_msg[1]+'\n'
    request += request_msg[2]+'\n'
    urls.extend(request_msg[3])
    return images, reference, request, urls


@free_chat.handle()
async def handle_group_message(bot: Bot, matcher: Matcher, event: GroupMessageEvent):
    args = event.original_message
    key = f'{event.get_session_id()}_history'
    state = matcher.state
    history = state.get(key, [])
    collection = state.get('chat_embeddings', None)
    if not collection:
        collection = create_or_recreate_collection(
            f"chat_history_{event.get_session_id()}")
        state['chat_embeddings'] = collection
    reply = [MessageSegment.at(event.get_user_id()),
             MessageSegment.reply(id_=event.message_id)]
    if len(args.extract_plain_text().strip()) > 0:
        if args.extract_plain_text().strip() == 'gg':
            await matcher.finish([* reply, '已退出聊天'])
            return
        await matcher.send([* reply, '正在思考回答'])
        images, reference, request, urls = await merge_msg(bot, matcher, args,collection)
        resp = await analysis_message_to_chat(history, images, reference, request, urls, collection=collection)
        reply.append(MessageSegment.text(resp))
        state[key] = history
        state['chat_embeddings_client'] = client
        await free_chat.reject(reply)
    else:
        matcher.set_receive("summary_last_receive", event)
        reply.append(MessageSegment.text('请输入问题'))
        await free_chat.reject(reply)


async def classify_macher(query: str, categories: list):
    import time
    message = [* CLASSFY_CONTENT, {
        "role": "user",
        "content": json.dumps({"input_text": query, "categories": [{'category_id': time.time(), "category_name": category} for category in categories]}, ensure_ascii=False),
        "images": []
    }]
    data = await chat_to_llm(plugin_config.assistant_chat_model, message=message)
    content = data['message']['content'].strip()
    if content.startswith('```json') and content.endswith('```'):
        content = content[content.index('{'):content.index('}')+1].strip()
    return content


@free_chat.handle()
async def handle_private_message(bot: Bot, matcher: Matcher, event: PrivateMessageEvent):
    args = event.get_message()
    key = f'{event.get_session_id()}_history'
    state = matcher.state
    history = state.get(key, [])
    collection = state.get('chat_embeddings', None)
    logger.debug('INFO', f'{args} length: {len(history)}')
    if not collection:
        collection = create_or_recreate_collection(
            f"chat_history_{event.get_session_id()}")
        state['chat_embeddings'] = collection
    if args.extract_plain_text():
        if args.extract_plain_text().strip() == 'gg':
            await matcher.finish('已退出聊天')
            return
        await matcher.send('正在思考回答')
        images, reference, request, urls = await merge_msg(bot, matcher, args,collection)
        resp = await analysis_message_to_chat(history, images, reference, request, urls, collection=collection)
        # logger.info(f"history: {[his['content'] for his in history]}")
        state[key] = history
        state['chat_embeddings_client'] = client
        await free_chat.reject(resp)
    else:
        matcher.set_receive("summary_last_receive", event)
        await matcher.reject("请输入内容")


async def get_file_content(msg_data: dict,collection: Collection):
    images = []
    reference = msg_data['file']
    file_size = int(msg_data['file_size'],10)
    resp = await http_invoke(f'http://192.168.1.107:6000/get_file', data=json.dumps({'file_id': msg_data['file_id']})) if file_size> 0 and file_size < 1024 * 1024*10 else ''
    import base64
    if resp and os.path.splitext(msg_data['file'])[-1] in ['.pdf', '.doc', '.docx', '.txt', '.json', '.csv']:
        b64 = resp['data']['base64']
        try:
            reference = base64.b64decode(b64).decode('utf-8')
        except UnicodeDecodeError as e:
            reference = base64.b64decode(b64).decode('gb2312')
    elif resp and os.path.splitext(msg_data['file'])[-1] in ['.jpg', '.png', '.jpeg', '.gif']:
        b64 = resp['data']['base64']
        return images.append(b64)
    if len(reference) > 8096:
        lines=reference.split('\n')
        split_to_embeding_text(lines,collection)
        return images, msg_data['file']
    return images, reference


async def get_content_from_msg(bot: Bot, message: Message,collection: Collection):
    images = []
    reference = ''
    request = ''
    urls = []
    import re
    for msg in message:
        logger.debug('INFO', f"msg:{msg}")
        msg_type = msg.type if isinstance(msg, MessageSegment) else msg['type']
        msg_data = msg.data if isinstance(msg, MessageSegment) else msg['data']
        if msg_type == "image":
            data = await http_invoke(msg_data['url'], method='GET')
            if data is None:
                continue
            import base64
            images.append(base64.b64encode(data).decode('utf-8'))
        elif msg_type == "reply" and msg_data['id'] not in [0, '', '0']:
            reply_msg = await bot.get_msg(message_id=msg_data['id'])
            reply = await get_content_from_msg(bot,  reply_msg['message'],collection)
            images.extend(reply[0])
            urls.extend(reply[3])
            reference += f'{reply[1]}\n{reply[2]}\n'
        elif msg_type == "forward" and msg_data['id'] not in [0, '', '0']:
            forward_msg = await bot.get_forward_msg(id=msg_data['id'])
            logger.debug('INFO', f"forward_msg:{forward_msg}")
            for sub_msg in forward_msg['messages']:
                forward = await get_content_from_msg(bot, sub_msg['message'],collection)
                images.extend(forward[0])
                urls.extend(forward[3])
                reference += f'{forward[1]}\n{forward[2]}\n'
        elif msg_type == "node":
            content = await get_content_from_msg(bot, msg_data['content'],collection)
            images.extend(content[0])
            urls.extend(content[3])
            reference += f'{msg_data['nickname']}:{content[1]}{content[2]}\n'
        elif msg_type == "file":
            content = await get_file_content(msg_data,collection)
            images.extend(content[0])
            reference += content[1]
        elif msg_type == 'text' and len(msg_data['text'].strip()) > 0:
            text = msg_data['text']
            if text.startswith('http') and (regex := re.findall(URL_REGEX, text)):
                urls.extend([str(url) for url in regex])
                reference += re.sub(URL_REGEX, '', text)
            else:
                request += f'{text}'
    return images, reference, request, urls


async def get_html_body(url: str):
    try:
        data = await http_invoke(url, method='GET')
        if not data or not isinstance(data, str):
            return ''
        html_parser = HtmlParser()
        html_parser.feed(data)
        return html_parser.text
    except Exception:
        return ''


def split_to_embeding_text(lines, collection: Collection,url=None):
    count = 0
    start = 0
    for i, line in enumerate(lines):
        count += len(line)
        if count > 512:
            count = 0
            collection.add(ids=str(uuid4().hex), documents=['\n'.join(lines[start:i])], metadatas={'url': url} if url else None)
            start = i
    text = lines[start:]
    if text:
        collection.add(ids=str(uuid4().hex), documents=['\n'.join(text)], metadatas={'url': url} if url else None)


async def search_from_net(query: str, collection: Collection):
    data = await http_invoke('https://baijunty.com/search', params={
        'q': query, 'format': 'json', 'language': 'all', 'image_proxy': 0, 'time_range': '', 'safesearch': 0, 'categories': 'general'}, method='GET')
    if not data:
        logger.error('ERROR', f'failed to get search {query}')
        return []
    results = [r for r in data['results'][:6] if r['content']]
    logger.debug('INFO', f'search {query} found {results}')
    await get_similar_from_urls(results, collection)


async def get_similar_from_urls(results: list, collection: Collection):
    for result in results:
        text = result['content']
        url = result['url']
        logger.debug('INFO', f'get similar from {url} content {text}')
        collection.add(ids=hashlib.md5(text.encode()).hexdigest(), documents=[text], metadatas={'url': url})
        html = await get_html_body(url)
        if html:
            split_to_embeding_text(html, collection,url)


def contains_image(history: list, images: list):
    return any([msg.get('images') for msg in history]) or len(images) > 0


async def analysis_message_to_chat(history: list, images, reference, request, urls, collection: Collection):
    if len(urls) > 0:
        for url in urls:
            html = await get_html_body(url)
            if html:
                split_to_embeding_text(html, collection,url)
    contains_image_flag = contains_image(history, images)
    reference = reference.strip()
    logger.debug('INFO', f"reference: {reference} request {request} urls {urls} images {len(images)}")
    if not plugin_config.assistant_chat_model:
        return f'尚未设置聊天模型'
    search = await classify_macher(query=request, categories=['需要联网搜索', '其他']) if len(urls) == 0 and not contains_image_flag and len(reference) == 0 else ''
    result_urls = set()
    if '需要联网搜索' in search:
        if not plugin_config.assistant_embeddings_model:
            return '尚未设置向量嵌入模型'
        search_words=request
        if (len(history) > 0 or len(reference) > 0) and not contains_image_flag:
            data = await chat_to_llm(plugin_config.assistant_chat_model, [{'role': 'system', 'content': RE_GENERATE_TITLE.format('\n'.join([f'{msg['role']}:{msg['content']}' for msg in history[-4:] if msg['role']=='user']), request)}, {'role': 'user', 'content': '重新表述的问题:'}])
            search_words = data['message']['content']
        await search_from_net(search_words, collection)
    results = collection.query(query_texts=request, n_results=10)
    result_reference = ''
    if results and len(results['documents']) > 0:
        metadatas = results['metadatas']
        for distances, document in zip(results['distances'], results['documents']):
            result_reference += ' '.join([text for distance, text in zip(distances, document) if distance > 0.5]) + '\n'
        if results['metadatas']:
            for metadatas in results['metadatas']:
                result_urls.update([data['url'] for data in metadatas if data and data.get('url')])
    if len(result_reference.strip()) > 0:
        import datetime
        history.append({'role': 'system', 'content': SEARCH_SYSTEM.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), result_reference)})
        history.append({'role': 'user', 'content': request,"images": images if contains_image_flag else None})
    else:
        history.append({'role': 'user', 'content': f'{reference}\n{request}', "images": images if contains_image_flag else None})
    logger.debug('INFO', f"reference: {reference} request {request} urls {urls} images {len(images)}")
    model = plugin_config.assistant_image_model if contains_image_flag else plugin_config.assistant_chat_model
    if not model:
        return f'尚未设置{"聊天" if contains_image_flag else "图像"}模型'
    data = await chat_to_llm(model, history)
    if data:
        history.append(data['message'])
        return f'{data['message']['content']}\n{f'引用:\n{'\n'.join(result_urls)}' if len(result_urls) > 0 else ""}'
    else:
        return '模型调用失败'


async def chat_to_llm(model: str, message: list):
    resp = await plugin_config.llm.chat(message=message, model=model)
    logger.debug('INFO', f'resp {resp}')
    return resp
