import json

from nonebot import logger, on_command
from nonebot.matcher import Matcher
from nonebot.params import ArgPlainText, CommandArg
from nonebot.adapters import Message, Event
from nonebot.adapters.onebot.v11 import PrivateMessageEvent,MessageSegment
from nonebot.rule import Rule, to_me
from nonebot.permission import SUPERUSER
from .common import plugin_config, is_enable, http_invoke

async def private_check(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent) and is_enable()


weather = on_command(
    "天气", rule=to_me(), priority=plugin_config.assistant_command_priority)

settings = on_command('设置', rule=private_check, priority=1, permission=SUPERUSER)

logger.info(f'插件已加载{plugin_config.model_dump_json()}')


@settings.handle()
async def handle_assistant(matcher: Matcher, settings_msg: Message = CommandArg()):
    matcher.stop_propagation()
    if settings := settings_msg.extract_plain_text():
        keys=['llm_provider','base_url','api_key','chat_model','image_model','code_model','embeddings_model']
        values = [plugin_config.assistant_llm_provider, plugin_config.assistant_llm_base_url,
                  plugin_config.assistant_openai_api_key,
                  plugin_config.assistant_chat_model, plugin_config.assistant_image_model,
                  plugin_config.assistant_code_model, plugin_config.assistant_embeddings_model]
        from argparse import ArgumentParser
        parser = ArgumentParser()
        splited_settings=settings.split()
        for i,key in enumerate(keys):
            parser.add_argument(key, type=str,default=values[i])
        for i,value in enumerate(splited_settings):
            if '='  in value:
                key,value=value.split('=')
                values[keys.index(key)]=value
            else:
                values[i]=value
        args = parser.parse_args(values)
        logger.info(f"handle msg {splited_settings} config: {values}")
        for key in keys:
            matcher.set_arg(key, Message([MessageSegment.text(getattr(args,key))]))


@settings.got("llm_provider", prompt="请输入提供商")
@settings.got("base_url", prompt="请输入模型地址")
@settings.got("chat_model", prompt="请输入聊天模型")
@settings.got("image_model", prompt="请输入视觉模型")
async def got_asistant_config(matcher: Matcher):
    llm_provider = matcher.get_arg("llm_provider").extract_plain_text()
    base_url = matcher.get_arg("base_url").extract_plain_text()
    chat_model = matcher.get_arg("chat_model").extract_plain_text()
    image_model = matcher.get_arg("image_model").extract_plain_text()
    api_key = matcher.get_arg("api_key").extract_plain_text() if  matcher.get_arg("api_key") else ''
    code_model = matcher.get_arg("code_model").extract_plain_text() if matcher.get_arg("code_model") else chat_model
    embeddings_model = matcher.get_arg("embeddings_model").extract_plain_text() if matcher.get_arg("embeddings_model") else chat_model
    logger.info(f"设置 config: {matcher.state}")
    plugin_config.assistant_chat_model = chat_model
    plugin_config.assistant_image_model = image_model
    plugin_config.assistant_code_model = code_model
    plugin_config.assistant_embeddings_model = embeddings_model
    plugin_config.assistant_llm_provider = llm_provider
    plugin_config.assistant_llm_base_url = base_url
    plugin_config.assistant_openai_api_key = api_key
    plugin_config.save_to_file()
    await settings.finish('设置成功!')


@weather.handle()
async def handle_weather(matcher: Matcher, args: Message = CommandArg()):
    matcher.stop_propagation()
    if args.extract_plain_text():
        matcher.set_arg("location", args)


@weather.got("location", prompt="请输入地名")
async def got_weather(location: str = ArgPlainText()):
    await weather.send("正在查询中...")
    data = await http_invoke('https://geoapi.qweather.com/v2/city/lookup', headers={'Content-Type': 'application/json'},
                             params={'key': plugin_config.assistant_plugin_weather_api_key, 'location': location}, method='GET')
    if data and any(key == 'location' for key in data.keys()):
        weather_location = await http_invoke('https://devapi.qweather.com/v7/weather/now', headers={'Content-Type': 'application/json'},
                                             params={'key': plugin_config.assistant_plugin_weather_api_key, 'location': data["location"][0]["id"]}, method='GET')
        resp = await plugin_config.llm.chat([{'role': 'system', 'content': f'根据以下json内容,总结城市或地区"{location}"的天气\n:{weather_location['now']}'}])
        return await weather.finish(resp['message']['content'])
    else:
        return await weather.finish("未找到该地名。")
