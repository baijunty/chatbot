from nonebot.plugin import PluginMetadata
from .config import Config
from nonebot import logger
logger=logger
from . import free_chat,assistant
__plugin_meta__ = PluginMetadata(
    name="assistant",
    description="AI助手",
    usage="",
    type="application",
    homepage="",
    config=Config,
    supported_adapters=None,
)
__ALL__ = ["is_enable", "http_invoke", "plugin_config","logger"]
