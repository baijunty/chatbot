import nonebot
from nonebot.adapters.onebot.v11.adapter import Adapter


if __name__ == '__main__':
    # 初始化时
    nonebot.init()
    driver = nonebot.get_driver()
    driver.register_adapter(Adapter)
    nonebot.load_plugins('plugins')
    nonebot.run()
