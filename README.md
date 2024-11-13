# chatbot

## 介绍
这是一个基于NoneBot2的聊天机器人项目，支持多种消息适配器，并具有自定义规则和基本的LLM功能。

## 特性
- **多适配器支持**：通过`pyproject.toml`配置文件，支持OneBot V12和OneBot V11适配器。
- **自定义规则**：在`free_chat.py`中实现了自定义消息处理规则，包括对超级用户的特殊处理。
- **LLM功能**：在`llm.py`中定义了基本的LLM（大语言模型）类，提供了聊天、生成和嵌入向量的功能。

## 安装
1. 确保你已经安装了Python 3.9或更高版本。
2. 克隆项目到本地：
   ```bash
   git clone https://github.com/baijunty/chatbot
   cd chatbot
   ```
3. 安装依赖：
   ```bash
   pip install -e .
   ```

## 配置
1. 根据需要修改`pyproject.toml`文件中的适配器和插件配置。
2. 在NoneBot的配置文件中添加或修改超级用户列表。

## 使用
启动机器人后，它将根据自定义规则处理消息，并使用LLM功能进行智能回复。

```bash
nb run --relaod
```

## 贡献
欢迎贡献代码！请在提交PR前确保遵循项目的编码规范和测试用例。

## 许可证
MIT License

Copyright (c) [2024] [baijunty]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
