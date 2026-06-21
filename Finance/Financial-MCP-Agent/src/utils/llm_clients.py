import os
import time
import backoff
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# 设置日志记录
logger = setup_logger('llm_clients')


class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def get_completion(self, messages, **kwargs):
        """获取模型回答"""
        pass


class OpenAICompatibleClient(LLMClient):
    """OpenAI 兼容 API 客户端 — 支持多 API 配置文件（通过 env_prefix 切换）"""

    def __init__(self, api_key=None, base_url=None, model=None,
                 env_prefix="OPENAI_COMPATIBLE", extra_body=None,
                 http_timeout=60, http_connect_timeout=10):
        """
        Args:
            api_key: API key（None 则从环境变量 {env_prefix}_API_KEY 读取）
            base_url: API 地址（None 则从环境变量 {env_prefix}_BASE_URL 读取）
            model: 模型名（None 则从环境变量 {env_prefix}_MODEL 读取）
            env_prefix: 环境变量前缀，如 "OPENAI_COMPATIBLE" 或 "QWEN"
            extra_body: 额外请求体（如 thinking 模式控制）
            http_timeout: HTTP 请求超时（秒）
            http_connect_timeout: HTTP 连接超时（秒）
        """
        if env_prefix:
            self.api_key = api_key or os.getenv(f"{env_prefix}_API_KEY")
            self.base_url = base_url or os.getenv(f"{env_prefix}_BASE_URL")
            self.model = model or os.getenv(f"{env_prefix}_MODEL")
        else:
            self.api_key = api_key
            self.base_url = base_url
            self.model = model
        self.extra_body = extra_body or {}

        if not self.api_key:
            label = env_prefix or "API"
            logger.error(f"{ERROR_ICON} 未找到 {label} 的 API_KEY")
            raise ValueError(f"{label}_API_KEY not found in environment variables")

        if not self.base_url:
            label = env_prefix or "API"
            logger.error(f"{ERROR_ICON} 未找到 {label} 的 BASE_URL")
            raise ValueError(f"{label}_BASE_URL not found in environment variables")

        if not self.model:
            label = env_prefix or "API"
            logger.error(f"{ERROR_ICON} 未找到 {label} 的 MODEL")
            raise ValueError(f"{label}_MODEL not found in environment variables")

        # 初始化 OpenAI 客户端（带超时防止挂起）
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(http_timeout, connect=http_connect_timeout),
        )
        logger.info(f"{SUCCESS_ICON} OpenAI Compatible 客户端初始化成功 "
                     f"(model={self.model}, base_url={self.base_url[:40]}...)")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=3,
        max_time=120,
        giveup=lambda e: any(kw in str(e).lower() for kw in (
            "401", "403", "invalid_api_key", "incorrect api key",
            "authentication", "unauthorized", "permission"))
    )
    def call_api_with_retry(self, messages, stream=False, extra_body=None):
        """带重试机制的 API 调用函数（401/403 不重试）"""
        try:
            logger.info(f"{WAIT_ICON} 正在调用 OpenAI Compatible API ({self.model})...")
            logger.debug(f"请求内容: {messages}")

            kwargs = dict(
                model=self.model,
                messages=messages,
                stream=stream
            )
            # 合并默认 extra_body 和调用时传入的 extra_body
            body = {**self.extra_body, **(extra_body or {})}
            if body:
                kwargs["extra_body"] = body
                logger.debug(f"extra_body: {body}")

            response = self.client.chat.completions.create(**kwargs)
            logger.info(f"{SUCCESS_ICON} API 调用成功")
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=2, initial_retry_delay=1, **kwargs):
        """获取聊天完成结果，包含重试逻辑"""
        try:
            logger.info(f"{WAIT_ICON} 使用模型: {self.model}")
            logger.debug(f"消息内容: {messages}")

            for attempt in range(max_retries):
                try:
                    response = self.call_api_with_retry(messages)

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    # 处理不同类型的响应
                    content = None

                    if isinstance(response, dict):
                        if 'choices' in response and len(response['choices']) > 0:
                            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                                content = response['choices'][0]['message']['content']
                            elif 'text' in response['choices'][0]:
                                content = response['choices'][0]['text']
                    elif hasattr(response, 'choices') and len(response.choices) > 0:
                        if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                            content = response.choices[0].message.content

                    if content is None:
                        if hasattr(response, 'text'):
                            content = response.text
                        elif hasattr(response, 'content'):
                            content = response.content
                        else:
                            content = str(response)
                            logger.warning(f"{WAIT_ICON} 无法直接提取响应内容，使用字符串化响应")

                    if content:
                        logger.debug(f"API 响应内容: {content[:500]}...")
                        logger.info(f"{SUCCESS_ICON} 成功获取响应")
                        return content
                    else:
                        logger.warning(f"{ERROR_ICON} 无法从响应中提取内容")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        return "无法从响应中提取内容"

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} get_completion 发生错误: {str(e)}")
            return None


class LLMClientFactory:
    """LLM 客户端工厂类（单 API 配置，通过 .env 统一管理）"""

    @staticmethod
    def create_client(client_type="auto", **kwargs):
        """
        创建 LLM 客户端

        Args:
            client_type: "auto" | "openai_compatible"
            **kwargs: 覆盖参数 (api_key, base_url, model, extra_body, http_timeout 等)
        """
        if client_type == "auto":
            client_type = "openai_compatible"
            logger.info(f"{WAIT_ICON} 自动选择 OpenAI Compatible API")

        if client_type == "openai_compatible":
            return OpenAICompatibleClient(
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model"),
                env_prefix="OPENAI_COMPATIBLE",
                extra_body=kwargs.get("extra_body"),
                http_timeout=kwargs.get("http_timeout", 60),
                http_connect_timeout=kwargs.get("http_connect_timeout", 10),
            )

        else:
            raise ValueError(f"不支持的客户端类型: {client_type}")
