import logging
import os
import requests

from aiolimiter import AsyncLimiter
# from openai.error import APIError, RateLimitError, Timeout, TryAgain
from litellm.exceptions import APIError # compatibility with other parts of the code
from alpha_codium.settings.config_loader import get_settings
from alpha_codium.log import get_logger

logger = get_logger(__name__)

endpoint = get_settings().config.ollama_endpoint


class AiHandler:
    """
    This class handles interactions with the OpenAI API for chat completions.
    It initializes the API key and other settings from a configuration file,
    and provides a method for performing chat completions using the OpenAI ChatCompletion API.
    """

    def __init__(self):
        """
        Loads model into memory.
        Raises a ValueError if it is unavailable.
        """
        self.limiter = AsyncLimiter(get_settings().config.max_requests_per_minute)
        try:
            model = get_settings().get("config.model")
            resp = requests.post(
                    url=endpoint + "/generate",
                    json = {
                        "model": model
                    }
            )
            resp.raise_for_status()
            logger.debug("Model loaded")
        except AttributeError as e:
            raise ValueError("Ollama model is not available") from e

    @property
    def deployment_id(self):
        """
        Returns the deployment ID. Doesn't really apply for Ollama.
        """
        return None

    async def chat_completion(
            self, model: str,
            system: str,
            user: str,
            temperature: float = 0.2,
            frequency_penalty: float = 0.0,
    ):
        try:
            deployment_id = self.deployment_id
            if get_settings().config.verbosity_level >= 2:
                logging.debug(
                    f"Generating completion with {model}"
                    f"{(' from deployment ' + deployment_id) if deployment_id else ''}"
                )

            async with self.limiter:
                logger.info("-----------------")
                logger.info("Running inference ...")
                logger.debug(f"system:\n{system}")
                logger.debug(f"user:\n{user}")
                response_obj = requests.post(
                    url=endpoint + "/generate",
                    json = {
                        "model": model,
                        "prompt": user,
                        "system": system,
                        "stream": False,
                        "options": {
                            "num_predict": 2000,
                            "stop": ["<|EOT|>"],
                            "repeat_penalty": frequency_penalty+1,
                            "temperature": temperature
                            }
                        }
                    )
                response = response_obj.json()
                response["response"] = response["response"].rstrip()
                if response["response"].endswith("<|EOT|>"):
                    response["response"] = response["response"][:-7]
        except Exception as e:
            logging.error("Error during Ollama inference: ", e)
            raise APIError from e
        if response is None:
            raise APIError
        resp = response["response"]
        finish_reason = 'stop'
        available_resp_size = 2000
        if response["prompt_eval_count"]: # this gave an error sometimes
            available_resp_size -= response["prompt_eval_count"]
        else:
            available_resp_size -= 100 # inaccurate
        if response["eval_count"] == available_resp_size: # cut short (unless the token length lienes up perfectly somehow)
            finish_reason = None
        logger.debug(f"response:\n{resp}")
        logger.info('done')
        logger.info("-----------------")
        return resp, finish_reason
