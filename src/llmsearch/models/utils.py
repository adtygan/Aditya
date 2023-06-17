from typing import Union

from llmsearch.models.config import OpenAIModelConfig, LlamaModelConfig, AutoGPTQModelConfig, HuggingFaceModelConfig

from llmsearch.models.llama import LlamaModel
from llmsearch.models.hf import HuggingFaceModel
from llmsearch.models.openai import OpenAIModel
from llmsearch.models.autogptq import AutoGPTQModel

model_mappings = {
        LlamaModelConfig: LlamaModel,
        HuggingFaceModelConfig: HuggingFaceModel,
        OpenAIModelConfig: OpenAIModel,
        AutoGPTQModelConfig: AutoGPTQModel
    }

def get_llm(llm_config: Union[OpenAIModelConfig, LlamaModelConfig, AutoGPTQModelConfig, HuggingFaceModelConfig]):
    model_type = model_mappings.get(type(llm_config), None)  # type: ignore
    if model_type is None:
        raise TypeError(f"Unknown model type: {type(llm_config)}")
        
    llm = model_type(llm_config)  # type: ignore
    return llm
    