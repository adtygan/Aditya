import argparse
import gc
import os
from io import StringIO
from pathlib import Path
from typing import List

import langchain
import streamlit as st
from streamlit_chat import message
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from streamlit import chat_message

from llmsearch.config import Config
from llmsearch.chroma import VectorStoreChroma
from llmsearch.process import get_and_parse_response
from llmsearch.utils import get_llm_bundle, set_cache_folder
from llmsearch.embeddings import (
    update_embeddings,
    create_embeddings,
    EmbeddingsHashNotExistError,
)

st.set_page_config(page_title="Aditya", page_icon=":robot:", layout="wide")

load_dotenv()

langchain.debug = True


@st.cache_data
def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Web application for LLMSearch")
    parser.add_argument("--doc_config_path", dest="cli_doc_config_path", type=str, default="")
    parser.add_argument("--model_config_path", dest="cli_model_config_path", type=str, default="")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)
    return args


def hash_func(obj: Config) -> str:
    return str(obj.embeddings.embeddings_path)


def generate_index(config: Config):
    with st.spinner("Creading index, please wait..."):
        logger.debug("Unloading existing models...")
        unload_model()
        set_cache_folder(str(config.cache_folder))

        vs = VectorStoreChroma(
            persist_folder=str(config.embeddings.embeddings_path), config=config
        )
        create_embeddings(config, vs)
        logger.debug("Cleaning memory and re-Loading model...")
        vs.unload()
        vs = None
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    st.success("Done generating index.")


def udpate_index(doc_config_path: str, model_config_file):
    """Updates index on-fly

    Args:
        config_file (str): _description_
    """

    with st.spinner("Updating index, please wait..."):
        logger.debug("Unloading model...")
        unload_model()
        config = load_config(doc_config_path, model_config_file)
        set_cache_folder(str(config.cache_folder))

        vs = VectorStoreChroma(
            persist_folder=str(config.embeddings.embeddings_path), config=config
        )
        try:
            logger.debug("Updating embeddings")
            stats = update_embeddings(config, vs)
        except EmbeddingsHashNotExistError:
            st.error(
                "Couldn't find hash files. Please re-create the index using current version of the app."
            )
        else:
            logger.info(stats)
        finally:
            logger.debug("Cleaning memory and re-Loading model...")

            vs.unload()

            vs = None

            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            reload_model(doc_config_path=doc_config_path, model_config_file=model_config_file)
    st.success("Done updating.")


@st.cache_data
def load_config(doc_config, model_config) -> Config:
    """Loads doc and model configurations, combines, and returns an instance of Config"""

    doc_config_dict = load_yaml_file(doc_config)
    model_config_dict = load_yaml_file(model_config)

    config_dict = {**doc_config_dict, **model_config_dict}
    return Config(**config_dict)

@st.cache_data
def load_yaml_file(config) -> dict:
    if isinstance(config, str):
        logger.info(f"Loading doc config from a file: {config}")
        with open(config, "r") as f:
            string_data = f.read()
    else:
        stringio = StringIO(config.getvalue().decode("utf-8"))
        string_data = stringio.read()

    config_dict = yaml.safe_load(string_data)
    return config_dict


def unload_model():
    """Unloads llm_bundle from the state to free up the GPU memory"""

    if st.session_state["llm_bundle"] is not None:
        st.session_state["llm_bundle"].store = None
        st.session_state["llm_bundle"].chain = None
        st.session_state["llm_bundle"].reranker = None
        st.session_state["llm_bundle"].hyde_chain = None
        st.session_state["llm_bundle"].multiquery_chain = None

    st.cache_data.clear()
    st.cache_resource.clear()
    gc.collect()

    st.session_state["llm_bundle"] = None
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()


@st.cache_data
def generate_response(
    question: str,
    use_hyde: bool,
    use_multiquery,
    _config: Config,
    _bundle,
    label_filter: str = "",
):
    # _config and _bundle are under scored so paratemeters aren't hashed

    output = get_and_parse_response(
        query=question, config=_config, llm_bundle=_bundle, label=label_filter
    )
    return output


@st.cache_data
def get_config_paths(config_dir: str) -> List[str]:
    root = Path(config_dir)
    config_paths = sorted([str(p) for p in root.glob("*.yaml")])
    return config_paths


def reload_model(doc_config_path: str, model_config_file: str):
    if st.session_state["disable_load"]:
        logger.info("In process of loading the model, please wait...")
        return
    st.session_state["disable_load"] = True
    # This is required for memory management, we need to try and unload the model before loading new one

    logger.info("Clearing state and re-loading model...")
    unload_model()

    logger.debug(f"Reload model got DOC CONFIG FILE NAME: {doc_config_path}")
    logger.debug(f"Reload model got MODEL CONFIG FILE NAME: {model_config_file}")
    with st.spinner("Loading configuration"):
        config = load_config(doc_config_path, model_config_file)
        if config.check_embeddings_exist():
            st.session_state["llm_bundle"] = get_llm_bundle(config)
            st.session_state["llm_config"] = {"config": config, "doc_config_path": doc_config_path, "model_config_file": model_config_file}
        else:
            st.error(
                "Couldn't find embeddings in {}. Please generate first.".format(
                    config.embeddings.embeddings_path
                )
            )
            update_embeddings_button = st.button(
                "Generate", on_click=generate_index, args=(config,), type="secondary"
            )

    st.session_state["disable_load"] = False


st.title("Hello, I'm Aditya. Nice to meet you! :wave:")
st.text('••/)_/)\n=(^.^)=\n () ( ")••')
args = parse_cli_arguments()
st.sidebar.subheader(":hammer_and_wrench: Configuration")

# Initialsize state for historical resutls
if "llm_bundle" not in st.session_state:
    st.session_state["llm_bundle"] = None

if "llm_config" not in st.session_state:
    st.session_state["llm_config"] = None

# Initialsize state for historical resutls
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "disable_load" not in st.session_state:
    st.session_state["disable_load"] = False

if Path(args.cli_doc_config_path).is_dir():
    config_paths = get_config_paths(args.cli_doc_config_path)
    doc_config_path = 'src/llmsearch/config.yaml'
    model_config_file = args.cli_model_config_path
    logger.debug(f"CONFIG FILE: {doc_config_path}")

if st.session_state["llm_bundle"] is None:
    reload_model(doc_config_path, model_config_file)


if st.session_state["llm_bundle"] is not None:
    config = st.session_state["llm_config"]["config"]

    doc_config_path = st.session_state["llm_config"]["doc_config_path"]
    model_config_file = st.session_state["llm_config"]["model_config_file"]
    config_file_name = doc_config_path if isinstance(doc_config_path, str) else doc_config_path.name

    st.sidebar.write(f"**Model type:** GPT-3.5-Turbo")

    st.sidebar.write(
        f"**Prompt**: I want you to act as a person called Aditya. I will provide you with an individual looking to know about Aditya, and your task is to introduce Aditya and answer questions about Aditya. Always use first person to answer questions. Context information is provided below. Given only the context and not prior knowledge, provide concise answer to the question. If context does not provide enough details, answer it without hallucinating."
    )
    st.sidebar.write(
        f"**Max char size (semantic search):** {config.semantic_search.max_char_size}"
    )
    label_filter = ""

    text = st.chat_input("Enter text", disabled=False)
    is_hyde = st.sidebar.checkbox(
        label="Use HyDE (costs 2 API calls)",
        value=st.session_state["llm_bundle"].hyde_enabled,
    )
    is_multiquery = st.sidebar.checkbox(
        label="Use MultiQuery (costs 2 API calls)",
        value=st.session_state["llm_bundle"].multiquery_enabled,
    )

    if text:
        # Dynamically switch hyde
        st.session_state["llm_bundle"].hyde_enabled = is_hyde
        st.session_state["llm_bundle"].multiquery_enabled = is_multiquery
        output = generate_response(
            question=text,
            use_hyde=st.session_state["llm_bundle"].hyde_enabled,
            use_multiquery=st.session_state["llm_bundle"].multiquery_enabled,
            _bundle=st.session_state["llm_bundle"],
            _config=config,
            label_filter=label_filter,
        )

        # Add assistant response to chat history
        st.session_state["messages"].append(
            {
                "question": text,
                "response": output.response,
            }
        )
        for h_response in st.session_state["messages"]:
            message(h_response["question"], is_user=True)
            message(h_response["response"])


else:
    st.info("Load the AI to start...")
