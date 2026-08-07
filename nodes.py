import multiprocessing
import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
from glob import glob
from rapidfuzz import fuzz, process
import llama_cpp
from llama_cpp import Llama
from comfy_api.latest import io
from pathlib import Path
import folder_paths

from .utils import *

# Standard GGML enum integer values: 1 = F16, 8 = Q8_0, 2 = Q4_0
KV_CACHE_TYPES = {
    "f16": getattr(getattr(llama_cpp, "llama_cpp", llama_cpp), "GGML_TYPE_F16", 1),
    "q8_0": getattr(getattr(llama_cpp, "llama_cpp", llama_cpp), "GGML_TYPE_Q8_0", 8),
    "q4_0": getattr(getattr(llama_cpp, "llama_cpp", llama_cpp), "GGML_TYPE_Q4_0", 2),
}

class DavchaLLMLoader(io.ComfyNode):
    gguf_folder = os.path.join(folder_paths.models_dir, "LLM", "GGUF")
    
    # Cache pour les métadonnées (pour accélérer define_schema et F5)
    _metadata_cache = {}
    _path_map = {}
        
    @classmethod
    def define_schema(cls):
        path = os.path.join(cls.gguf_folder, "**", "*.gguf")
        files = list(filter(lambda x: "mmproj" not in x, glob(path, recursive=True)))
        
        # Nettoyage du cache si on a supprimé un fichier du dossier
        cls._metadata_cache = {k: v for k, v in cls._metadata_cache.items() if k in files}
        
        options = []
        cls._path_map.clear()
        
        for file in files:
            # On vérifie la date de modification du fichier
            mtime = os.path.getmtime(file)
            
            # On ne lit le GGUF que si c'est un nouveau fichier ou s'il a été modifié
            if file not in cls._metadata_cache or cls._metadata_cache[file]['mtime'] != mtime:
                try:
                    llm_meta = Llama(file, vocab_only=True, verbose=False)
                    arch = llm_meta.metadata.get("general.architecture", "llama")
                    max_n_ctx = int(llm_meta.metadata.get(f"{arch}.context_length", 8192))
                    max_n_gpu_layers = int(llm_meta.metadata.get(f"{arch}.block_count", 99))
                    
                    # Détection si c'est un modèle MoE (ex: DeepSeek, Mixtral, Qwen MoE, etc.)
                    expert_count = int(llm_meta.metadata.get(f"{arch}.expert_count", 0))
                    expert_used_count = int(llm_meta.metadata.get(f"{arch}.expert_used_count", 0))
                    is_moe = expert_count > 0 or expert_used_count > 0 or any(kw in arch.lower() for kw in ["moe", "mixtral", "dbrx"])
                    
                    cls._metadata_cache[file] = {
                        'mtime': mtime,
                        'max_n_ctx': max_n_ctx,
                        'max_n_gpu_layers': max_n_gpu_layers,
                        'is_moe': is_moe,
                        'expert_count': expert_count,
                        'expert_used_count': expert_used_count,
                    }
                except Exception as e:
                    print(f"[DavchaLLM] Erreur lecture métadonnées {file}: {e}")
                    cls._metadata_cache[file] = {'mtime': mtime, 'max_n_ctx': 8192, 'max_n_gpu_layers': 99, 'is_moe': False, 'expert_count': 0, 'expert_used_count': 0}
            
            cached_data = cls._metadata_cache[file]
            
            # Sub-inputs per selected model
            inputs = [
                io.Int.Input("n_ctx", min=512, max=cached_data['max_n_ctx'], default=4096),
                io.Int.Input("n_gpu_layers", min=-1, max=cached_data['max_n_gpu_layers'], default=-1),
            ]
            
            # Dynamically add n_cpu_moe ONLY for MoE models
            if cached_data['is_moe']:
                inputs.append(io.Int.Input("n_cpu_moe", min=0, max=cached_data['max_n_gpu_layers'], default=0))
                inputs.append(io.Int.Input("expert_used_count", min=1, max=cached_data['expert_count'], default=cached_data['expert_used_count']))
                
            # name = Path(os.path.relpath(file, cls.gguf_folder)).parent.name
            name = Path(file).stem
            cls._path_map[name] = file
            option = io.DynamicCombo.Option(name, inputs)
            options.append(option)
        
        default_threads = max(1, multiprocessing.cpu_count() // 2)
        
        return io.Schema(
            node_id="DavchaLLMLoader",
            category="davcha/llm",
            inputs=[
                io.DynamicCombo.Input("model", options=options),
                io.Int.Input("n_batch", min=64, max=32768, default=2048),
                io.Int.Input("n_ubatch", min=64, max=4096, default=512),
                io.Int.Input("n_threads", min=1, max=128, default=default_threads),
                io.Combo.Input("kv_cache_type", options=["f16", "q8_0", "q4_0"], default="q8_0"),
                io.Boolean.Input("flash_attn", default=True),
            ],
            outputs=[
                io.Custom("DavchaLLMModel").Output(),
            ]
        )
    
    @classmethod
    def execute(cls, model, n_batch, n_ubatch, n_threads, kv_cache_type, flash_attn):
        n_ctx = model.get("n_ctx", 4096)
        n_gpu_layers = model.get("n_gpu_layers", -1)
        # Defaults to 0 safely for non-MoE models where "n_cpu_moe" is not present
        n_cpu_moe = model.get("n_cpu_moe", 0)
        expert_used_count = model.get("expert_used_count", None)
        
        model_name = model.get("model", None)
        if model_name is None:
            raise FileExistsError("model")
        
        model_file = cls._path_map[model_name]
        mmproj_files = glob(os.path.join(os.path.dirname(model_file), "*mmproj*.gguf"))
        mmproj = mmproj_files[0] if mmproj_files else None
        # models = sorted(glob(os.path.join(cls.gguf_folder, model_name, "*.gguf")), key=lambda x: "mmproj" in x)
        # if len(models) == 1:
        #     model_file = models[0]
        #     mmproj = None
        # elif len(models) == 2:
        #     model_file, mmproj = models
        # else:
        #     raise FileExistsError(f"Multiple GGUF files found for model '{model_name}' in {cls.gguf_folder}. Expected 1 or 2 (with mmproj), found {len(models)}.")
        
        if mmproj:
            mmproj = get_chat_handler(model_file, mmproj)

        kv_type = KV_CACHE_TYPES.get(kv_cache_type, KV_CACHE_TYPES["q8_0"])
        
        try:
            meta_llm = Llama(model_path=model_file, vocab_only=True, verbose=False)
            arch = meta_llm.metadata.get("general.architecture", "llama")
        except:
            arch = "llama"
            
        kv_overrides = {}
        if expert_used_count is not None:
            override_key = f"{arch}.expert_used_count"
            kv_overrides[override_key] = int(expert_used_count)
            print(f"[DavchaLLM] Overriding MoE Top-K: {override_key} = {expert_used_count}")

        llm = Llama(
            model_path=model_file,
            chat_handler=mmproj,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe=n_cpu_moe,
            kv_overrides=kv_overrides,
            type_k=kv_type,
            type_v=kv_type,
            flash_attn=flash_attn,
            swa_full=True,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            offload_kqv=True,
            logits_all=False,
            verbose=False
        )
        
        return io.NodeOutput(llm)

class DavchaLLM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaLLM",
            category="davcha/llm",
            inputs=[
                io.Custom("DavchaLLMModel").Input("llm"),
                io.String.Input("system", multiline=True, dynamic_prompts=False),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("max_tokens", min=1, default=512),
                io.Float.Input("temperature", min=0.0, max=2.0, default=0.9),
                io.Float.Input("top_p", min=0.0, max=1.0, default=0.95),
                io.Int.Input("top_k", min=0, max=100, default=40),
                io.Float.Input("repeat_penalty", min=0.5, max=2.0, default=1.2),
                io.Boolean.Input("enable_thinking", default=True),
                io.String.Input("response_format", multiline=True, dynamic_prompts=False, default=""),
                io.Image.Input("images", optional=True),
            ],
            outputs=[
                io.String.Output()
            ]
        )
    
    @classmethod
    def execute(cls, llm, system, prompt, seed, max_tokens, temperature, top_p, top_k, repeat_penalty, enable_thinking, response_format, images=None):
        if images is not None:
            content = []
            if not isinstance(images, list):
                images = [images]
            for image in images:
                array = (image[0]*255).clamp(0, 255).byte().cpu().numpy()
                pil_img = Image.fromarray(array, mode="RGB")
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content.append({'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{b64}"}})
            
            content.append({'type': 'text', 'text': prompt})

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
        response_format = json.loads(response_format) if response_format.strip() != "" else None
        
        with disable_thinking_context(llm, enable_thinking):
            result = llm.create_chat_completion(
                seed=seed,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                response_format=response_format,
            )
        
        result = strip_via_fuzzy_tags(result['choices'][0]['message']['content'], llm)
        
        return io.NodeOutput(result)

class DavchaPromptEnricher(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaPromptEnricher",
            category="davcha/llm",
            inputs=[
                io.Custom("DavchaLLMModel").Input("llm"),
                io.String.Input("system", multiline=True, dynamic_prompts=False, default="You will be presented with a prompt, an image, and dictionary keys. Keys are placeholders to be expanded to modify the image.\n\nKeep the SAME KEYS.\nKeep the SAME DICTIONARY STRUCTURE.\nProcess ALL keys.\nNO empty values.\nFit the values nicely in the prompt.\nEnsure all values are coherent with each other.\n\nPair each key with a description expanding the concept given by the key.\nWrite a string,string pair json dictionary.\nOnly output json. No commentary."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("max_tokens", min=1, default=512),
                io.Float.Input("temperature", min=0.0, max=2.0, default=0.9, step=0.01),
                io.Float.Input("top_p", min=0.0, max=1.0, default=0.95, step=0.01),
                io.Int.Input("top_k", min=0, max=100, default=40),
                io.Float.Input("repeat_penalty", min=0.5, max=2.0, default=1.2, step=0.01),
                io.Boolean.Input("enable_thinking", default=True),
                io.Boolean.Input("force_json_output", default=True),
                io.Image.Input("images", optional=True),
            ],
            outputs=[
                io.String.Output()
            ]
        )
    
    @classmethod
    def execute(cls, llm, system, prompt, seed, max_tokens, temperature, top_p, top_k, repeat_penalty, enable_thinking, force_json_output, images=None):
        if not re.search(r'\{([^}]+)\}', prompt):
            return io.NodeOutput(prompt)
        
        m = re.findall(r'\{([^}]+)\}', prompt)
        keys = {x: "" for x in m}

        p = f"""Here is a prompt with some variables:\n{prompt}\n\n---\n\nWrite detailed visual descriptions for the following variables: {keys.keys()}. Minimum length per key > 0. Result in the same JSON format: {keys}"""
        messages = [{"role": "system", "content": system or ""}] if system else []
        
        if images is not None:
            content = []
            if not isinstance(images, list):
                images = [images]
            for image in images:
                array = (image[0]*255).clamp(0, 255).byte().cpu().numpy()
                pil_img = Image.fromarray(array, mode="RGB")
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content.append({'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{b64}"}})
            
            content.append({'type': 'text', 'text': p})

            messages.append(
                {"role": "user", "content": content}
            )
        else:
            messages.append(
                {"role": "user", "content": p}
            )

        if force_json_output:
            response_format = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {key:{"type":"string"} for key in m},
                    "required": m
                }
            }
        else:
            response_format = None
        
        with disable_thinking_context(llm, enable_thinking):
            result = llm.create_chat_completion(
                seed=seed,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                response_format=response_format,
            )
        groups = result['choices'][0]['message']['content']
        print(f'------------------\nLLM Response\n------------------\n\n{groups}\n\n------------------')
        groups = strip_via_fuzzy_tags(groups, llm)
        
        import json_repair
        p = prompt
        j = json_repair.loads(groups)
        d = {}
        if isinstance(j, list):
            for item in j:
                d.update(item)
        else:
            d = j
        for k, v in d.items():
            # find correct key
            best_match = process.extractOne(
                query=k,
                choices=list(keys.keys()),
                scorer=fuzz.partial_ratio
            )
            if best_match:
                matched_key, *_ = best_match
                p = p.replace(f"{{{matched_key}}}", "\n".join([str(x) for x in v]) if isinstance(v, list) else str(v))
            else:
                print(f"unmatched key: {k}")
        
        return io.NodeOutput(p)