import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
from glob import glob
from rapidfuzz import fuzz, process
from llama_cpp import Llama
from comfy_api.latest import io
from pathlib import Path
import folder_paths

from .utils import *

class DavchaLLMLoader(io.ComfyNode):
    gguf_folder = os.path.join(folder_paths.models_dir, "LLM", "GGUF")
    
    # Cache pour les métadonnées (pour accélérer define_schema et F5)
    _metadata_cache = {}
        
    @classmethod
    def define_schema(cls):
        path = os.path.join(cls.gguf_folder, "**", "*.gguf")
        files = list(filter(lambda x: "mmproj" not in x, glob(path, recursive=True)))
        
        # Nettoyage du cache si on a supprimé un fichier du dossier
        cls._metadata_cache = {k: v for k, v in cls._metadata_cache.items() if k in files}
        
        options = []
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
                    
                    cls._metadata_cache[file] = {
                        'mtime': mtime,
                        'max_n_ctx': max_n_ctx,
                        'max_n_gpu_layers': max_n_gpu_layers
                    }
                except Exception as e:
                    print(f"[DavchaLLM] Erreur lecture métadonnées {file}: {e}")
                    cls._metadata_cache[file] = {'mtime': mtime, 'max_n_ctx': 8192, 'max_n_gpu_layers': 99}
            
            cached_data = cls._metadata_cache[file]
            
            inputs = [
                io.Int.Input("n_ctx", min=512, max=cached_data['max_n_ctx'], default=4096),
                io.Int.Input("n_gpu_layers", min=-1, max=cached_data['max_n_gpu_layers'], default=-1),
            ]
                
            name = Path(os.path.relpath(file, cls.gguf_folder)).parent.name
            option = io.DynamicCombo.Option(name, inputs)
            options.append(option)
        
        return io.Schema(
            node_id="DavchaLLMLoader",
            category="davcha/llm",
            inputs=[
                io.DynamicCombo.Input("model", options=options),
                io.Int.Input("n_batch", min=64, max=32768, default=512),
                io.Int.Input("pool_size", min=1048576, max=10485760, default=4194304, step=524288),
            ],
            outputs=[
                io.Custom("DavchaLLMModel").Output(),
            ]
        )
    
    @classmethod
    def execute(cls, model, n_batch, pool_size):
        n_ctx = model.get("n_ctx", 4096)
        n_gpu_layers = model.get("n_gpu_layers", -1)
        model = model.get("model", None)
        if model is None:
            raise FileExistsError("model")
        
        models = sorted(glob(os.path.join(cls.gguf_folder, model, "*.gguf")), key=lambda x: "mmproj" in x)
        if len(models) == 1:
            model = models[0]
            mmproj = None
        elif len(models) == 2:
            model, mmproj = models
        else:
            raise FileExistsError(f"Multiple GGUF files found for model '{model}' in {cls.gguf_folder}. Expected 1 or 2 (with mmproj), found {len(models)}.")
        
        if mmproj:
            mmproj = get_chat_handler(model, mmproj)

        llm = Llama(
            model_path=model,
            chat_handler=mmproj,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            swa_full=True,
            n_batch=n_batch,
            pool_size=pool_size,
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
                io.String.Input("response_format", multiline=True, dynamic_prompts=False, default=""),
                io.Image.Input("images", optional=True),
            ],
            outputs=[
                io.String.Output()
            ]
        )
    
    @classmethod
    def execute(cls, llm, system, prompt, seed, max_tokens, temperature, top_p, top_k, repeat_penalty, response_format, images=None):
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
                io.Boolean.Input("force_json_output", default=True),
                io.Image.Input("images", optional=True),
            ],
            outputs=[
                io.String.Output()
            ]
        )
    
    @classmethod
    def execute(cls, llm, system, prompt, seed, max_tokens, temperature, top_p, top_k, repeat_penalty, force_json_output, images=None):
        if not re.search(r'\{([^}]+)\}', prompt):
            return io.NodeOutput(prompt)
        
        m = re.findall(r'\{([^}]+)\}', prompt)
        keys = {x: "" for x in m}

        p = f"""Here is a prompt with some variables:\n{prompt}\n\n---\n\nWrite detailed visual descriptions for the following variables. Minimum length: 1 sentence. Result in the same format:\n{keys}"""
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