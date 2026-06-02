import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
from glob import glob

from llama_cpp import Llama
from comfy_api.latest import io
import folder_paths

from .utils import *

class DavchaLLMLoader(io.ComfyNode):
    gguf_folder = os.path.join(folder_paths.models_dir, "LLM", "GGUF")
    
    @classmethod
    def define_schema(cls):
        path = os.path.join(cls.gguf_folder, "**", "*.gguf")
        files = list(filter(lambda x: "mmproj" not in x, glob(path, recursive=True)))
        
        options = []
        for file in files:
            llm = Llama(file, vocab_only=True, verbose=False)
            arch = llm.metadata.get("general.architecture", "llama")
            max_n_ctx = int(llm.metadata.get(f"{arch}.context_length"))
            max_n_gpu_layers = int(llm.metadata.get(f"{arch}.block_count"))
            
            inputs = [
                io.Int.Input("n_ctx", min=512, max=max_n_ctx, default=4096),
                io.Int.Input("n_gpu_layers", min=-1, max=max_n_gpu_layers, default=-1),
            ]
                
            name = os.path.relpath(file, cls.gguf_folder)
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
        model = os.path.join(cls.gguf_folder, model)
        
        mmproj = next(iter(glob(os.path.join(os.path.dirname(model), "*mmproj*"))), None)
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
                io.String.Input("system", multiline=True, dynamic_prompts=False),
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
        keys = '\n'.join([f'"""{x.replace("\n", "\\n")}""": "",' for x in m])

        p = f"""PROMPT:
        {prompt}
        
        ---
        Fill the following dictionary:
        {{
            {keys}
        }}"""
        
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

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": p}
            ]

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
        groups = strip_via_fuzzy_tags(groups, llm)
        
        import json_repair
        p = prompt
        for k, v in json_repair.loads(groups).items():
            p = p.replace(f"{{{k}}}", "\n".join([str(x) for x in v]) if isinstance(v, list) else str(v))
        
        return io.NodeOutput(p)