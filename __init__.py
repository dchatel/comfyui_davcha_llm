from comfy_api.latest import io, ComfyExtension
from .nodes import *

class DavchaLLMExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DavchaLLMLoader,
            DavchaLLM,
            DavchaPromptEnricher,
        ]
        
async def comfy_entrypoint() -> DavchaLLMExtension:
    return DavchaLLMExtension()