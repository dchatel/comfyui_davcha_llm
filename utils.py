import os
import re
import inspect
from rapidfuzz import process, fuzz
from llama_cpp import llama_chat_format
from gguf import GGUFReader

def get_chat_handler(model_path: str, clip_model_path: str, threshold: float = 70.0):
        """
        Dynamically extracts multimodal chat handlers and uses RapidFuzz to map 
        the model filename to the handler, respecting **kwargs and inheritance.
        """
        # Extract valid chat handlers based purely on naming convention
        handlers = {}
        for name, cls in inspect.getmembers(llama_chat_format, inspect.isclass):
            if name.endswith("ChatHandler"):
                # Strip "ChatHandler" and lowercase (e.g., "Gemma4ChatHandler" -> "gemma4")
                clean_name = name.replace("ChatHandler", "").lower()
                handlers[clean_name] = cls

        if not handlers:
            raise RuntimeError("No chat handlers found in llama_cpp.llama_chat_format.")

        filename = os.path.basename(model_path).lower()
        
        best_match = None
        
        # Try matching against projector type first
        gguf = GGUFReader(clip_model_path)
        projector_type = gguf.get_field('clip.projector_type')
        if projector_type is not None:
            projector_type = projector_type.contents()
            best_match = process.extractOne(
                query=projector_type,
                choices=list(handlers.keys()),
                scorer=fuzz.partial_ratio
            )
        
        if not best_match or best_match[1] < threshold:
            # Try matching against the filename
            # Normalize the model filename for fuzzy matching
            # E.g., "gemma-4-8b-multimodal-q4_k.gguf" -> "gemma48bmultimodalq4kgguf"
            normalized_filename = re.sub(r'[^a-z0-9]', '', filename)

            # Fuzzy match using RapidFuzz
            best_match = process.extractOne(
                query=normalized_filename,
                choices=list(handlers.keys()),
                scorer=fuzz.partial_ratio
            )

        if not best_match:
            raise ValueError("Could not find any suitable chat handlers.")

        matched_key, score, _ = best_match

        # Enforce confidence threshold
        if score < threshold:
            # Fallback on default Llava
            handler_cls = llama_chat_format.Llava15ChatHandler
        else:
            handler_cls = handlers[matched_key]
            print(f"[Auto-Detect] Filename '{filename}' mapped to '{handler_cls.__name__}' (Fuzzy Score: {score:.1f}%)")
        
        # Safe Instantiation
        # Since classes like Gemma4ChatHandler use **kwargs, we just pass clip_model_path
        # and let Python's native MRO (Method Resolution Order) handle the routing.
        kwargs = {}
        
        # Check if handler_cls supports 'enable_thinking'
        try:
            sig = inspect.signature(handler_cls)
            has_enable_thinking = "enable_thinking" in sig.parameters
        except ValueError:
            # Fallback if signature extraction fails
            has_enable_thinking = False

        if has_enable_thinking:
            kwargs["enable_thinking"] = False

        try:
            return handler_cls(clip_model_path=clip_model_path, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"The matched handler '{handler_cls.__name__}' rejected the arguments. "
                f"Is this truly a multimodal model? Details: {e}"
            )

def strip_via_fuzzy_tags(text: str, llm, threshold: float = 85.0) -> str:
    """
    Finds all native tags and uses RapidFuzz on ALL possible pairs to 
    dynamically locate and strip the reasoning block, regardless of structural tags.
    """
    # 1. Extract all tags
    matches = list(re.finditer(r'<[^>]+>', text))
    verified =[]
    
    # 2. Verify they are single tokens in the model's vocabulary
    for m in matches:
        tag_str = m.group()
        tokens = llm.tokenize(tag_str.encode('utf-8'), special=True)
        if len(tokens) <= 2:
            verified.append(m)
            
    if not verified:
        return text
        
    # 3. Search ALL combinations of tags to find the reasoning pair
    best_score = 0
    best_pair = None
    
    # Compare every tag against every subsequent tag (n^2 complexity, but n is tiny)
    for i in range(len(verified)):
        for j in range(i + 1, len(verified)):
            score = fuzz.ratio(verified[i].group(), verified[j].group())
            if score > best_score:
                best_score = score
                best_pair = (verified[i], verified[j])
                
    # 4. Strip the thought block if a highly similar pair is found
    # 85.0+ safely captures <think>/</think> (93%) while rejecting <|im_start|>/<|im_end|> (77%)
    if best_pair and best_score > threshold:
        open_tag, close_tag = best_pair
        print(f"[Fuzzy Tags] Paired '{open_tag.group()}' and '{close_tag.group()}' (Score: {best_score:.1f}%).")
        
        # Reconstruct the string: Everything BEFORE the open tag + everything AFTER the close tag
        return text[:open_tag.start()] + text[close_tag.end():]

    # 5. Implicit Fallback
    # If the system prompt forced the model into a thought block, it might output
    # text immediately, followed ONLY by the closing tag (e.g. <channel|>).
    first_match = verified[0]
    if len(text[:first_match.start()].strip()) > 0:
        print(f"[Fuzzy Tags] Implicit thought block. Stripping up to '{first_match.group()}'.")
        return text[first_match.end():].strip()

    return text