"""HuggingFace ``transformers`` backend. Lazy, GPU-friendly, no API key."""

from __future__ import annotations

import gc
import time
from collections import OrderedDict
from typing import Any

from .base import Backend, GenerateRequest, GenerateResult


class HFModelPool:
    """Loads at most ``max_loaded`` causal LMs at once.

    The harness iterates cell -> round -> agent -> cases, so each model is used
    for a long run of cases and then evicted. That keeps 3-model cells inside a
    24 GB GPU.
    """

    def __init__(
        self,
        max_loaded: int = 1,
        device_map: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        **_: Any,
    ) -> None:
        self.max_loaded = max(1, int(max_loaded))
        self.device_map = device_map
        self.dtype = dtype
        self.load_in_4bit = bool(load_in_4bit)
        self.trust_remote_code = bool(trust_remote_code)
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()

    def _dtype(self):
        import torch

        if self.dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16
            return torch.float32
        return getattr(torch, str(self.dtype).replace("torch.", ""))

    def _load(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = self._dtype()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except TypeError:
            # transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``.
            if "torch_dtype" not in kwargs:
                raise
            kwargs["dtype"] = kwargs.pop("torch_dtype")
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        return model, tokenizer

    def get(self, model_name: str):
        if model_name in self._cache:
            self._cache.move_to_end(model_name)
            return self._cache[model_name]
        while len(self._cache) >= self.max_loaded:
            old_name, (old_model, _) = self._cache.popitem(last=False)
            del old_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        model, tokenizer = self._load(model_name)
        self._cache[model_name] = (model, tokenizer)
        return model, tokenizer


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_name: str, pool: HFModelPool) -> None:
        self.model_name = model_name
        self.pool = pool

    def generate(self, request: GenerateRequest) -> GenerateResult:
        import torch

        start = time.perf_counter()
        model, tokenizer = self.pool.get(request.model or self.model_name)
        try:
            prompt = tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Base models often have no chat template. Fall back to a plain
            # role-prefixed transcript rather than failing the whole run.
            prompt = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in request.messages
            ) + "\n\nASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = request.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(request.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(request.temperature), 1e-5)
            gen_kwargs["top_p"] = float(request.top_p)

        if request.seed is not None:
            torch.manual_seed(int(request.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(request.seed))

        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerateResult(
            text=text.strip(),
            model=self.model_name,
            latency_s=time.perf_counter() - start,
        )

    def close(self) -> None:
        self.pool._cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
