import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


def _resolve_model_class(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = getattr(config, "architectures", [])
    if architectures and "Qwen3_5ForConditionalGeneration" in architectures:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    return AutoModelForCausalLM


def load_causal_lm(model_path, device_map="auto", torch_dtype="auto", **kwargs):
    """Load a causal LM with correct architecture class (Qwen3.5-aware).

    Consolidates the 6 duplicated _load_causal_lm() implementations.
    """
    ModelClass = _resolve_model_class(model_path)
    return ModelClass.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **kwargs,
    )


class ModelInference:
    def __init__(self, model_path, device="auto", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ModelClass = _resolve_model_class(model_path)
        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device

        self.model = ModelClass.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        self.base_model = getattr(self.model, "model",
                          getattr(self.model, "transformer",
                          getattr(self.model, "base_model", None)))
        self.lm_head = self.model.get_output_embeddings()
        if self.lm_head is None:
            self.lm_head = getattr(self.model, "lm_head", None)

        self._device = next(self.model.parameters()).device
        self.vocab_size = self.model.get_input_embeddings().weight.size(0)

    @property
    def device(self):
        return self._device

    def tokenize(self, text, max_length=2048, **kwargs):
        if isinstance(text, str):
            text = [text]
        return self.tokenizer(
            text, max_length=max_length, truncation=True,
            return_tensors="pt", padding=True, **kwargs
        )

    @torch.no_grad()
    def get_logits(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        ).logits

    @torch.no_grad()
    def chunked_logits(self, input_ids, attention_mask=None, chunk_size=4096):
        if self.base_model is None or self.lm_head is None:
            return self.get_logits(input_ids, attention_mask)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        hidden = self.base_model(input_ids, attention_mask=attention_mask)[0]
        B, S, H = hidden.shape
        logits = torch.empty(B, S, self.vocab_size, dtype=torch.float32, device=self.device)

        for i in range(0, S, chunk_size):
            h_chunk = hidden[:, i:i + chunk_size, :].contiguous()
            logits[:, i:i + chunk_size] = self.lm_head(h_chunk).float()
        return logits

    @torch.no_grad()
    def get_logprobs(self, input_ids, attention_mask=None, target_ids=None, chunked=False, chunk_size=4096):
        if chunked:
            logits = self.chunked_logits(input_ids, attention_mask, chunk_size)
        else:
            logits = self.get_logits(input_ids, attention_mask)
        logprobs = F.log_softmax(logits.float(), dim=-1)

        if target_ids is not None:
            target_ids = target_ids.to(self.device)
            return logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return logprobs

    @torch.no_grad()
    def get_hidden_states(self, input_ids, attention_mask=None, output_all=False):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        forward_target = self.base_model if self.base_model is not None else self.model
        outputs = forward_target(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=output_all
        )
        if output_all:
            return outputs.hidden_states
        return outputs[0]

    @torch.no_grad()
    def compute_loss(self, input_ids, attention_mask=None, chunked=False, chunk_size=4096):
        if chunked:
            logits = self.chunked_logits(input_ids, attention_mask, chunk_size)
        else:
            logits = self.get_logits(input_ids, attention_mask)
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous().to(self.device)
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="mean"
        )
        return loss

    @torch.no_grad()
    def compute_ppl(self, input_ids, attention_mask=None, chunked=False, chunk_size=4096):
        loss = self.compute_loss(input_ids, attention_mask, chunked, chunk_size)
        return torch.exp(loss).item()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, default="Hello, world!")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--chunked", action="store_true")
    args = parser.parse_args()

    inf = ModelInference(args.model_path, device=args.device)
    tokens = inf.tokenize(args.text, max_length=args.max_length)

    print(f"Model: {args.model_path}")
    print(f"Vocab size: {inf.vocab_size}")
    print(f"Input shape: {list(tokens['input_ids'].shape)}")
    print(f"Text tokens: {inf.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")

    logits = inf.get_logits(tokens["input_ids"], tokens.get("attention_mask"))
    print(f"Logits shape: {list(logits.shape)}")

    logprobs = inf.get_logprobs(tokens["input_ids"], tokens.get("attention_mask"))
    print(f"Logprobs shape: {list(logprobs.shape)}")

    ppl = inf.compute_ppl(tokens["input_ids"], tokens.get("attention_mask"), chunked=args.chunked)
    print(f"PPL: {ppl:.4f}")

    hidden = inf.get_hidden_states(tokens["input_ids"], tokens.get("attention_mask"))
    print(f"Last hidden shape: {list(hidden.shape)}")

    # per-token loss
    loss = inf.compute_loss(tokens["input_ids"], tokens.get("attention_mask"), chunked=args.chunked)
    print(f"Loss: {loss.item():.4f}")
