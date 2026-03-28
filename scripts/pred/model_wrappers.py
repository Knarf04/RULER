# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import requests
import torch
from typing import Dict, List, Optional
from torch import distributed as dist

def _strip_compiled_prefix(sd):
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

class FMSModel:
    def __init__(self, name_or_path: str, variant: str, tokenizer_path=None, **generation_kwargs) -> None:
        # Enable MiniKV cache eviction logging
        logging.basicConfig(level=logging.WARNING)
        for _log_name in ("fms.models.llama", "fms.utils.minikv"):
            logging.getLogger(_log_name).setLevel(logging.INFO)

        # Replicate lm_eval_harness device setup (huggingface.py:132-133)
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
        from datetime import timedelta
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self.device = accelerator.device

        from transformers import AutoTokenizer, pipeline
        from fms.models import get_model
        from fms import models
        from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM
        from fms.models.hf.llama.configuration_llama_hf import HFAdaptedLLaMAConfig
        from fms.models.hf.granite.modeling_granite_hf import HFAdaptedGraniteForCausalLM
        from fms.models.hf.granite.configuration_granite_hf import HFAdaptedGraniteConfig
        from fms_fsdp.utils.config_utils import get_model_config
        from fms.models.llama import _llama_factory_factory

        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(name_or_path) if os.path.isfile(name_or_path) else name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        _config_data = get_model_config(variant)
        _architecture_name, _variant = variant.split('_', 1)

        if _architecture_name == 'llama':
            models.register_model(_architecture_name, _variant, _llama_factory_factory(_config_data))
        elif _architecture_name == 'gdn':
            pass  # GDN uses fla model directly, no fms registration needed
        else:
            raise NotImplementedError()

        print(f'{_variant=}')

        # dist.init_process_group()
        # # Fix until PT 2.3
        # torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

        from fms.models.llama import LLaMA
        from torch.distributed._shard.checkpoint import FileSystemReader, load

        if _architecture_name == 'gdn':
            from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetConfig as FLAGDNConfig
            # fla uses vocab_size; fms config_utils returns src_vocab_size
            fla_config_data = dict(_config_data)
            if "src_vocab_size" in fla_config_data:
                fla_config_data["vocab_size"] = fla_config_data.pop("src_vocab_size")
            fla_config = FLAGDNConfig(**fla_config_data)
            self._fla_model = GatedDeltaNetForCausalLM(fla_config)
            print(f'{self._fla_model=}')

            print(f"Reading state dict from {name_or_path}")
            if name_or_path.endswith('.pth'):
                ckpt = torch.load(name_or_path, map_location="cpu")
                self._fla_model.load_state_dict(_strip_compiled_prefix(ckpt["model_state"]))
            else:
                state_dict = {"model_state": self._fla_model.state_dict()}
                load(state_dict=state_dict, storage_reader=FileSystemReader(name_or_path))
                self._fla_model.load_state_dict(_strip_compiled_prefix(state_dict["model_state"]))

            print("Loading state dict into the model...")
            self._fla_model.to(self.device, dtype=torch.bfloat16)
        else:
            self._fms_model = LLaMA(_config_data)
            print(f'{self._fms_model=}')

            print(f"Reading state dict from {name_or_path}")
            if name_or_path.endswith('.pth'):
                ckpt = torch.load(name_or_path, map_location="cpu")
                self._fms_model.load_state_dict(_strip_compiled_prefix(ckpt["model_state"]))
            else:
                state_dict = {"model_state": self._fms_model.state_dict()}
                load(state_dict=state_dict, storage_reader=FileSystemReader(name_or_path))
                self._fms_model.load_state_dict(_strip_compiled_prefix(state_dict["model_state"]))

            print("Loading state dict into the model...")
            self._fms_model.to(self.device, dtype=torch.bfloat16)
        # Disable 'tp' for universal attention, put *.pth
        # self._fms_model = get_model(
        #     _architecture_name,
        #     _variant,
        #     name_or_path,
        #     device_type='cuda',
        #     data_type=torch.bfloat16,
        #     distributed_strategy=None,
        #     checkpoint_sharding=None,
        #     linear_config={"linear_type": "torch_linear"},
        #     fused_weights=True,
        # )

        torch.set_grad_enabled(False)
        self.pipeline = None

        # Must disable weight init: from_fms_model triggers PreTrainedModel.__init__
        # -> post_init() -> init_weights() which re-initializes all submodule weights
        # in transformers >= 4.57.0 (where _init_weights changed from no-op to active).
        from transformers.modeling_utils import no_init_weights

        if _architecture_name == 'gdn':
            # fla model is already HF-compatible; wrap via HFAdaptedGDNForCausalLM
            self._fla_model.eval()
            print(f'{self._fla_model=}')
            print(f'{self._fla_model.config=}')

            from fms.models.hf.gated_delta_net.modeling_gated_delta_net_hf import HFAdaptedGDNForCausalLM
            from fms.models.hf.gated_delta_net.configuration_gated_delta_net_hf import HFAdaptedGDNConfig
            fms_hf_config = HFAdaptedGDNConfig.from_dict(self._fla_model.config.to_dict())
            with no_init_weights():
                self.model = HFAdaptedGDNForCausalLM._hf_model_from_fms(self._fla_model, fms_hf_config)
        else:
            self._fms_model.eval()
            print(f'{self._fms_model=}')
            print(f'{self._fms_model.config=}')

            if _architecture_name == 'llama':
                fms_hf_config = HFAdaptedLLaMAConfig.from_fms_config(self._fms_model.get_config())
                with no_init_weights():
                    self.model = HFAdaptedLLaMAForCausalLM.from_fms_model(self._fms_model, **fms_hf_config.to_dict())
            elif _architecture_name == 'mamba':
                raise NotImplementedError()
            elif _architecture_name == 'granite':
                fms_hf_config = HFAdaptedGraniteConfig.from_fms_config(self._fms_model.get_config())
                with no_init_weights():
                    self.model = HFAdaptedGraniteForCausalLM.from_fms_model(self._fms_model, **fms_hf_config.to_dict())
            else:
                raise NotImplementedError()

        self.model.eval()

        print(f'HF Adapted Version of Model: {self.model=}')

        generation_kwargs['use_cache'] = True

        print(f'Generation kwargs: {generation_kwargs}')

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, variant: str = None, tokenizer_path=None, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        # Replicate lm_eval_harness device setup (huggingface.py:132-133)
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
        from datetime import timedelta
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self.device = accelerator.device

        if variant is not None:
            # FMS checkpoint loading path — load on CPU, then move to device
            # (replicates lm_eval mamba_lm.py:_load_fms_checkpoint + _create_model)
            from mamba_ssm.models.config_mamba import MambaConfig
            from fms_fsdp.utils.config_utils import get_model_config
            from torch.distributed._shard.checkpoint import FileSystemReader, load

            if tokenizer_path is None:
                tokenizer_path = os.path.dirname(name_or_path) if os.path.isfile(name_or_path) else name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

            config_data = get_model_config(variant)
            config = MambaConfig(**config_data)
            self.model = MambaLMHeadModel(config)

            print(f"Reading state dict from {name_or_path}")
            if name_or_path.endswith('.pth'):
                ckpt = torch.load(name_or_path, map_location="cpu")
                self.model.load_state_dict(_strip_compiled_prefix(ckpt["model_state"]))
            else:
                state_dict = {"model_state": self.model.state_dict()}
                load(state_dict=state_dict, storage_reader=FileSystemReader(name_or_path))
                self.model.load_state_dict(_strip_compiled_prefix(state_dict["model_state"]))

            print("Loading state dict into the model...")
            self.model.to(self.device, dtype=torch.bfloat16)
        else:
            # Original HF-pretrained loading path
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)

        self.model.eval()
        torch.set_grad_enabled(False)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0
        # temperature=0 with top_k>1 causes div-by-zero; use top_k=1 for greedy
        if self.generation_kwargs.get('temperature', 1.0) == 0.0:
            self.generation_kwargs['temperature'] = 1.0
            self.generation_kwargs['top_k'] = 1

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=False,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
