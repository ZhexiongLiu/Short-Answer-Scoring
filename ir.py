import math
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback


class IRCallback(TrainerCallback):
    def __init__(self, model, dataset, data_collator, tracker, batch_size, split_num=1):
        super().__init__()
        self.split_num = split_num
        self.tracker = tracker
        self.batch_size = batch_size
        self.model = model.get_base_model()
        self.dataset = dataset
        self.data_collator = data_collator
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
        self.dataloader = iter(self.dataloader)
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'PhiForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'LlamaForSequenceClassification': 'model.model.layers',
            'MistralForSequenceClassification': 'model.model.layers',
            'PhiForSequenceClassification': 'model.model.layers',
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            print(model_class_name)
            raise NotImplementedError

        self.total_layers = len(eval('self.' + self.layers_attribute))
        self.importance_score = torch.zeros(self.total_layers)
        self.layer_norm_list = [1000] * self.total_layers
        self.gradient_norms = []
        self.per_layer_gradient_norms = []
        self.active_layers_indices = []
        self.trainable_module_name = []

        layers = eval('self.' + self.layers_attribute)
        for idx in range(self.total_layers):
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if param.requires_grad and name not in self.trainable_module_name:
                            self.trainable_module_name.append(name)

    def sampling_important_layer_gradient_norms(self):
        norms = []
        layers = []
        for i, layer_norm in enumerate(self.layer_norm_list):
            norms.append(layer_norm)
            layers.append(i)

        def split_layer_norms(layers, norms):
            if sum(norms) == 1000 * self.total_layers:
                norms = [random.random() for _ in range(self.total_layers)]

            ranked_norm = [x for x, y in sorted(zip(norms, layers))]
            ranked_layer = [y for x, y in sorted(zip(norms, layers))]

            var_list = []
            para1 = 0.35  # hyper-parameter in case distribution is very skewed
            para2 = 0.65  # hyper-parameter in case distribution is very skewed
            quantile1 = int(len(ranked_norm) * para1)
            quantile2 = int(len(ranked_norm) * para2)
            for k in range(1, len(ranked_norm)):
                low_norm = np.array(ranked_norm[:k])
                high_norm = np.array(ranked_norm[k:])
                total_var = np.var(low_norm) + np.var(high_norm)
                var_list.append(total_var)
            threshold = min(var_list[quantile1:quantile2 + 1])
            # threshold = min(var_list)
            split = var_list.index(threshold)
            high_norm_layer_selected = ranked_layer[split:]
            high_norm_selected = ranked_norm[split:]
            return high_norm_layer_selected, high_norm_selected

        high_norm_layer_selected, high_norm_selected = split_layer_norms(layers, norms)
        if self.split_num >= 2:
            high_norm_layer_selected, high_norm_selected = split_layer_norms(high_norm_layer_selected, high_norm_selected)
        select = torch.tensor(high_norm_layer_selected)

        return select

    def freeze_all_layers(self):
        layers = eval('self.' + self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_end(self, args, state, control, **kwargs):
        self.layer_norm_list = self.tracker.log_and_reset(state.global_step)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.switch_active_layers()

    def switch_active_layers(self):
        self.freeze_all_layers()
        layers = eval('self.' + self.layers_attribute)
        self.active_layers_indices = self.sampling_important_layer_gradient_norms()
        print(f"Total layers: {self.total_layers}, Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

        for idx in self.active_layers_indices:
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if name in self.trainable_module_name:
                            param.requires_grad = True


class CosineCallback(TrainerCallback):
    def __init__(self, model, tracker, split_num=1):
        super().__init__()
        self.split_num = split_num
        self.tracker = tracker
        self.model = model.get_base_model()

        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'PhiForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'LlamaForSequenceClassification': 'model.model.layers',
            'MistralForSequenceClassification': 'model.model.layers',
            'PhiForSequenceClassification': 'model.model.layers',
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            print(model_class_name)
            raise NotImplementedError

        self.total_layers = len(eval('self.' + self.layers_attribute))
        self.importance_score = torch.zeros(self.total_layers)
        self.layer_norm_list = [1000] * self.total_layers

        self.gradient_norms = []
        self.per_layer_gradient_norms = []
        self.active_layers_indices = []
        self.trainable_module_name = []
        layers = eval('self.' + self.layers_attribute)
        for idx in range(self.total_layers):
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if param.requires_grad and name not in self.trainable_module_name:
                            self.trainable_module_name.append(name)

    def sampling_important_layer_similarity(self):
        norms = []
        layers = []
        for i, layer_norm in enumerate(self.layer_norm_list):
            norms.append(layer_norm)
            layers.append(i)

        def split_layer_norms(layers, norms):
            if sum(norms) == 1000 * self.total_layers:
                norms = [random.random() for _ in range(self.total_layers)]

            ranked_norm = [x for x, y in sorted(zip(norms, layers))]
            ranked_layer = [y for x, y in sorted(zip(norms, layers))]

            var_list = []
            para1 = 0.35  # hyper-parameter in case distribution is very skewed
            para2 = 0.65  # hyper-parameter in case distribution is very skewed
            quantile1 = int(len(ranked_norm) * para1)
            quantile2 = int(len(ranked_norm) * para2)
            for k in range(1, len(ranked_norm)):
                low_norm = np.array(ranked_norm[:k])
                high_norm = np.array(ranked_norm[k:])
                total_var = np.var(low_norm) + np.var(high_norm)
                var_list.append(total_var)

            threshold = min(var_list[quantile1:quantile2])
            # threshold = min(var_list)
            split = var_list.index(threshold)

            high_norm_layer_selected = ranked_layer[:split]
            high_norm_selected = ranked_norm[:split]
            return high_norm_layer_selected, high_norm_selected

        high_norm_layer_selected, high_norm_selected = split_layer_norms(layers, norms)
        if self.split_num >= 2:
            high_norm_layer_selected, high_norm_selected = split_layer_norms(high_norm_layer_selected, high_norm_selected)
        select = torch.tensor(high_norm_layer_selected)

        return select

    def freeze_all_layers(self):
        layers = eval('self.' + self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_end(self, args, state, control, **kwargs):
        self.layer_norm_list = self.tracker.compute_and_reset(state.global_step)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.switch_active_layers()

    def switch_active_layers(self):
        self.freeze_all_layers()
        layers = eval('self.' + self.layers_attribute)
        self.active_layers_indices = self.sampling_important_layer_similarity()
        print(f"Total layers: {self.total_layers}, Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

        for idx in self.active_layers_indices:
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if name in self.trainable_module_name:
                            param.requires_grad = True


class WeightCallback(TrainerCallback):
    def __init__(self, model, split_num=1):
        super().__init__()
        self.split_num = split_num
        self.model = model.get_base_model()
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'PhiForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'LlamaForSequenceClassification': 'model.model.layers',
            'MistralForSequenceClassification': 'model.model.layers',
            'PhiForSequenceClassification': 'model.model.layers',
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            print(model_class_name)
            raise NotImplementedError

        self.total_layers = len(eval('self.' + self.layers_attribute))
        self.importance_score = torch.zeros(self.total_layers)
        self.layer_norm_list = [1000] * self.total_layers

        self.gradient_norms = []
        self.per_layer_gradient_norms = []
        self.active_layers_indices = []
        self.trainable_module_name = []
        layers = eval('self.' + self.layers_attribute)
        for idx in range(self.total_layers):
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if param.requires_grad and name not in self.trainable_module_name:
                            self.trainable_module_name.append(name)

    def sampling_important_layer_parameter_weights(self):
        norms = []
        layers = []
        for i, layer in enumerate(self.model.model.layers):
            weight = layer.self_attn.o_proj.weight
            if weight is not None:
                layer_norm = weight.norm(2).item()
            else:
                layer_norm = 0
            norms.append(layer_norm)
            layers.append(i)
            print(f"  Layer {i:2d}: weight parameter {layer_norm:.4f}")

        if sum(norms) == 1000 * self.total_layers:
            norms = [random.random() for _ in range(self.total_layers)]

        def split_layer_norms(layers, norms):
            if sum(norms) == 1000 * self.total_layers:
                norms = [random.random() for _ in range(self.total_layers)]

            ranked_norm = [x for x, y in sorted(zip(norms, layers))]
            ranked_layer = [y for x, y in sorted(zip(norms, layers))]

            var_list = []
            para1 = 0.35  # hyper-parameter in case distribution is very skewed
            para2 = 0.65  # hyper-parameter in case distribution is very skewed
            quantile1 = int(len(ranked_norm) * para1) + 1
            quantile2 = int(len(ranked_norm) * para2)
            for k in range(1, len(ranked_norm)):
                low_norm = np.array(ranked_norm[:k])
                high_norm = np.array(ranked_norm[k:])
                total_var = np.var(low_norm) + np.var(high_norm)
                var_list.append(total_var)

            threshold = min(var_list[quantile1:quantile2 + 1])
            split = var_list.index(threshold)

            high_norm_layer_selected = ranked_layer[split:]
            high_norm_selected = ranked_norm[split:]
            return high_norm_layer_selected, high_norm_selected

        high_norm_layer_selected, high_norm_selected = split_layer_norms(layers, norms)
        if self.split_num >= 2:
            high_norm_layer_selected, high_norm_selected = split_layer_norms(high_norm_layer_selected, high_norm_selected)
        select = torch.tensor(high_norm_layer_selected)

        return select

    def freeze_all_layers(self):
        layers = eval('self.' + self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        self.switch_active_layers()

    def switch_active_layers(self):
        self.freeze_all_layers()
        layers = eval('self.' + self.layers_attribute)
        self.active_layers_indices = self.sampling_important_layer_parameter_weights()

        print(f"Total layers: {self.total_layers}, Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

        for idx in self.active_layers_indices:
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if name in self.trainable_module_name:
                            param.requires_grad = True


class GradientTracker:
    def __init__(self, model):
        self.model = model
        self.param_to_layer = {}
        self.grad_norms = defaultdict(float)
        self._register_hooks()
        self.layer_norm_list = [1000] * len(self.model.base_model.model.model.layers)

    def _register_hooks(self):
        for i, layer in enumerate(self.model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    self.param_to_layer[param] = i
                    param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook_fn(grad):
            norm = grad.detach().data.norm(2).item()
            if np.isnan(norm) or np.isinf(norm):
                norm = 1000
            layer_idx = self.param_to_layer[param]
            self.grad_norms[layer_idx] += norm ** 0.5

        return hook_fn

    def log_and_reset(self, step):
        print(f"\n[Step {step}] Gradient Norms Per Transformer Layer:")
        for i in sorted(self.grad_norms.keys()):
            norm = math.sqrt(self.grad_norms[i])
            self.layer_norm_list[i] = norm
        for i in range(len(self.layer_norm_list)):
            print(f"  Layer {i:2d}: {self.layer_norm_list[i]:.4f}")
        self.grad_norms.clear()
        return self.layer_norm_list


class CosineSimilarityTracker:
    def __init__(self, model):
        self.model = model
        self.before_states = defaultdict(list)
        self.after_states = defaultdict(list)
        self.layer_cos_list = [1.0] * len(self.model.base_model.model.model.layers)
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.model.base_model.model.model.layers):
            layer.register_forward_hook(self._make_hook_before(i))
            layer.register_forward_hook(self._make_hook_after(i))

    def _make_hook_before(self, layer_idx):
        def hook_fn(module, inputs, outputs):
            self.before_states[layer_idx].append(inputs[0].detach().cpu().clone())

        return hook_fn

    def _make_hook_after(self, layer_idx):
        def hook_fn(module, inputs, outputs):
            self.after_states[layer_idx].append(outputs[0].detach().cpu().clone())

        return hook_fn

    def compute_and_reset(self, step):
        print(f"\n[Step {step}] Layer-wise Cosine Similarity:")
        for i in range(len(self.layer_cos_list)):
            if self.before_states[i] and self.after_states[i]:
                before = self.before_states[i][0].to("cuda")
                after = self.after_states[i][0].to("cuda")
                before = before.mean(dim=0)
                after = after.mean(dim=0)

                before_norm = before / before.norm(dim=1, keepdim=True)
                after_norm = after / after.norm(dim=1, keepdim=True)
                cos_sim = torch.mm(before_norm, after_norm.t()).float()
                similarity = (torch.trace(cos_sim) / cos_sim.size(0)).detach().cpu().item()

                self.layer_cos_list[i] = similarity
                print(f"  Layer {i:2d}: Cosine Similarity = {similarity:.4f}")
                del before, after, before_norm, after_norm, cos_sim

            self.before_states[i].clear()
            self.after_states[i].clear()
        return self.layer_cos_list