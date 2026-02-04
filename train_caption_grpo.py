import os
import sys
from typing import Optional,Union, List
import json
import time
import hydra
import ray
import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoProcessor

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils import hf_processor
from verl.utils import torch_functional as verl_F

try:
    from verl.models.transformers.qwen2_vl import get_rope_index
except ImportError:
    from verl.utils.model import get_rope_index 

from auxiliary_worker import VisualAuxiliaryWorker

AUX_MODEL_PATH = "/groups/g900403/home/share/lyl/models/Qwen2.5-7B-Instruct"

try:
    from verl.models.transformers.qwen2_vl import get_rope_index
except ImportError:
    from verl.utils.model import get_rope_index

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
OFFICIAL_CONFIG_PATH = os.path.join(VERL_ROOT, "verl/trainer/config")
TIME_STAMP = time.strftime("%m%d_%H%M")


VISUAL_CAPTION_SYSTEM_PROMPT = """You are an expert visual analyst. Given an image and a question, your goal is to provide a **comprehensive and detailed objective description** of the visual content.

**Context:** This description will be used by a blind person to reason about and answer the question. Therefore, your description must act as the **raw visual evidence** required for reasoning.

**Strict Guidelines:**
1. **Focus on Visual Evidence, NOT Conclusions:** Describe the visible elements, text, spatial relationships, colors, and specific details that are relevant to the question. Do not just state the final answer.
2. **Be Comprehensive:** Do not be concise. Include ALL visual details, no matter how small, if they are necessary to support the reasoning process for the question.
3. **No Direct Answers:** You must NOT explicitly output the answer(e.g., do not say "The answer is B", "It is a cat" or "Select Option A"). Instead, provide all the necessary visual evidence so that a blind person can deduce the answer themselves.
4. **Format:** Output ONLY the detailed caption."""

class Qwen2VLDataset(RLHFDataset):
    def __init__(self, *args, **kwargs):
        # 1. 正常初始化父类
        super().__init__(*args, **kwargs)
        
        self.vision_chat_template = r"""{% if messages is string %}{{ messages }}{% else %}{% for message in messages %}{% if 'content' in message %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}{% endif %}"""
        
        if self.tokenizer:
            self.tokenizer.chat_template = self.vision_chat_template
        if self.processor:
            self.processor.chat_template = self.vision_chat_template

        # 定义你的 System Prompt
        self.system_prompt = VISUAL_CAPTION_SYSTEM_PROMPT

    def _build_messages(self, example: dict):
        count = 0
        if not hasattr(self, '_debug_printed'):
            print(f"\n🔴 [DEBUG]: {list(example.keys())}")
            for k, v in example.items():
                if isinstance(v, str) and len(v) < 100: 
                    print(f"   -> {k}: {v}")
            self._debug_printed = True
        prompt_key = self.prompt_key if self.prompt_key in example else 'question'
        image_key = self.image_key if self.image_key in example else 'image'
        gt_key = next((k for k in ['answer', 'ground_truth', 'label','response'] if k in example), None)
        # gt_key = self.gt
        
        question = example.get(prompt_key, "Unknown Question")
        image_path = example.get(image_key, "cant find imagepath")
        gt = example.get(gt_key, "cant get gt")
        if 'extra_info' not in example:
            example['extra_info'] = {}
        
        example['extra_info']['question'] = question
        example['extra_info']['ground_truth'] = gt
        example['extra_info']['answer'] = gt    
        example['extra_info']['image_path'] = image_path
        

        if image_path:
            if isinstance(image_path, str):
                example[self.image_key] = [{"image": image_path}]
            elif isinstance(image_path, list):
                example[self.image_key] = [
                    {"image": img} if isinstance(img, str) else img 
                    for img in image_path
                ]
        else:
            example[self.image_key] = []

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image"}, 
                {"type": "text", "text": str(question) + "\nVisual Caption:"} 
            ]}
        ]
        
        return messages




class CaptionRewardManager:
    def __init__(self, tokenizer, judge_worker, rollout_log_file_path, num_samples_per_prompt):
        self.tokenizer = tokenizer
        self.judge_worker = judge_worker
        self.rollout_log_file_path = rollout_log_file_path
        self.num_samples_per_prompt = num_samples_per_prompt 
        self._has_checked_keys = False 
        
        
        os.makedirs(os.path.dirname(self.rollout_log_file_path), exist_ok=True)
        with open(self.rollout_log_file_path, "w") as f:
            f.write("") 
        print(f"✅ Log file initialized at: {self.rollout_log_file_path}")

    def __call__(self, batch, **kwargs):
        response_ids = batch.batch['responses']
        captions = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        non_tensor = batch.non_tensor_batch
        extra_infos = non_tensor.get('extra_info', [{} for _ in captions])
        reward_models = non_tensor.get('reward_model', [{} for _ in captions])
        batch_images = non_tensor.get('images', [None] * len(captions))
        
        prompt_list = non_tensor.get('prompt')
        
        if isinstance(prompt_list, np.ndarray):
            prompt_list = prompt_list.tolist()
            
        if prompt_list is None or len(prompt_list) == 0:
            if 'prompts' in batch.batch:
                prompt_list = self.tokenizer.batch_decode(batch.batch['prompts'], skip_special_tokens=True)
            else:
                prompt_list = ["<Unknown Prompt>"] * len(captions)
        
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list] * len(captions)
        
        prompts_for_judge = []
        ground_truths = []
        questions = []
        image_paths = [] 

        for i in range(len(captions)):
            caption = captions[i]
            info_item = extra_infos[i]
            if hasattr(info_item, 'item'): info_item = info_item.item()
            question = info_item.get('question', 'Unknown Question')
            questions.append(question)
            
            img_path = info_item.get('image_path') or info_item.get('image')
            if img_path is None and i < len(batch_images):
                img_path = batch_images[i]
            if isinstance(img_path, list):
                img_path = str(img_path)
            
            image_paths.append(str(img_path)) 
            
            gt = None
            if reward_models is not None and len(reward_models) > i:
                rm_item = reward_models[i]
                if hasattr(rm_item, 'item'): rm_item = rm_item.item()
                if rm_item:
                    gt = rm_item.get('ground_truth')
            
            if gt is None:
                gt = info_item.get('answer', '')
            ground_truths.append(gt)


            
            
            prompt2 = (
                f"<|im_start|>system\n"
                f"You are an impartial judge and a blind solver. I will provide a 'Visual Description' generated by an AI and a 'Question'.\n"
                f"\n"
                f"**Task 1: Cheat Detection**\n"
                f"Check if the description is trying to cheat. 'Cheating' means:\n"
                f"- Directly explicitly stating 'The answer is X' WITHOUT visual context.\n"
                f"- Providing the answer directly (e.g., just saying 'Yes', '2', or 'Blue') WITHOUT any descriptive sentence.\n"
                f"**Important Exception**: It is **NOT** cheating if the description states a visual fact that happens to be the answer (e.g., if the question is 'What color is the car?', saying 'The car is red' is VALID visual evidence, NOT cheating).\n"
                f"\n"
                f"**Task 2: Solve**\n"
                f"Use the description to answer the question briefly.\n"
                f"**Output Format (CRITICAL):**\n"
                f"If you detect cheating, start with: \"[[CHEAT]]\" and then follow with the answer . If not, start with: \"[[honest]]\" and give the answer.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Visual Description: {caption}\n\n"
                f"Question: {question}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompts_for_judge.append(prompt2)

        scores, aux_responses = ray.get(self.judge_worker.compute_rewards.remote(prompts_for_judge, ground_truths))
                
        try:
            with open(self.rollout_log_file_path, "a", encoding="utf-8") as f:
                for i in range(len(questions)):
                    current_prompt = prompt_list[i] if i < len(prompt_list) else "N/A"
                    is_cheating = "[[CHEAT]]" in aux_responses[i]
                    log_entry = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "question": questions[i],
                        "ground_truth": str(ground_truths[i]),
                        "image_path": image_paths[i],
                        "prompt_for_actor": current_prompt,
                        "actor_caption": captions[i].strip(),
                        "auxiliary_response": aux_responses[i],
                        "score": scores[i],
                        "is_cheating": is_cheating
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"❌ Error writing to log file: {e}")
        
        if len(questions) > 0:
            print("\n" + "="*20 + " 👀 MONITOR " + "="*20)
            print(f"❓ Q: {questions[0]}")
            # print(f"prompt for actor model: {prompt_list[0] if prompt_list else 'N/A'}")
            print(f"🖼️ Img: {image_paths[0]}") 
            print(f"🖼️ Cap: {captions[0].strip()[:100]}...") 
            print(f"🤖 Aux: {aux_responses[0]}")
            print(f"🏆 Score: {scores[0]}")
            print("="*50 + "\n")

        reward_tensor = torch.tensor(scores, dtype=torch.float32)
        if kwargs.get('return_dict', False):
            return {'reward_tensor': reward_tensor.unsqueeze(-1)}
        else:
            return reward_tensor

class VisualCaptionTrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, processor, role_worker_mapping, resource_pool_manager, train_dataset, val_dataset, reward_fn):
        self.tokenizer = tokenizer
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            reward_fn=reward_fn,
            val_reward_fn=reward_fn
        )

def maybe_add_ref_policy_worker(config, role_worker_mapping, mapping, ref_policy_cls):
    """
    Minimal port of official add_ref_policy_worker logic:
    Add Role.RefPolicy if KL loss or KL-in-reward is enabled.
    This is REQUIRED to produce `ref_log_prob` in the training batch.
    """
    from verl.trainer.ppo.utils import Role
    import ray

    need_ref = bool(config.algorithm.use_kl_in_reward) or bool(config.actor_rollout_ref.actor.use_kl_loss)
    if not need_ref:
        print("[CONFIG] RefPolicy not needed (no KL loss / no KL-in-reward).")
        return

    role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)

    mapping[Role.RefPolicy] = mapping.get(Role.ActorRollout, "default")

    print("[CONFIG] ✅ Added Role.RefPolicy worker because KL is enabled.")

@hydra.main(config_path=OFFICIAL_CONFIG_PATH, config_name='ppo_trainer', version_base=None)
def main(base_config: DictConfig):
    OUTPUT_DIR = "/groups/g900403/home/share/lyl/alignment/verl-0.6.1/examples/CAPTION_GRPO"
    local_config = OmegaConf.load(os.path.join(CURRENT_DIR, "config.yaml"))
    OmegaConf.set_struct(base_config, False)
    config = OmegaConf.merge(base_config, local_config)
    
    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    ray_init_kwargs["include_dashboard"] = False  
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

    if config.transfer_queue.enable:
        # Add runtime environment variables for transfer queue
        runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
        runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
        runtime_env_kwargs["env_vars"] = runtime_env_vars

    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


    
    
    print("output_dir:", OUTPUT_DIR)
    project_name = config.trainer.project_name
    experiment_name = config.trainer.experiment_name
    timestamp = time.strftime("%m%d_%H%M")
    exp_name_with_time = f"{experiment_name}_{timestamp}"
    
    base_results_dir = os.path.join(OUTPUT_DIR, "results")
    current_exp_dir = os.path.join(base_results_dir, project_name, exp_name_with_time)
    rollout_log_file_path = os.path.join(current_exp_dir, f"detailed_rollout_log_{timestamp}.jsonl")

    os.makedirs(current_exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(current_exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    config.trainer.default_local_dir = current_exp_dir
    config.trainer.experiment_name = exp_name_with_time
    
    with open(os.path.join(current_exp_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
    
    print(f"\n🚀 Experiment: {exp_name_with_time}")
    print(f"📂 Results Dir: {current_exp_dir}")
    print(f"📝 Logs will be saved to: {rollout_log_file_path}\n")
    
    print(f"Starting AuxiliaryWorker ({AUX_MODEL_PATH})...")
    aux_worker = VisualAuxiliaryWorker.remote(model_path=AUX_MODEL_PATH)

    policy_path = config.actor_rollout_ref.model.path
    print(f"Loading Components from {policy_path}...")
    tokenizer = AutoTokenizer.from_pretrained(policy_path, trust_remote_code=True)
    processor = hf_processor(policy_path, trust_remote_code=True)

    max_samples = config.data.get("max_samples", None)
    print(f"📉 Data Limit: Using {max_samples if max_samples else 'ALL'} samples")

    print("Initializing Datasets...")
    train_dataset = Qwen2VLDataset(
        data_files=config.data.train_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
        max_samples=max_samples
    )
    val_dataset = Qwen2VLDataset(
        data_files=config.data.val_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
        max_samples=max_samples
    )

    print("Initializing Resource Pool...")
    resource_pool_spec = {'default': [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: 'default'}
    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    
    maybe_add_ref_policy_worker(
    config=config,
    role_worker_mapping=role_worker_mapping,
    mapping=mapping,
    ref_policy_cls=ActorRolloutRefWorker,   # official also uses actor_rollout_cls
)
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()

    print("Initializing Trainer...")
    
    n_samples = config.actor_rollout_ref.rollout.n
    reward_fn = CaptionRewardManager(tokenizer=tokenizer, judge_worker=aux_worker, rollout_log_file_path=rollout_log_file_path, num_samples_per_prompt=n_samples)
    trainer = VisualCaptionTrainer(
        config=config, tokenizer=tokenizer, processor=processor,
        role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager,
        train_dataset=train_dataset, val_dataset=val_dataset, reward_fn=reward_fn
    )
    
    print(f"📝 Detailed logs will be saved to: {rollout_log_file_path}")
    print("Spawning Ray Workers...")
    trainer.init_workers()
    print("Start Fitting...")
    
    print("=== FINAL CONFIG CHECK ===")
    print("use_kl_loss =", config.actor_rollout_ref.actor.get("use_kl_loss", None))
    print("use_kl_in_reward =", config.algorithm.get("use_kl_in_reward", None))
    print("actor_rollout_ref.ref =", OmegaConf.to_yaml(config.actor_rollout_ref.get("ref", {})))
    print("==========================")

    trainer.fit()

if __name__ == '__main__':
    main()