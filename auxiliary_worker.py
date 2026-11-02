import ray
from vllm import LLM, SamplingParams
import string
import re

@ray.remote(num_gpus=1)
class VisualAuxiliaryWorker:
    def __init__(self, model_path):
        print(f"[JudgeWorker] Loading Aux Model: {model_path} ...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.55, 
            trust_remote_code=True,
            dtype="auto"
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
        
    def clean_answer_content(self, s):
        """
        深度清洗：去除 'Answer:', 'Option:', 'The answer is' 等常见前缀，
        并保护单字母选项不被当做冠词删除。
        """
        if not s: return ""
        s = str(s).lower().strip()
        
        prefixes = [
            r"^answer\s*:\s*",        # Answer: ...
            r"^answer\s+is\s+",       # Answer is ...
            r"^the\s+answer\s+is\s+", # The answer is ...
            r"^option\s*:\s*",        # Option: ...
            r"^choice\s*:\s*",        # Choice: ...
        ]
        
        for pattern in prefixes:
            s = re.sub(pattern, "", s).strip()
            
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        
        if len(s.split()) == 1 and len(s) == 1:
            return s
        
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        s = re.sub(regex, " ", s)
        
        return " ".join(s.split())
    def check_negation(self, pred_text, gt_text, match_start_index):
        """
        检查匹配到的关键词前面是否有否定词。
        返回 True 表示被否定了（即匹配无效），False 表示没被否定（匹配有效）。
        """
        look_behind_length = 15
        start = max(0, match_start_index - look_behind_length)
        prefix = pred_text[start:match_start_index].lower()
        
        negation_patterns = [
            r"not\s+$",       # "is not "
            r"no\s+$",        # "there is no "
            r"n't\s+$",       # "isn't "
            r"never\s+$",     # "never "
            r"neither\s+$",   # "neither "
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, prefix):
                return True # 发现否定词
        
        return False
    def detect_repetition(self, text, ngram_size=4, threshold=0.3):
        """
        检测 Caption 是否包含严重重复。
        """
        if not text or len(text) < 50: 
            return False, ""
        
        text_clean = text.replace('\n', ' ')

        stutter_pattern = re.compile(r'\b(\w+)(?:\s+\1){5,}\b', re.IGNORECASE)
        match = stutter_pattern.search(text_clean)
        if match:
            return True, f"stuttering_word_'{match.group(1)}'"

        loop_pattern = re.compile(r'(.{10,})\1{2,}', re.DOTALL)
        if loop_pattern.search(text):
            return True, "loop_long_phrase"

        words = text.split()
        if len(words) < ngram_size:
            return False, ""
            
        ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words)-ngram_size+1)]
        unique_ngrams = set(ngrams)
        
        ratio = len(unique_ngrams) / len(ngrams)
        
        if ratio < threshold: 
            return True, f"ngram_ratio_{ratio:.2f}"
            
        return False, ""

    def check_correctness(self, pred_clean, gt_clean):
        """
        智能判题逻辑：包含匹配 + 否定检测
        """
        if pred_clean == gt_clean:
            return True
            
        if len(gt_clean) == 1 and gt_clean.isalpha():
             pattern = r"\b" + re.escape(gt_clean) + r"\b"
             match = re.search(pattern, pred_clean)
             if match:
                 if self.check_negation(pred_clean, gt_clean, match.start()):
                     return False # 虽然匹配到了 A，但是前面有 not，判错
                 return True
             return False

        if len(gt_clean) > 1:
            idx = pred_clean.find(gt_clean)
            if idx != -1:
                if self.check_negation(pred_clean, gt_clean, idx):
                    return False 
                
                end_idx = idx + len(gt_clean)
                if end_idx < len(pred_clean) and pred_clean[end_idx].isalpha():
                    return False 

                if idx > 0 and pred_clean[idx-1].isalpha():
                    return False 
                    
                return True
                
        return False

    def compute_rewards(self, prompts, ground_truths):
        outputs = self.llm.generate(prompts, self.sampling_params)
        rewards = []
        predictions = []
        
        for i, (output, gt, prompt_text) in enumerate(zip(outputs, ground_truths, prompts)):
            raw_pred_text = output.outputs[0].text.strip()
            pred_text_lower = raw_pred_text.lower()
            
            is_cheating = "[[cheat]]" in pred_text_lower or "cheat detected" in pred_text_lower
            text_for_eval = pred_text_lower.replace("[[cheat]]", "").replace("cheat detected", "").strip()
            text_for_eval = pred_text_lower.replace("[[honest]]", "").strip()
            
            
            pred_clean = self.clean_answer_content(text_for_eval)
            gt_clean = self.clean_answer_content(gt)
            
            is_correct = self.check_correctness(pred_clean, gt_clean)
            
            current_score = 0.0
            status_tag = ""

            if is_correct:
                current_score = 1.0
                if is_cheating:
                    current_score -= 1
                    status_tag = "[[CHEAT]] reward-1"
                else:
                    is_repetitive, reason = self.detect_repetition(prompt_text)
                    if is_repetitive:
                        current_score -= 0.5
                        status_tag = f"[[REPEAT]] {reason} reward-0.5]"
            else:
                current_score = 0.0
                if is_cheating:
                    status_tag = "[[CHEAT]] + [[WRONG]]"
                else:
                    status_tag = "[[WRONG]]"
            
            final_log_text = f"status_tag:{status_tag}\n auxiliary_model_output:{raw_pred_text}\n after filter:(Eval: '{pred_clean}' vs '{gt_clean}')".strip()

            rewards.append(current_score)
            predictions.append(final_log_text)
            
        return rewards, predictions
