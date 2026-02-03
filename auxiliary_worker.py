import ray
from vllm import LLM, SamplingParams
import string
import re

# @ray.remote(num_gpus=1)
# class VisualAuxiliaryWorker:
#     def __init__(self, model_path):
#         print(f"[JudgeWorker] Loading Aux Model: {model_path} ...")
#         self.llm = LLM(
#             model=model_path,
#             tensor_parallel_size=1,
#             gpu_memory_utilization=0.55, 
#             trust_remote_code=True,
#             dtype="auto"
#         )
#         self.sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
        
#     def clean_answer_content(self, s):
#         """
#         深度清洗：去除 'Answer:', 'Option:', 'The answer is' 等常见前缀，
#         只保留核心内容。
#         """
#         if not s: return ""
#         s = str(s).lower().strip()
        
#         # 1. 去除常见的前缀模式
#         prefixes = [
#             r"^answer\s*:\s*",        # Answer: ...
#             r"^answer\s+is\s+",       # Answer is ...
#             r"^the\s+answer\s+is\s+", # The answer is ...
#             r"^option\s*:\s*",        # Option: ...
#             r"^choice\s*:\s*",        # Choice: ...
#         ]
        
#         for pattern in prefixes:
#             s = re.sub(pattern, "", s).strip()
            
#         # 2. 去除标点 (VQA 标准操作)
#         exclude = set(string.punctuation)
#         s = ''.join(ch for ch in s if ch not in exclude)
        
#         # 3. 去除冠词
#         regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
#         s = re.sub(regex, " ", s)
        
#         # 4. 规范化空格
#         return " ".join(s.split())

#     def detect_repetition(self, text, ngram_size=4, threshold=0.3):
#         """检测重复（保持不变）"""
#         if not text or len(text) < 50: 
#             return False, ""

#         loop_pattern = re.compile(r'(.{10,})\1{2,}', re.DOTALL)
#         if loop_pattern.search(text):
#             return True, "loop_detected"

#         words = text.split()
#         if len(words) < ngram_size:
#             return False, ""
            
#         ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words)-ngram_size+1)]
#         unique_ngrams = set(ngrams)
#         ratio = len(unique_ngrams) / len(ngrams)
        
#         if ratio < threshold: 
#             return True, f"ngram_ratio_{ratio:.2f}"
#         return False, ""

#     def check_correctness(self, pred_clean, gt_clean):
#         """
#         更鲁棒的判题逻辑
#         """
#         # 1. 完全相等
#         if pred_clean == gt_clean:
#             return True
            
#         # 2. 针对单字母答案 (A, B, C, D) 的特殊处理
#         # 如果 GT 是单个字母 (如 "c")，要求 Pred 也是单个字母，或者 Pred 仅仅是 "c"
#         if len(gt_clean) == 1 and gt_clean.isalpha():
#              # 防止 "cat" 匹配 "c" (False Positive)
#              # 这里要求 pred 清洗后也是单个字母且相等，或者是单词边界匹配
#              if pred_clean == gt_clean:
#                  return True
#              # 如果 pred 是 "answer c"，清洗后变成了 "c"，上面已经 cover 了
#              return False

#         # 3. 包含关系 (针对长文本答案)
#         # GT 包含 Pred (GT="white cat", Pred="cat") -> True
#         # Pred 包含 GT (Pred="a white cat", GT="cat") -> True
#         # 增加长度限制，防止短单词匹配 (如 "no" in "not")
        
#         # 确保不是极短的子串匹配 (例如 'a' in 'apple')
#         if len(pred_clean) > 1 and len(gt_clean) > 1:
#             if gt_clean in pred_clean or pred_clean in gt_clean:
#                 return True
                
#         return False

#     def compute_rewards(self, prompts, ground_truths):
#         outputs = self.llm.generate(prompts, self.sampling_params)
#         rewards = []
#         predictions = []
        
#         for i, (output, gt, prompt_text) in enumerate(zip(outputs, ground_truths, prompts)):
#             raw_pred_text = output.outputs[0].text.strip()
#             pred_text_lower = raw_pred_text.lower()
            
#             # --- 1. 提取状态标签 ---
#             is_cheating = "[[cheat]]" in pred_text_lower or "cheat detected" in pred_text_lower
            
#             # --- 2. 清洗文本用于判题 ---
#             # 移除标签
#             text_for_eval = pred_text_lower.replace("[[cheat]]", "").replace("cheat detected", "").strip()
            
#             # 使用新的深度清洗函数
#             pred_clean = self.clean_answer_content(text_for_eval)
#             gt_clean = self.clean_answer_content(gt)
            
#             # --- 3. 判断对错 ---
#             is_correct = self.check_correctness(pred_clean, gt_clean)
            
#             # --- 4. 打分逻辑 ---
#             current_score = 0.0
#             status_tag = ""

#             if is_correct:
#                 current_score = 1.0
#                 if is_cheating:
#                     current_score -= 0.5
#                     status_tag = "[CHEAT -0.5]"
#                 else:
#                     is_repetitive, reason = self.detect_repetition(prompt_text)
#                     if is_repetitive:
#                         current_score -= 0.5
#                         status_tag = f"[REPEAT {reason} -0.5]"
#             else:
#                 current_score = 0.0
#                 if is_cheating:
#                     status_tag = "[CHEAT + WRONG]"
            
#             # 调试信息: 打印出到底匹配了什么，方便查错
#             # 格式: "Pred [TAG] (Clean: 'c' vs 'c')"
#             final_log_text = f"status_tag:{status_tag}\n auxiliary_model_output:{raw_pred_text}\n after filter:(Eval: '{pred_clean}' vs '{gt_clean}')".strip()
            
#             rewards.append(current_score)
#             predictions.append(final_log_text)
            
#         return rewards, predictions



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
        
        # 1. 去除常见的前缀模式
        prefixes = [
            r"^answer\s*:\s*",        # Answer: ...
            r"^answer\s+is\s+",       # Answer is ...
            r"^the\s+answer\s+is\s+", # The answer is ...
            r"^option\s*:\s*",        # Option: ...
            r"^choice\s*:\s*",        # Choice: ...
        ]
        
        for pattern in prefixes:
            s = re.sub(pattern, "", s).strip()
            
        # 2. 去除标点 (保留字母和数字)
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        
        # 🔥🔥🔥 修复点 1：保护单字母选项 🔥🔥🔥
        # 如果剩余内容仅仅是一个字母 (如 'a', 'b')，直接返回，不走后面的去冠词逻辑
        if len(s.split()) == 1 and len(s) == 1:
            return s
        
        # 3. 去除冠词 (仅当它不是唯一答案时)
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        s = re.sub(regex, " ", s)
        
        # 4. 规范化空格
        return " ".join(s.split())
    def check_negation(self, pred_text, gt_text, match_start_index):
        """
        检查匹配到的关键词前面是否有否定词。
        返回 True 表示被否定了（即匹配无效），False 表示没被否定（匹配有效）。
        """
        # 1. 截取关键词前面的文本（比如往前看 10-15 个字符）
        # 我们只看紧邻的前面几个词
        look_behind_length = 15
        start = max(0, match_start_index - look_behind_length)
        prefix = pred_text[start:match_start_index].lower()
        
        # 2. 定义否定词列表
        # 注意包含带空格的版本，防止匹配到单词内部
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
        
        # 清洗换行符，防止正则匹配不到跨行重复
        text_clean = text.replace('\n', ' ')

        # --- A. 单词级“结巴”检测 (针对 "and and and and") ---
        # 逻辑：匹配任何单词 (\b\w+\b)，后面紧跟着同样的单词 5 次以上
        # 例如: "and and and and and and"
        stutter_pattern = re.compile(r'\b(\w+)(?:\s+\1){5,}\b', re.IGNORECASE)
        match = stutter_pattern.search(text_clean)
        if match:
            return True, f"stuttering_word_'{match.group(1)}'"

        # --- B. 强力 Loop 检测 (针对长句子/段落重复) ---
        # 逻辑：查找任意长度超过 10 个字符，且连续出现 2 次以上的片段
        loop_pattern = re.compile(r'(.{10,})\1{2,}', re.DOTALL)
        if loop_pattern.search(text):
            return True, "loop_long_phrase"

        # --- C. N-gram 密度检测 (针对车轱辘话) ---
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
        # 1. 完美全等 (最高优先级)
        if pred_clean == gt_clean:
            return True
            
        # 2. 针对单字母答案 (A, B, C, D) 的特殊处理
        # 必须使用词边界匹配，且检查否定
        if len(gt_clean) == 1 and gt_clean.isalpha():
             pattern = r"\b" + re.escape(gt_clean) + r"\b"
             match = re.search(pattern, pred_clean)
             if match:
                 # 检查是否被否定 (例如 "not A")
                 if self.check_negation(pred_clean, gt_clean, match.start()):
                     return False # 虽然匹配到了 A，但是前面有 not，判错
                 return True
             return False

        # 3. 长文本包含匹配 (比如 GT="red", Pred="The apple is red")
        if len(gt_clean) > 1:
            # A. GT 包含 Pred (GT="white cat", Pred="cat") -> 这种通常不算对，除非你允许模糊
            # B. Pred 包含 GT (Pred="The apple is red", GT="red") -> 这种是我们想要的
            
            # 使用 find 查找位置
            idx = pred_clean.find(gt_clean)
            if idx != -1:
                # 找到了 GT，现在检查前面有没有否定词
                if self.check_negation(pred_clean, gt_clean, idx):
                    return False # 发现了 "not red"，判错
                
                # 还可以加一个词边界检查，防止 "scared" 匹配 "red"
                # 检查 GT 后面是不是字母
                end_idx = idx + len(gt_clean)
                if end_idx < len(pred_clean) and pred_clean[end_idx].isalpha():
                    return False # 匹配到了单词内部 (如 scared 里的 red)
                
                # 检查 GT 前面是不是字母 (如果没有被否定检测覆盖的话)
                if idx > 0 and pred_clean[idx-1].isalpha():
                    return False # 匹配到了单词内部
                    
                return True
                
        return False

    def compute_rewards(self, prompts, ground_truths):
        outputs = self.llm.generate(prompts, self.sampling_params)
        rewards = []
        predictions = []
        
        for i, (output, gt, prompt_text) in enumerate(zip(outputs, ground_truths, prompts)):
            raw_pred_text = output.outputs[0].text.strip()
            pred_text_lower = raw_pred_text.lower()
            
            # --- 1. 提取状态标签 ---
            is_cheating = "[[cheat]]" in pred_text_lower or "cheat detected" in pred_text_lower
            # no_cheating = "[[honest]]" in pred_text_lower
            # --- 2. 清洗文本用于判题 ---
            text_for_eval = pred_text_lower.replace("[[cheat]]", "").replace("cheat detected", "").strip()
            text_for_eval = pred_text_lower.replace("[[honest]]", "").strip()
            
            
            pred_clean = self.clean_answer_content(text_for_eval)
            gt_clean = self.clean_answer_content(gt)
            
            # --- 3. 判断对错 ---
            is_correct = self.check_correctness(pred_clean, gt_clean)
            
            # --- 4. 打分逻辑 ---
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
            
            # 调试信息
            # final_log_text = f"{raw_pred_text} {status_tag} (Eval: '{pred_clean}' vs '{gt_clean}')".strip()
            final_log_text = f"status_tag:{status_tag}\n auxiliary_model_output:{raw_pred_text}\n after filter:(Eval: '{pred_clean}' vs '{gt_clean}')".strip()

            rewards.append(current_score)
            predictions.append(final_log_text)
            
        return rewards, predictions