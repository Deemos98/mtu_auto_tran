import subprocess, sys, os, re, time, json, yaml, requests, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import hashlib
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings("ignore")

# ================================================================
# 1. 物理路径锁定 (确保脚本在任何地方运行都能找到配置文件)
# ================================================================
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent 

# 统一使用绝对路径
CONFIG_DIR = PROJECT_ROOT / "config"
PARAM_JSON = PROJECT_ROOT / "examples" / "custom_api_params.json"
CURRENT_LOG_FILE = None
API_URL = "http://127.0.0.1:8080/v1/chat/completions"



def log_llm_transaction(log_path, filename, payload, response_data):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*30} BEGIN LLM TRANSACTION ({timestamp}) {'='*30}\n")
        f.write(f"FILE: {filename}\n")
        f.write(">>> [REQUEST PAYLOAD]:\n")
        f.write(json.dumps(payload, ensure_ascii=False, indent=4))
        f.write("\n\n<<< [RESPONSE FROM LLM]:\n")
        f.write(json.dumps(response_data, ensure_ascii=False, indent=4))
        f.write(f"\n{'='*30} END TRANSACTION {'='*30}\n")

# --- 新增日志函数：记录回写结果汇总 (tran_debug.log) ---
def log_tran_summary(log_path, filename, final_data):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] 文件: {filename} 回写汇总:\n")
        f.write(json.dumps(final_data, ensure_ascii=False, indent=4))
        f.write("\n" + "-"*50 + "\n")
        
def log_print(*args, **kwargs):
    """
    代替系统的 print 函数，同时输出到控制台和 print_debug.log
    用法: log_print("message", var)
    """
    # 将所有参数转为字符串
    msg = " ".join(map(str, args))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    
    # 1. 依然在控制台打印 (保持实时可见)
    print(full_msg, **kwargs)
    
    # 2. 写入文件
    if CURRENT_LOG_FILE:
        try:
            with open(CURRENT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")
        except: pass
    
log_print(f"--- 环境检查 ---")
log_print(f"项目根目录: {PROJECT_ROOT}")
log_print(f"配置目录: {CONFIG_DIR}")
# 2. 核心清洗与解析逻辑
# ================================================================
import xgboost as xgb
import re
import numpy as np
from pathlib import Path
class MangaSentinel:
    _shared_model = None
    def __init__(self, regions, model_path=r'C:\pythonProject\manga-translator-ui-main\manga_xgb.txt'):
        # 1. 严格锁定 BorutaShap 筛选后的 16 个特征顺序
        if MangaSentinel._shared_model is None:
            log_print("🏠 加载全局 XGBoost 审计模型...")
            MangaSentinel._shared_model = xgb.Booster()
            MangaSentinel._shared_model.load_model(str(model_path))
        self.feature_names = [
            'font_size_to_max_ratio', 'hira_ratio', 'punct_ratio', 'norm_y', 
            'size_prob_prod', 'digit_ratio', 'kata_ratio', 'color_diff', 
            'aspect_ratio', 'prob_zscore', 'angle', 'prob', 
            'emotion_ratio', 'alpha_numeric_ratio', 'dist_to_edge', 'ratio_color_prod'
        ]
        
        # 2. 加载模型
        self.model = MangaSentinel._shared_model
        
        # 3. 预计算当前页面的统计基准（消除拍脑袋阈值）
        self.pg_stats = self._calc_page_stats(regions)

    def _calc_page_stats(self, regions):
        """计算全页统计量，用于生成相对特征 (Z-Score, MaxRatio)"""
        if not regions: return {}
        
        probs = [r.get('prob', 0) for r in regions]
        sizes = [r.get('font_size', 0) for r in regions]
        
        return {
            'prob_mean': np.mean(probs),
            'prob_std': np.std(probs) if len(probs) > 1 else 0.1,
            'font_size_max': max(sizes) if sizes else 1,
            'total_count': len(regions)
        }

    def _get_char_stats(self, text):
        """镜像训练脚本中的 V6.6/V6.7 语言学指纹逻辑"""
        text = str(text)
        t_len = len(text) if len(text) > 0 else 1
        
        # 1. 平假名
        hira = len(re.findall(r'[\u3040-\u309f]', text))
        # 2. 片假名 (含扩展)
        kata = len(re.findall(r'[\u30a0-\u30ff\u31f0-\u31ff]', text))
        # 3. 汉字
        han = len(re.findall(r'[\u4e00-\u9fa5\u3400-\u4dbf]', text))
        # 4. 数字 (全半角)
        digits = len(re.findall(r'[0-9\uff10-\uff19]', text))
        # 5. 拉丁字母 (全半角)
        latin = len(re.findall(r'[a-zA-Z\uff21-\uff3a\uff41-\uff5a]', text))
        # 6. 人类标点 (全半角)
        punct_pattern = r'[!?. , :;()\[\]{}\uff01\uff1f\u3002\u3001\uff0c\uff1a\uff1b\u300c\u300d\u300e\u300f\uff08\uff09\u3010\u3011\u2026\u2014\u2015\u00b7\u30fb\u2022\u301c\u30fc-]'
        punct = len(re.findall(punct_pattern, text))
        # 7. 情绪符号
        emotions = len(re.findall(r'[\u2600-\u26ff\u2700-\u27bf\U0001f300-\U0001f6ff]', text))

        return {
            'len': t_len,
            'hira': hira / t_len,
            'kata': kata / t_len,
            'digit': digits / t_len,
            'latin': latin / t_len,
            'punct': punct / t_len,
            'emotion': emotions / t_len,
            'alpha_numeric': (digits + latin) / t_len
        }

    def predict_is_noise(self, reg, upscale, pg_w, pg_h):
        """执行推理判断"""
        if not reg: return False, 0.0
        
        text = str(reg.get('text', '')).strip()
        c = self._get_char_stats(text)
        
        # 提取基础物理值
        prob = reg.get('prob', 0)
        size = reg.get('font_size', 0)
        angle = abs(reg.get('angle', 0))
        center = reg.get('center', [0, 0])
        
        # 颜色对比度
        fg, bg = reg.get('fg_colors', [0,0,0]), reg.get('bg_colors', [255,255,255])
        color_diff = sum(abs(fg[i] - bg[i]) for i in range(3))
        
        # 归一化坐标 (0-1)
        norm_x = center[0] / (pg_w * upscale + 1e-6)
        norm_y = center[1] / (pg_h * upscale + 1e-6)
        dist_to_edge = min(norm_x, 1 - norm_x, norm_y, 1 - norm_y)

        # 核心：计算那 16 个被选中的特征
        feat_dict = {
            'font_size_to_max_ratio': size / (self.pg_stats['font_size_max'] + 1e-6),
            'hira_ratio': c['hira'],
            'punct_ratio': c['punct'],
            'norm_y': norm_y,
            'size_prob_prod': size * prob,
            'digit_ratio': c['digit'],
            'kata_ratio': c['kata'],
            'color_diff': color_diff,
            'aspect_ratio': c['len'] / (len(reg.get('lines', [])) + 1e-6),
            'prob_zscore': (prob - self.pg_stats['prob_mean']) / (self.pg_stats['prob_std'] + 1e-6),
            'angle': angle,
            'prob': prob,
            'emotion_ratio': c['emotion'],
            'alpha_numeric_ratio': c['alpha_numeric'],
            'dist_to_edge': dist_to_edge,
            'ratio_color_prod': (size / c['len']) / (color_diff + 1)
        }

        # 构造输入矩阵 (必须保证顺序与 self.feature_names 一致)
        X_input = np.array([[feat_dict[f] for f in self.feature_names]], dtype=np.float32)
        dmatrix = xgb.DMatrix(X_input, feature_names=self.feature_names)
        
        # 执行预测
        p_noise = self.model.predict(dmatrix)[0]
        
        # 审计日志：只记录高危疑似噪声
        if p_noise > 0.4:
            # 这里的 print 可以改为你的日志记录函数
            print(f"🕵️ [XGB审计] '{text[:10]}' -> P_Noise={p_noise:.4f} | Angle={int(angle)} | Hira={c['hira']:.2f}")

        # 判定标准：超过 0.5 认为噪声，建议在 auto_tran 中结合双阈值使用
        return (p_noise >= 0.5), p_noise

class MangaMemory:
    def __init__(self, model, root_dir='tran_memory'):
        self.model = model  # 外部传入 BGE-M3 模型
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)
        
        # 1. HNSW 索引：支持百万级 O(log N) 检索
        self.dim = 1024
        self.index_path = self.root_dir / "hnsw_vector.index"
        self.db_path = self.root_dir / "kv_metadata.jsonl"
        
        # 2. 内存状态
        self.data_store = []
        self.hash_table = {} # {hash: index_id} 快速去重
        
        self._init_db()

    def _normalize(self, text):
        """借鉴点：归一化逻辑"""
        import unicodedata
        # 转全角为半角/标准化，去空格，转小写
        text = unicodedata.normalize('NFKC', text)
        text = "".join(text.split())
        return text.lower()

    def _init_db(self):
        # 初始化 HNSW 索引 (M=32 是工业标准参数)
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.hnsw.efConstruction = 40  # 构件精度
            
        # 加载元数据
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8-sig') as f:
                for idx, line in enumerate(f):
                    item = json.loads(line)
                    self.data_store.append(item)
                    norm_text = self._normalize(item['orig'])
                    self.hash_table[hashlib.md5(norm_text.encode()).hexdigest()] = idx

    def search_and_prepare(self, query_text, threshold=0.85):
        """翻译前：1. 归一化 2. 检索 3. 返回最高相似度(用于后续去重判断)"""
        norm_query = self._normalize(query_text)
        query_vec = self.model.encode([norm_query], normalize_embeddings=True)
        
        if self.index.ntotal == 0:
            return "", 0.0, query_vec

        distances, indices = self.index.search(np.array(query_vec).astype('float32'), 3)
        
        best_score = distances[0][0] if len(distances[0]) > 0 else 0
        refs = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist > threshold:
                item = self.data_store[idx]
                refs.append(f"原文: {item['orig']} -> 译文: {item['trans']}")
        
        context = "\n[参考记忆]:\n" + "\n".join(refs) if refs else ""
        return context, best_score, query_vec

    def add_memory(self, original, translated, best_score, query_vec):
        """翻译后：利用 search 阶段拿到的 best_score 智能决定是否写入"""
        # 1. 物理去重：Hash 检查
        norm_text = self._normalize(original)
        h = hashlib.md5(norm_text.encode()).hexdigest()
        if h in self.hash_table:
            # 更新逻辑：可以在这里更新 item 的 count 和 last_seen
            return 
        
        # 2. 语义去重：如果 search 时发现相似度已经 > 0.98，就不写了
        if best_score > 0.98:
            return

        # 3. 写入索引与元数据
        idx_id = self.index.ntotal
        self.index.add(np.array(query_vec).astype('float32'))
        
        item = {
            "orig": original,
            "trans": translated,
            "count": 1,
            "last_seen": datetime.now().isoformat()
        }
        
        with open(self.db_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
        self.data_store.append(item)
        self.hash_table[h] = idx_id
        
        # 4. 定期持久化索引 (由于 HNSW 写入很快，可以每 100 条写一次)
        if idx_id % 100 == 0:
            faiss.write_index(self.index, str(self.index_path))
    def batch_search_and_prepare(self, query_texts, threshold=0.85):
        if not query_texts: return []
        norm_queries = [self._normalize(t) for t in query_texts]
        # 批量编码 (这是提速核心)
        query_vecs = self.model.encode(norm_queries, normalize_embeddings=True)
        
        results = []
        for i, vec in enumerate(query_vecs):
            # --- 核心修复：强制转回 2D 形状 [1, 1024] ---
            # 这样 add_memory 里的 self.index.add 就不再报 unpack 错误
            vec_2d = vec.reshape(1, -1)
            
            if self.index.ntotal == 0:
                results.append(("", 0.0, vec_2d))
                continue
            
            # 使用 2D 向量进行检索
            distances, indices = self.index.search(vec_2d.astype('float32'), 3)
            best_score = distances[0][0] if len(distances[0]) > 0 else 0
            
            refs = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and dist > threshold:
                    item = self.data_store[idx]
                    refs.append(f"原文: {item['orig']} -> 译文: {item['trans']}")
            
            context = "\n".join(refs) if refs else ""
            results.append((context, best_score, vec_2d))
        return results
def safe_clean_line(translated, original):
    """
    高级防护清洗：
    1. 标点符号白名单对比 (原文没有的标点直接删)
    2. 括号一致性检查 (原文没有成对括号，译文直接删)
    3. 省略号标准化与标点压缩
    """
    res = translated.strip()
    
    # ==========================================
    # 1. 括号一致性物理删除 (解决：翻译多了 () [] 【】)
    # ==========================================
    bracket_pairs = [('(', ')'), ('（', '）'), ('[', ']'), ('［', '］'), ('【', '】')]
    for open_b, close_b in bracket_pairs:
        # 如果原文里没有这种开口括号，但译文里有
        if open_b not in original and open_b in res:
            # 物理删除：连带括号内的内容一起删掉（通常里面是 A/user/剧情解释）
            # [^\{close_b}]* 表示匹配到闭合括号为止
            res = re.sub(f'\\{open_b}[^\\{close_b}]*\\{close_b}', '', res)
            # 如果只剩下了单边括号，也清理掉
            res = res.replace(open_b, "").replace(close_b, "")

    # ==========================================
    # 2. 斩首逻辑 (保持高效切断)
    # ==========================================
    res = re.split(r'」。?\s*——', res)[0] 
    res = re.split(r'」\s*——', res)[0]
    res = re.sub(r'」\s*[「（\[].*$', '」', res)

    # ==========================================
    # 3. 标点符号过滤：只保留原文中出现过的标点类型
    # ==========================================
    # 定义所有可能的标点
    full_punct_set = r'！!？?。，,、~～〜\-—=」』》>「『《<]'
    # 提取原文中存在的标点字符集
    orig_puncts = set(re.findall(f'[{full_punct_set}]', original))
    # 宽容性补丁：允许翻译中正常增加 句号、逗号、感叹号、问号
    orig_puncts.update(['。', '，']) 
    
    # 如果译文中出现了原文没有且不在宽容补丁里的标点，直接删掉
    def punct_filter(match):
        p = match.group(0)
        return p if p in orig_puncts else ""
    
    res = re.sub(f'[{full_punct_set}]', punct_filter, res)

    # ==========================================
    # 4. 省略号标准化 (解决：........)
    # ==========================================
    # 将 3 个以上的连续点、或 2 个以上的连续句号统一转为标准省略号
    res = re.sub(r'\.{3,}', '……', res)
    res = re.sub(r'。{2,}', '……', res)
    # 限制省略号最多只出现一个 (即：……)
    res = re.sub(r'（……）{2,}', '……', res)

    # ==========================================
    # 5. 全局标点压缩 (≥3 压成 2)
    # ==========================================
    punct_limit_set = r'！!？?。，,、~～〜\-—=」』》>「『《<]'
    res = re.sub(f'([{punct_limit_set}]{{3,}})', lambda m: m.group(1)[:2], res)

    # 6. 特殊平衡清理：如果没有左引号，删掉右引号
    if '「' not in res and res.endswith('」'):
        res = res[:-1]
    
    return res.strip()
def parse_textarea_output(raw_output, expected_keys, v_chunk_data):
    """
    V2.2.1 物理锚点解析器
    """
    # 1. 提取 <textarea> 标签内容
    content_match = re.search(r'<textarea>(.*?)</textarea>', raw_output, re.DOTALL)
    text_content = (content_match.group(1).strip() if content_match else raw_output.strip()).replace('\ufffd', '')
    
    # 2. 全局归一化：将所有可能的数字变体（包括 〇）映射为标准 0-9
    # 同时将各种全角括号映射为半角，方便下一步正则处理
    text_content = text_content.translate(str.maketrans(
        "０１２３４５６７８９〇", 
        "01234567890"
    ))

    # 3. 统一括号格式至标准 【N】
    # 正则解释：
    # [\[\(\<...]: 匹配任何类型的开口括号
    # \s*(\d+)\s*?: 匹配数字，允许数字前后有空格
    # \.?\s*: 允许数字后面带个点（LLM 习惯），如 【1.】
    # [\]\)\>...]: 匹配任何类型的闭合括号
    text_content = re.sub(r'[\[\(\<〈［（＜]\s*(\d+)\s*\.?\s*[\]\)\>〉］）＞]', r'【\1】', text_content)
    
    # 针对已经是 【】 格式但内部有空格或点的情况进行二次纠偏
    text_content = re.sub(r'【\s*(\d+)\s*\.?\s*】', r'【\1】', text_content)

    # 4. 基于【数字】标签提取
    matches = re.findall(r'【(\d+)】\s*(.*?)(?=【\d+】|$)', text_content, re.DOTALL)
    trans_map = {m[0]: m[1].strip() for m in matches if m[1].strip()}
    # 5.在提取成 Map 时，立刻应用安全清洗
    expected_count = len(expected_keys)
    
    trans_map = {}
    for m_id, m_text in matches:
        original_text = v_chunk_data.get(m_id, "") # 获取该序号对应的原文
        trans_map[m_id] = safe_clean_line(m_text, original_text)
    final_map = {}

    # --- 情况 A：发现了标签 ---
    if len(trans_map) > 0:
        found_ids = [m[0] for m in matches]
        found_texts = [m[1].strip() for m in matches]
        
        # 序号偏移纠偏 (如模型吐出 104, 105...)
        is_wrong_numbering = not any(idx in expected_keys for idx in found_ids)
        if is_wrong_numbering and len(found_texts) == expected_count:
            for i, k in enumerate(expected_keys):
                final_map[k] = found_texts[i]
        else:
            # 精准 ID 匹配
            for k in expected_keys:
                # 修正点：使用参数名 v_chunk_data 确保回退原文成功
                final_map[k] = trans_map.get(k, v_chunk_data[k])
        return final_map

    # --- 情况 B：模型吞了标签，按物理行对齐 ---
    raw_lines = [l.strip() for l in text_content.split('\n') if l.strip()]
    valid_lines = [l for l in raw_lines if "翻译" not in l and "Translation" not in l]
    
    for i, k in enumerate(expected_keys):
        if i < len(valid_lines):
            # 移除行首残留符号
            clean_line = re.sub(r'^\d+[\.、：:\s]*', '', valid_lines[i])
            final_map[k] = clean_line
        else:
            final_map[k] = v_chunk_data[k] # 修正点：使用参数名 v_chunk_data
            
    return final_map

def clean_translation_prefix(text, original_id):
    """支持清洗 【1】 这种新格式的前缀，以及物理清理 Emoji 审计标记"""
    # 1. 清洗物理 ID 前缀
    prefixes = [f"【{original_id}】", f"[{original_id}]", f"{original_id}.", f"{original_id}．"]
    res = text.strip()
    for p in prefixes:
        if res.startswith(p): 
            res = res[len(p):].strip()
    zombie_tags = [
        r'\s*user$', r'\s*assistant$', 
        r'\s*Input:$', r'\s*Output:$', r'\s*user\\n$'
    ]
    for tag_pattern in zombie_tags:
        res = re.sub(tag_pattern, '', res, flags=re.IGNORECASE)
    # 2. 清洗 Emoji 审计标记 (⚠️)
    # 使用正则清理行首可能残留的特殊 Emoji 符号
    res = re.sub(r'^[⚠️]+', '', res).strip()
    return res

# ================================================================
# 3. 翻译流水线 (回退到 TextArea 模式)
# ================================================================
# --- 辅助函数：切分字典 ---
def chunk_dict(data, size):
    """
    最稳健的切分函数：将字典转换为列表项后进行切片
    """
    items = list(data.items()) # 转换为 [(k1, v1), (k2, v2)...]
    for i in range(0, len(items), size):
        # 将切片后的列表重新转回字典返回
        yield dict(items[i:i + size])
        
def translate_pipeline(target_dir, lang_type="JP", memory=None, max_lines=10, char_limit=50, prob_threshold=0.4):
    global CURRENT_LOG_FILE
    work_dir = Path(target_dir).parent
    llm_log_path = work_dir / "llm_debug.log"
    tran_log_path = work_dir / "tran_debug.log"
    CURRENT_LOG_FILE = work_dir / "print_debug.log"
    json_dir = work_dir / "json" 
    params = json.load(open(PARAM_JSON, 'r'))['translator'] if PARAM_JSON.exists() else {}
    prompt_map = {
        "JP": "prompt_jp_sakura.yaml",
        "EN": "prompt_en.yaml",
        "CN": "prompt_cn.yaml",
    }
    prompt_file = prompt_map.get(lang_type.upper(), "prompt_cn.yaml")

    with open(CONFIG_DIR / prompt_file, 'r', encoding='utf-8') as f:
        sys_prompt = yaml.safe_load(f)['system_prompt'].replace('{target_language}', '简体中文').replace('{{{target_lang}}}', '简体中文')

    for txt_file in sorted(Path(target_dir).glob("*.txt")):
        if not txt_file.exists() or txt_file.stat().st_size == 0:
            log_print(f"⏩ 跳过空文件 (无OCR内容): {txt_file.name}")
            continue
        log_print(f"\n正在处理: {txt_file.name}")
        
        # --- Stage 1.1: 载入 prob 审计数据 ---
        meta_map = {}
        json_filename = txt_file.name.replace("_original.txt", "_translations.json")
        json_meta_path = json_dir / json_filename
        purge_indices = set() 
        all_regions = []
        upscale, pg_w, pg_h = 1, 1, 1
        if json_meta_path.exists():
            try:
                with open(json_meta_path, 'r', encoding='utf-8-sig') as jf:
                    m_data = json.load(jf)
                    img_node = m_data[list(m_data.keys())[0]]
                    all_regions = img_node.get('regions', [])
                    pg_w, pg_h = img_node.get('original_width', 1), img_node.get('original_height', 1)
                    upscale = img_node.get('upscale_ratio', 1)
                    for reg in all_regions:
                        for t_line in reg.get('texts', []):
                            meta_map[str(t_line).strip()] = reg
            except Exception as e:
                log_print(f"   ❌ 读取 JSON 审计失败: {e}")
        else:
            log_print(f"   ⚠️ 未找到审计文件: {json_meta_path} (请检查文件夹是否存在)")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                log_print(f"⏩ 跳过空文件: {txt_file.name}")
                continue
            data = json.loads(content)
        except json.JSONDecodeError as e:
            log_print(f"❌ 文件 {txt_file.name} 内容不是合法的 JSON: {e}")
            continue
        except Exception as e:
            log_print(f"❌ 读取 {txt_file.name} 发生未知错误: {e}")
            continue

        sentinel = MangaSentinel(all_regions)
        items = list(data.items())
        full_translated_page = {}
        valid_items_to_send = []
        regions_to_purge = set()
        for idx, reg in enumerate(all_regions):
            texts_list = reg.get('texts', [])
            if lang_type != "CN":
                is_noise, p_val = sentinel.predict_is_noise(reg, upscale, pg_w, pg_h)
                if is_noise:
                    purge_indices.add(idx)
                    regions_to_purge.add(id(reg))
                    for t in texts_list:
                        full_translated_page[str(t).strip()] = str(t)
                    log_print(f"   🚫 [XGB旁路] Prob={p_val:.4f} | 内容='{reg.get('text')[:10]}'")

        for real_key, val in items:
            t_key = str(real_key).strip()
            reg_meta = meta_map.get(t_key)
            if reg_meta and id(reg_meta) in regions_to_purge:
                full_translated_page[real_key] = real_key 
            else:
                valid_items_to_send.append((real_key, val))

        # 2. 动态切块翻译
        chunks = []
        temp_chunk = []
        if valid_items_to_send:
            for item in valid_items_to_send:
                val_len = len(item[1]) if lang_type == "JP" else len(item[1].split())
                if val_len > char_limit or len(temp_chunk) >= max_lines:
                    if temp_chunk: chunks.append(temp_chunk)
                    chunks.append([item]); temp_chunk = []
                else:
                    temp_chunk.append(item)
            if temp_chunk: chunks.append(temp_chunk)

        # 跑 API 翻译
        if lang_type != "CN":
            for raw_chunk in chunks:
                # ================================================================
                # 回退：逐行查询记忆库 (最原始且稳健的逻辑)
                # ================================================================
                raw_refs = []
                chunk_rag_info = {} 
                
                for idx, (real_key, original_val) in enumerate(raw_chunk):
                    # 修改 search_and_prepare，让它只返回具体的 "原文 -> 译文" 对
                    context, score, vec = memory.search_and_prepare(original_val)
                    if context:
                        raw_refs.append(context)
                    # 存入中转站，供后续写入记忆库使用
                    chunk_rag_info[str(idx+1)] = {"vec": vec, "score": score}
                
                # 合并成一个参考块
                if raw_refs:
                    unique_refs = list(set(raw_refs))
                    rag_prompt_addition = "\n" + "\n".join(unique_refs)
                else:
                    rag_prompt_addition = ""
                
                current_sys_prompt = sys_prompt + rag_prompt_addition
                # ================================================================

                v_chunk_data = {str(idx+1): val for idx, (key, val) in enumerate(raw_chunk)}
                v_input = "\n".join([f"【{k}】{v}" for k, v in v_chunk_data.items()])
                expected_count = len(raw_chunk)
                line_rules = []
                line_refs = []
                for i in range(1, expected_count + 1):
                    line_rules.append(f'line{i} ::= "【{i}】" [^\\n]+ "\\n"')
                    line_refs.append(f"line{i}")
                
                gbnf_rules = [f'root ::= "<textarea>\\n" {" ".join(line_refs)} "</textarea>"']
                gbnf_rules.extend(line_rules)
                gbnf_code = "\n".join(gbnf_rules) 
                try:
                    payload = {
                        "model": "local-model",
                        "messages": [
                            {"role": "system", "content": current_sys_prompt},
                            {"role": "user", "content": v_input}
                        ],
                        "grammar": gbnf_code,
                        "stop": ["</textarea>","<|im_end|>","<|endoftext|>"],
                        **params
                    }
                    
                    r = requests.post(API_URL, json=payload, timeout=120).json()
                    log_llm_transaction(llm_log_path, txt_file.name, payload, r)
                    
                    raw_response = r['choices'][0]['message']['content']
                    v_trans_res = parse_textarea_output(raw_response, list(v_chunk_data.keys()), v_chunk_data)
                    
                    for idx, (real_key, original_val) in enumerate(raw_chunk):
                        v_id = str(idx + 1)
                        translated_text = v_trans_res.get(v_id, original_val)
                        final_trans = clean_translation_prefix(translated_text, v_id)
                        full_translated_page[real_key] = final_trans
                        
                        # ================================================================
                        # 回退：写入记忆库
                        # ================================================================
                        if final_trans and final_trans != original_val:
                            info = chunk_rag_info.get(v_id)
                            if info:
                                memory.add_memory(
                                    original=original_val, 
                                    translated=final_trans, 
                                    best_score=info["score"], 
                                    query_vec=info["vec"]
                                )
                        # ================================================================
                        
                except Exception as e:
                    log_print(f"   ⚠️ 块处理失败 ({e})，保留原文")
                    for r_key, r_val in raw_chunk:      
                        full_translated_page[r_key] = r_val
        else:
            for raw_chunk in chunks:
                for r_key, r_val in raw_chunk:      
                    full_translated_page[r_key] = r_val
                    
        # 记录汇总并写回文件
        log_tran_summary(tran_log_path, txt_file.name, full_translated_page)
        with open(txt_file, 'w', encoding='utf-8') as f:
            json.dump(full_translated_page, f, ensure_ascii=False, indent=4)
        log_print(f"✅ {txt_file.name} 翻译成功并回写")
# 4. MTU 与坐标修复 (保持稳定版逻辑)
# ================================================================

def fix_and_scale_json(json_path):
    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0: return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return # 如果 JSON 损坏则跳过
    for img_key in data:
        img_data = data[img_key]
        if img_data.get('upscale_ratio') == 2:
            log_print(f"降维处理: {os.path.basename(json_path)}")
            for reg in img_data.get('regions', []):
                reg['center'] = [int(c / 2) for c in reg['center']]
                if 'lines' in reg:
                    new_lines = []
                    for line in reg['lines']:
                        new_line = [[int(p[0]/2), int(p[1]/2)] for p in line]
                        if len(new_line) >= 3: # 只有有效的多边形才保留
                            new_lines.append(new_line)
                    reg['lines'] = new_lines
                reg['font_size'] = max(12, int(reg['font_size'] / 2))
            img_data['upscale_ratio'] = 1
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

 
        
def run_mtu_command(image_path, lang_type="JP"):
    
    if lang_type == "JP" :
        config_file = "config_save_text_jp.json" 
    elif lang_type == "CN":
        config_file = "config_save_text_cn.json"
    else:
        config_file = "config_save_text_en.json"
    config_path = CONFIG_DIR / config_file
    out_path = Path(image_path).resolve().parent
    cmd = [sys.executable, "-m", "manga_translator", "--config", str(config_path), "-i", str(image_path), "-o", str(out_path), "--overwrite"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

def run_render_stage(image_path, output_path):
    config_path = CONFIG_DIR / "config_load_text.json"
    if os.path.exists(output_path): shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    subprocess.run([sys.executable, "-m", "manga_translator", "local", "--config", str(config_path), "--input", str(image_path), "--output", str(output_path)], cwd=str(PROJECT_ROOT), check=True)
    # 路径平铺
    redundant_dir = Path(output_path) / Path(image_path).name
    if redundant_dir.exists():
        for item in redundant_dir.iterdir():
            shutil.move(str(item), str(Path(output_path) / item.name))
        redundant_dir.rmdir()

# ================================================================
# 5. 执行入口
# ================================================================
from tqdm import tqdm

if __name__ == "__main__":
    # 采用普通的 for 循环，确保一本漫画跑完再跑下一本
    # 这保证了 LLM 翻译时，你的 Python 脚本只会在一个时间点发一个请求给 LLM
    all_paths = [r'C:\tk\迅雷下载\test']
    lang = "EN"
    BGE_MODEL_PATH = r"C:\pythonProject\manga-translator-ui-main\models\bge-m3"
    bge_model = SentenceTransformer(BGE_MODEL_PATH, device='cpu')
    memory = MangaMemory(model=bge_model, root_dir=PROJECT_ROOT / "tran_memory")
    err_files = []
    for path_str in tqdm(all_paths, desc="总体进度"):
        input_path = Path(path_str).resolve()
        # 1. OCR 阶段：这一步会调用我们破解后的 MTU
        # 它会利用内部多线程（asyncio）飞速完成 57 张图的 OCR
        # 由于是单进程，显存非常安全
        run_mtu_command(input_path, lang)

        # 2. 翻译阶段：在 OCR 全部写完硬盘后，进入单线程翻译
        # 你的 translate_pipeline 内部是按顺序读 TXT 的，不会有任何幻觉干扰
        work_dir = input_path / "manga_translator_work"
        originals_dir = work_dir / "originals"

        rendered_results = input_path / "成品"
        if originals_dir.exists():
            translate_pipeline(originals_dir, lang, memory=memory)
        
        json_dir = work_dir / "json"
        for jf in json_dir.glob("*.json"):
            fix_and_scale_json(jf)
        # 3. 渲染阶段
        try:
            run_render_stage(input_path, rendered_results)
        except subprocess.CalledProcessError as e:
            log_print(f"❌ {input_path.name} 渲染过程中进程崩溃 (Error {e.returncode})。")
            log_print(f"trackback: {traceback.print_exc()}")
            err_files.append(input_path)
        except Exception as e:
            log_print(f"❌ {input_path.name} 发生未知错误: {e}")
            err_files.append(input_path)
    faiss.write_index(memory.index, str(memory.index_path))
    log_print(f"✅ 所有项目处理完成，共 {len(all_paths)} 个，其中 {len(err_files)} 个项目处理失败。")
    log_print(f"❌ 失败项目列表: {err_files}")
   