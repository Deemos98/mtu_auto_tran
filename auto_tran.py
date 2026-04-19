import subprocess
import sys
import os
import re
import time
import json
import os
import re
import yaml
import requests
from pathlib import Path
def write_trans_log(log_path, filename, status, original=None, response=None):
    """在工作目录下记录翻译日志以便 Debug"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] 文件: {filename} | 状态: {status}\n")
        if original:
            f.write(f"--- 原文 Chunks ---\n{original}\n")
        if response:
            f.write(f"--- LLM 原始响应 ---\n{response}\n")
        f.write("-" * 50 + "\n")
def run_mtu_command(image_path, lang_type="JP"):
    """
    运行 MTU 的 CLI 命令
    使用 sys.executable 确保调用的是当前 venv 的 python
    """
    # 构造命令列表，使用列表方式传参最安全，自动处理空格和引号
    match = re.match(r'^(.*)\\[^\\]+$', image_path)
    if match:
        out_path = match.group(1)
    if lang_type == "JP":
        config_dir = r"config\config_save_text_jp.json"
    elif lang_type == "CN":
        config_dir = r"config\config_save_text_cn.json"
    else:
        config_dir = r"config\config_save_text_en.json"
    cmd = [sys.executable, "-m", "manga_translator", "--config", config_dir, "-i", image_path, "-o", out_path , "--overwrite"]
    
    # 定义工作目录（确保路径正确）
    # 如果你的图片在 C:\pics，这里可以传该路径
    working_dir = os.path.dirname(os.path.abspath(image_path))
    
    print(f"正在执行: {' '.join(cmd)}")
    
    # 执行命令
    # capture_output=True 可以捕获日志，方便调试
    # text=True 让输出直接显示为字符串而不是字节流
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        encoding='utf-8' 
    )
    
    if result.returncode == 0:
        print("✅ 命令执行成功！")
        print(result.stdout)
    else:
        print("❌ 命令执行失败！")
        print("错误信息:", result.stderr)
        
def fix_and_scale_json(json_path):
    """
    物理降维打击：将 2x 坐标转换为 1x，彻底解决嵌字空白问题
    """
    if not os.path.exists(json_path):
        print(f"跳过：找不到JSON文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img_key in data:
        img_data = data[img_key]
        # 核心逻辑：只有 upscale_ratio 为 2 时才执行
        if img_data.get('upscale_ratio') == 2:
            print(f"检测到超分数据: {os.path.basename(img_key)}，正在进行坐标放缩...")
            
            for reg in img_data.get('regions', []):
                # 1. 缩放中心点
                reg['center'] = [c / 2 for c in reg['center']]
                
                # 2. 缩放文本行坐标点 (四角坐标)
                if 'lines' in reg:
                    new_lines = []
                    for line in reg['lines']:
                        new_lines.append([[p[0]/2, p[1]/2] for p in line])
                    reg['lines'] = new_lines
                
                # 3. 缩放字体大小 (非常关键，否则字会巨大)
                reg['font_size'] = max(12, int(reg['font_size'] / 2))
                
                # 4. 清理翻译内容中的标识（双重保险）
                if reg.get('translation'):
                    reg['translation'] = clean_llm_text(reg['translation'])

            # 5. 修正元数据：告诉 MTU 渲染器现在已经是 1x 空间了
            img_data['upscale_ratio'] = 1
            
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 坐标放缩完成！")


# --- 1. 基础配置与路径 ---
CONFIG_DIR = Path(r".\config")
PARAM_JSON = Path(r"examples/custom_api_params.json")
API_URL = "http://127.0.0.1:8080/v1/chat/completions" # 根据您的模型后端调整

# --- 2. 核心清洗函数 (并集优化版) ---

def natural_sort_key(s):
    """文件名自然排序: 1.txt < 2.txt < 10.txt"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def clean_mtu_json_text(content):
    """
    清洗MTU导出的txt内容，使其符合标准JSON格式
    并集处理：修复非法转义、修复行内双引号
    """
    # 修复非法转义序列
    content = re.sub(r'\\(?![ntrfb"\\/]|u[0-9a-fA-F]{4})', r'\\\\', content)
    
    # 修复行内非法双引号 (利用正则匹配 ID: "Text" 结构)
    # 只保留 key 和 value 两侧的引号，中间的引号转为单引号
    def fix_quotes(match):
        key, val = match.groups()
        val = val.replace('"', "'")
        return f'"{key}": "{val}"'
    
    clean_content = re.sub(r'"(\d+)":\s*"(.*)"', fix_quotes, content)
    return clean_content

# --- 3. 文本分块与处理逻辑 ---

def chunk_text_dict(text_dict, max_lines=30):
    """将大字典切分为小块，防止LLM疲劳"""
    items = list(text_dict.items())
    for i in range(0, len(items), max_lines):
        yield dict(items[i:i + max_lines])

def load_prompt(lang_type):
    file_name = "prompt_jp_sakura.yaml" if lang_type == "JP" else "prompt_en.yaml"
    path = CONFIG_DIR / file_name
    if not path.exists(): return ""
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    system_prompt = data.get('system_prompt', '')
    
    # 彻底替换所有可能的占位符
    target_lang = "简体中文"
    source_lang = "日语" if lang_type == "JP" else "英语"
    
    system_prompt = system_prompt.replace('{target_language}', target_lang)
    system_prompt = system_prompt.replace('{{{target_lang}}}', target_lang)
    system_prompt = system_prompt.replace('{source_language}', source_lang)
    
    return system_prompt

def load_llm_params():
    """加载API参数"""
    if PARAM_JSON.exists():
        try:
            with open(PARAM_JSON, 'r', encoding='utf-8') as f:
                return json.load(f).get('translator', {})
        except Exception as e:
            print(f"加载API参数失败: {e}")
    return {"temperature": 0.3, "top_p": 0.8, "max_tokens": 4096}

def call_local_llm_with_retry(system_prompt, user_input, params, max_retries=3):
    """带重试机制的本地LLM调用"""
    for attempt in range(max_retries):
        print(f"\n尝试第 {attempt + 1} 次翻译请求...")
        
        if attempt > 0:
            print("等待2秒后重试...")
            import time
            time.sleep(2)
        
        try:
            result = call_local_llm(system_prompt, user_input, params)
            if result and not result.startswith("[MISSING]"):
                print(f"第 {attempt + 1} 次尝试成功")
                return result
            else:
                print(f"第 {attempt + 1} 次尝试失败，结果为空或无效")
                if attempt == max_retries - 1:
                    return None
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试异常: {e}")
            if attempt == max_retries - 1:
                return None
    
    return None

# --- 4. 翻译与对齐校验 ---

def parse_llm_output(raw_output, original_keys):
    """
    解析 LLM 输出，支持多种格式并强制对齐
    """
    print(f"原始LLM输出: {raw_output}")
    
   # 1. 提取 <textarea> 内容
    content_match = re.search(r'<textarea>(.*?)</textarea>', raw_output, re.DOTALL)
    text_content = content_match.group(1).strip() if content_match else raw_output.strip()

    # 2. 归一化处理：全角转半角 (针对数字、句号、冒号)
    # 建立映射表：全角 0-9, 句号, 冒号, 间隔号
    full_width = "０１２３４５６７８９．：・·"
    half_width = "0123456789..::"
    trans_table = str.maketrans(full_width, half_width)
    normalized_content = text_content.translate(trans_table)

    # 3. 增强版正则匹配
    # 逻辑：行首数字 + (点/冒号/空格/间隔号中的任意个) + 译文
    # 支持格式: "1. 译文", "1．译文", "1:译文", "1 译文", "1·译文"
    pattern = re.compile(r'^\s*(\d+)\s*[:\.．：·・\s-]\s*(.*)$', re.MULTILINE)
    matches = pattern.findall(normalized_content)
    
    # 4. 清洗翻译文本 (物理清除 Qwen 标识残留)
    trans_map = {}
    for m in matches:
        val = m[1].strip()
        # 清除 <|1|> 或 [1] 这种标识
        val = re.sub(r'^[<|\|\[\]\s>]+', '', val)
        val = re.sub(r'[<|\|\[\]\s>]+$', '', val)
        trans_map[m[0]] = val

    print(f"解析结果 (前3项): {list(trans_map.items())[:3]}")

    # 5. 强制映射校验
    final_map = {}
    missing_count = 0
    for k in original_keys:
        if k in trans_map:
            final_map[k] = trans_map[k]
        else:
            final_map[k] = "[MISSING]"
            missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️ 警告：该页有 {missing_count} 行匹配失败，请检查归一化逻辑。")

    
    print(f"最终翻译映射: {final_map}")
    return final_map

def call_local_llm(system_prompt, user_input, params):
    """调用本地 127.0.0.1 接口"""
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        **params
    }
    
    print(f"发送API请求到: {API_URL}")
    # print(f"请求参数: {payload}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        print(f"API响应状态码: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return ""
            
        response_data = response.json()
        print(f"API响应数据: {response_data}")
        
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            print("API响应中没有choices字段")
            return ""
            
        if 'message' not in response_data['choices'][0]:
            print("API响应中没有message字段")
            return ""
            
        return response_data['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        print("API请求超时")
        return ""
    except requests.exceptions.ConnectionError:
        print("API连接失败，请检查本地LLM服务是否启动")
        return ""
    except json.JSONDecodeError as e:
        print(f"API响应JSON解析失败: {e}")
        print(f"响应内容: {response.text}")
        return ""
    except Exception as e:
        print(f"API请求失败: {e}")
        return ""

# --- 5. 工程化回写逻辑 ---

def save_translated_txt(file_path, translated_dict):
    """以标准 JSON 格式写回原 TXT 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(translated_dict, f, ensure_ascii=False, indent=4)

# --- 6. 主流程控制 ---

def translate_pipeline(target_dir, lang_type="JP"):
    work_dir = Path(target_dir).parent
    debug_log = work_dir / "translation_debug.log"
    
    print(f"开始翻译，日志记录于: {debug_log}")
    params = load_llm_params()
    sys_prompt = load_prompt(lang_type)
    txt_files = sorted(Path(target_dir).glob("*.txt"), key=natural_sort_key)
    
    for txt_file in txt_files:
        print(f"\n处理文件: {txt_file.name}")
        with open(txt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        full_translated_page = {}
        
        # 1. 建立虚拟 ID 映射关系
        # original_keys 将存储真正的 Key (如 "東京 17:30")
        # llm_input_dict 将存储 { "1": "東京 17:30", "2": "私、大宮美奈子は" }
        original_keys = list(data.keys())
        llm_input_dict = {str(i+1): data[original_keys[i]] for i in range(len(original_keys))}

        # 2. 分块翻译
        for chunk in chunk_text_dict(llm_input_dict, max_lines=30):
            # 现在的 user_input 绝对是 1.xxx 2.xxx
            user_input = "\n".join([f"{k}.{v}" for k, v in chunk.items()])
            
            raw_response = call_local_llm_with_retry(sys_prompt, user_input, params)
            
            # 3. 解析 (注意现在 original_keys 传入的是虚拟数字 ID)
            chunk_result = parse_llm_output(raw_response, chunk.keys())
            
            if chunk_result:
                write_trans_log(debug_log, txt_file.name, "成功", user_input, raw_response)
                # 4. 映射回原始的 Key
                for virtual_id, translated_text in chunk_result.items():
                    # 通过虚拟 ID 找到原始索引
                    idx = int(virtual_id) - 1
                    real_key = original_keys[idx]
                    full_translated_page[real_key] = translated_text
            else:
                print(f"警告: {txt_file.name} 块翻译失败，已记录到 Log")
                write_trans_log(debug_log, txt_file.name, "解析失败/幻觉", user_input, raw_response)
                # 失败退避：将该块映射回原文
                for virtual_id in chunk.keys():
                    idx = int(virtual_id) - 1
                    real_key = original_keys[idx]
                    full_translated_page[real_key] = data[real_key]

        # 5. 回写
        save_translated_txt(txt_file, full_translated_page)
        print(f"回写成功: {txt_file.name}")

def run_render_stage(image_path, output_path, config_path=r"config\config_load_text.json"):
    """
    运行 MTU 渲染阶段
    优化：使用时间戳和更专业的目录命名
    """
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path), "rendered_results")
    os.makedirs(output_path, exist_ok=True)
    cmd = [
        sys.executable, "-m", "manga_translator", 
        "local",
        "--config", config_path,
        "--input", image_path,
        "--output", output_path
    ]
    
    print(f"\n🚀 启动工程化渲染模式")
    print(f"目标路径: {output_path}")
    
    try:
        # 建议直接在终端看到实时输出，不要 capture_output，除非需要日志审计
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"✅ 渲染成功！成品位于: {output_path}")
    except Exception as e:
        print(f"❌ 渲染执行异常: {e}")
        
if __name__ == "__main__":
    input_path  = r"C:\tk\迅雷云盘\原"
    lang_type = "JP"
    base_folder = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
    
    # 定义核心目录（物理锚点）
    work_dir = Path(base_folder) / "manga_translator_work"
    originals_dir = work_dir / "originals"
    json_dir = work_dir / "json"
    rendered_output_dir = Path(base_folder) / "rendered_results"
    run_mtu_command(input_path, lang_type)
    translate_pipeline(str(originals_dir), lang_type)
     # 翻译完所有的 txt 之后，开始修 JSON,解决超分后json识别框参数超过范围
    for json_file in json_dir.glob("*.json"):
        fix_and_scale_json(str(json_file))
    run_render_stage(input_path, output_path=str(rendered_output_dir))
# 调用测试


