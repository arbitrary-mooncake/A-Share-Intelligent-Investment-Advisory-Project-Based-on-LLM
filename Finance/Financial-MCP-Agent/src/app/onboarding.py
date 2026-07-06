"""
导览页面（Onboarding）：首次运行引导用户选择版本并配置 API Key。

流程：
1. 展示版本对比表
2. 用户选择 Lite 或 Full
3. 根据选择展示配置向导
4. 写入 .env 文件
5. 进入系统
"""
import os
import streamlit as st


def _get_env_path() -> str:
    """获取 .env 文件路径"""
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", ".env"
    ))


def check_first_run() -> bool:
    """检查是否为首次运行（.env 不存在或仍为模板状态）"""
    env_path = _get_env_path()
    if not os.path.exists(env_path):
        return True
    with open(env_path, encoding="utf-8") as f:
        content = f.read()
    return "your_" in content or "your_api_key" in content


def show_onboarding():
    """展示导览页面"""
    st.markdown("""
    <style>
    .onboarding-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transition: all 0.3s;
        height: 100%;
    }
    .onboarding-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 24px rgba(59,130,246,0.12);
    }
    .onboarding-card.lite { border-left: 6px solid #3b82f6; }
    .onboarding-card.full { border-left: 6px solid #d97706; }
    .onboarding-title {
        font-size: 1.6em;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .onboarding-subtitle {
        font-size: 0.95em;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5em; font-weight: 800; color: #1e40af;">
            🎉 欢迎使用 A 股智能投顾 Agent 助手
        </h1>
        <p style="font-size: 1.1em; color: #64748b; margin-top: 0.5rem;">
            在开始之前，请选择您的使用模式
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="onboarding-card lite">
            <div class="onboarding-title">⚡ 精简版 (Lite)</div>
            <div class="onboarding-subtitle">零成本开始使用，适合学习体验</div>
            <ul style="color: #475569; line-height: 2;">
                <li>✅ 仅需 1 个 DeepSeek API Key</li>
                <li>✅ 免费 Tushare (120 积分) + AKShare</li>
                <li>✅ 5/7 个功能页面开放</li>
                <li>⚠️ 单模型分析，深度略低</li>
                <li>🔒 「模拟分析与迭代」不可用</li>
                <li>🔒 「智能投顾」不可用</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("选择精简版", key="choose_lite", use_container_width=True, type="primary"):
            st.session_state["chosen_mode"] = "lite"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="onboarding-card full">
            <div class="onboarding-title">🚀 完整版 (Full)</div>
            <div class="onboarding-subtitle">全部功能，最高分析质量</div>
            <ul style="color: #475569; line-height: 2;">
                <li>✅ 6 个专用 LLM 模型</li>
                <li>✅ Tushare 5000+ 积分（付费会员）</li>
                <li>✅ 全部 7 个功能页面开放</li>
                <li>✅ 专用模型（MiMo 做基本面，Qwen 做技术面）</li>
                <li>✅ 模拟分析与迭代</li>
                <li>✅ 智能投顾</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("选择完整版", key="choose_full", use_container_width=True, type="primary"):
            st.session_state["chosen_mode"] = "full"
            st.rerun()

    st.divider()
    if st.button("跳过导览，手动配置 .env", key="skip_onboarding"):
        st.session_state["skip_onboarding"] = True
        st.rerun()

    chosen = st.session_state.get("chosen_mode")
    if chosen == "lite":
        _show_lite_config()
    elif chosen == "full":
        _show_full_config()


def _show_lite_config():
    """精简版配置向导"""
    st.markdown("### ⚡ 精简版配置向导")

    with st.form("lite_config_form"):
        st.markdown("**Step 1/2: 配置 DeepSeek API Key**")
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="获取地址: https://platform.deepseek.com/",
            key="lite_deepseek_key",
        )

        st.markdown("**Step 2/2: 配置 Tushare Token（可选）**")
        tushare_token = st.text_input(
            "Tushare Token",
            type="password",
            help="注册地址: https://tushare.pro/，免费账号即可（120 积分）",
            key="lite_tushare_token",
        )

        if st.form_submit_button("完成配置 →", use_container_width=True, type="primary"):
            if not deepseek_key:
                st.error("请填写 DeepSeek API Key")
            else:
                _write_env_lite(deepseek_key, tushare_token)
                st.success("✅ 配置已保存！")
                st.session_state["onboarding_done"] = True
                st.rerun()


def _show_full_config():
    """完整版配置向导"""
    st.markdown("### 🚀 完整版配置向导")

    with st.form("full_config_form"):
        st.markdown("#### LLM API Keys")

        m1_key = st.text_input("M1 (MiMo-V2.5-Pro) API Key", type="password", key="full_m1_key")
        m1_url = st.text_input("M1 Base URL", value="https://api.xiaomimimo.com/v1", key="full_m1_url")
        m1_model = st.text_input("M1 Model Name", value="mimo-v2.5-pro", key="full_m1_model")

        m2_key = st.text_input("M2 (Qwen3.6-Flash) API Key", type="password", key="full_m2_key")
        m2_url = st.text_input("M2 Base URL", value="https://dashscope.aliyuncs.com/compatible-mode/v1", key="full_m2_url")
        m2_model = st.text_input("M2 Model Name", value="qwen3.6-flash", key="full_m2_model")

        m3_key = st.text_input("M3 (Qwen3.7-Plus) API Key", type="password", key="full_m3_key")
        m3_url = st.text_input("M3 Base URL", value="https://dashscope.aliyuncs.com/compatible-mode/v1", key="full_m3_url")
        m3_model = st.text_input("M3 Model Name", value="qwen3.7-plus", key="full_m3_model")

        m5_key = st.text_input("M5 (MiMo-V2.5) API Key", type="password", key="full_m5_key")
        m5_url = st.text_input("M5 Base URL", value="https://api.xiaomimimo.com/v1", key="full_m5_url")
        m5_model = st.text_input("M5 Model Name", value="mimo-v2.5", key="full_m5_model")

        m6_key = st.text_input("M6 (DeepSeek V4 Pro) API Key", type="password", key="full_m6_key")
        m6_url = st.text_input("M6 Base URL", value="https://api.deepseek.com/v1", key="full_m6_url")
        m6_model = st.text_input("M6 Model Name", value="deepseek-chat", key="full_m6_model")

        st.markdown("#### Tushare")
        tushare_token = st.text_input("Tushare Token（需要 5000+ 积分）", type="password", key="full_tushare_token")

        if st.form_submit_button("完成配置 →", use_container_width=True, type="primary"):
            _write_env_full(m1_key, m1_url, m1_model, m2_key, m2_url, m2_model,
                           m3_key, m3_url, m3_model, m5_key, m5_url, m5_model,
                           m6_key, m6_url, m6_model, tushare_token)
            st.success("✅ 配置已保存！")
            st.session_state["onboarding_done"] = True
            st.rerun()


def _write_env_lite(deepseek_key: str, tushare_token: str):
    """写入精简版 .env 并立即更新 os.environ"""
    env_path = _get_env_path()
    content = f"""# ============================================================
# A股智能投顾Agent助手 — 环境变量配置（精简版）
# ============================================================
APP_MODE=lite

# ─── DeepSeek API（唯一需要的 LLM Key）────────────────────
DEEPSEEK_API_KEY={deepseek_key}
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL_PRO=deepseek-chat
DEEPSEEK_MODEL_FLASH=deepseek-chat

# ─── Tushare API（免费 120 积分）──────────────────────────
TUSHARE_TOKEN={tushare_token or ''}
TUSHARE_URL=https://api.tushare.pro

# ─── LLM 模式设置 ────────────────────────────────────────
USE_LOCAL_MODEL=api

# ─── 完整版配置（切换到 full 模式时需要填写）────────────────
# OPENAI_COMPATIBLE_API_KEY=
# OPENAI_COMPATIBLE_BASE_URL=
# OPENAI_COMPATIBLE_MODEL=
# OPENAI_COMPATIBLE_API_KEY_2=
# OPENAI_COMPATIBLE_BASE_URL_2=
# OPENAI_COMPATIBLE_MODEL_2=
# OPENAI_COMPATIBLE_API_KEY_3=
# OPENAI_COMPATIBLE_BASE_URL_3=
# OPENAI_COMPATIBLE_MODEL_3=
# OPENAI_COMPATIBLE_API_KEY_5=
# OPENAI_COMPATIBLE_BASE_URL_5=
# OPENAI_COMPATIBLE_MODEL_5=
# OPENAI_COMPATIBLE_API_KEY_6=
# OPENAI_COMPATIBLE_BASE_URL_6=
# OPENAI_COMPATIBLE_MODEL_6=
"""
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)

    os.environ["APP_MODE"] = "lite"
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"
    os.environ["DEEPSEEK_MODEL_PRO"] = "deepseek-chat"
    os.environ["DEEPSEEK_MODEL_FLASH"] = "deepseek-chat"
    if tushare_token:
        os.environ["TUSHARE_TOKEN"] = tushare_token
    os.environ["TUSHARE_URL"] = "https://api.tushare.pro"
    os.environ["USE_LOCAL_MODEL"] = "api"


def _write_env_full(m1_key, m1_url, m1_model, m2_key, m2_url, m2_model,
                    m3_key, m3_url, m3_model, m5_key, m5_url, m5_model,
                    m6_key, m6_url, m6_model, tushare_token):
    """写入完整版 .env"""
    env_path = _get_env_path()
    content = f"""# ============================================================
# A股智能投顾Agent助手 — 环境变量配置（完整版）
# ============================================================
APP_MODE=full

# ─── Model 1: MiMo-V2.5-Pro ──────────────────────────────
OPENAI_COMPATIBLE_API_KEY={m1_key}
OPENAI_COMPATIBLE_BASE_URL={m1_url}
OPENAI_COMPATIBLE_MODEL={m1_model}

# ─── Model 2: Qwen3.6-Flash ──────────────────────────────
OPENAI_COMPATIBLE_API_KEY_2={m2_key}
OPENAI_COMPATIBLE_BASE_URL_2={m2_url}
OPENAI_COMPATIBLE_MODEL_2={m2_model}

# ─── Model 3: Qwen3.7-Plus ───────────────────────────────
OPENAI_COMPATIBLE_API_KEY_3={m3_key}
OPENAI_COMPATIBLE_BASE_URL_3={m3_url}
OPENAI_COMPATIBLE_MODEL_3={m3_model}

# ─── Model 4: 保留位 ──────────────────────────────────────
OPENAI_COMPATIBLE_API_KEY_4=
OPENAI_COMPATIBLE_BASE_URL_4=
OPENAI_COMPATIBLE_MODEL_4=

# ─── Model 5: MiMo-V2.5 ──────────────────────────────────
OPENAI_COMPATIBLE_API_KEY_5={m5_key}
OPENAI_COMPATIBLE_BASE_URL_5={m5_url}
OPENAI_COMPATIBLE_MODEL_5={m5_model}

# ─── Model 6: DeepSeek V4 Pro ────────────────────────────
OPENAI_COMPATIBLE_API_KEY_6={m6_key}
OPENAI_COMPATIBLE_BASE_URL_6={m6_url}
OPENAI_COMPATIBLE_MODEL_6={m6_model}

# ─── Tushare API（需要 5000+ 积分）────────────────────────
TUSHARE_TOKEN={tushare_token or ''}
TUSHARE_URL=https://api.tushare.pro

# ─── LLM 模式设置 ────────────────────────────────────────
USE_LOCAL_MODEL=api

# ─── DeepSeek API（Lite 模式使用）─────────────────────────
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL_PRO=deepseek-chat
DEEPSEEK_MODEL_FLASH=deepseek-chat
"""
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)

    os.environ["APP_MODE"] = "full"
    os.environ["OPENAI_COMPATIBLE_API_KEY"] = m1_key or ""
    os.environ["OPENAI_COMPATIBLE_BASE_URL"] = m1_url
    os.environ["OPENAI_COMPATIBLE_MODEL"] = m1_model
    os.environ["OPENAI_COMPATIBLE_API_KEY_2"] = m2_key or ""
    os.environ["OPENAI_COMPATIBLE_BASE_URL_2"] = m2_url
    os.environ["OPENAI_COMPATIBLE_MODEL_2"] = m2_model
    os.environ["OPENAI_COMPATIBLE_API_KEY_3"] = m3_key or ""
    os.environ["OPENAI_COMPATIBLE_BASE_URL_3"] = m3_url
    os.environ["OPENAI_COMPATIBLE_MODEL_3"] = m3_model
    os.environ["OPENAI_COMPATIBLE_API_KEY_5"] = m5_key or ""
    os.environ["OPENAI_COMPATIBLE_BASE_URL_5"] = m5_url
    os.environ["OPENAI_COMPATIBLE_MODEL_5"] = m5_model
    os.environ["OPENAI_COMPATIBLE_API_KEY_6"] = m6_key or ""
    os.environ["OPENAI_COMPATIBLE_BASE_URL_6"] = m6_url
    os.environ["OPENAI_COMPATIBLE_MODEL_6"] = m6_model
    if tushare_token:
        os.environ["TUSHARE_TOKEN"] = tushare_token
    os.environ["TUSHARE_URL"] = "https://api.tushare.pro"
    os.environ["USE_LOCAL_MODEL"] = "api"
