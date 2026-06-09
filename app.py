"""
EDC (Extract, Define, Canonicalize) - Streamlit Web UI
"""
import streamlit as st
import os
import sys
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pdf_processor import extract_text_from_pdf


def extract_text_from_pdf_upload(uploaded_file) -> list:
    """
    アップロードされたPDFをページごとのテキストに変換（オンプレ既定）。

    PDF_PROCESSOR / PDF_BACKEND env に従い pdf_processor.extract_text_from_pdf へ委譲。
    Streamlitのアップロードファイルを一時 .pdf に書き出してパスで処理する。

    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト

    Returns:
        ページごとのテキストリスト
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

st.set_page_config(
    page_title="EDC - 知識トリプル抽出",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 EDC: Extract, Define, Canonicalize")
st.markdown("LLMベースの知識トリプル抽出フレームワーク")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ 設定")

    # Check if Azure is configured in .env
    azure_configured = bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT") and
        os.environ.get("AZURE_OPENAI_API_KEY")
    )

    # API Provider selection（完全ローカル運用では vLLM が既定）
    api_provider = st.selectbox(
        "API Provider",
        ["vLLM (ローカル)", "Azure OpenAI", "OpenAI"],
        index=0,
        help="使用するLLM APIを選択（既定: ローカルvLLM）"
    )

    if api_provider == "vLLM (ローカル)":
        st.subheader("vLLM 設定")
        st.success("✅ .env の VLLM_* 設定を使用（完全ローカル）")
        st.caption(f"LLM: {os.environ.get('VLLM_ENDPOINT', '(未設定)')} / {os.environ.get('VLLM_MODEL', '(未設定)')}")
        st.caption(f"Embedding: {os.environ.get('VLLM_EMBEDDING_ENDPOINT', '(未設定)')} / {os.environ.get('VLLM_EMBEDDING_MODEL', '(未設定)')}")
        # "vllm" を使うと llm_utils が .env の VLLM_* 設定を参照
        llm_model = "vllm"
        embedder_model = "vllm"

    elif api_provider == "Azure OpenAI":
        st.subheader("Azure OpenAI 設定")

        # Show status from .env
        if azure_configured:
            st.success("✅ .envから設定を読み込みました")

        azure_endpoint = st.text_input(
            "Azure Endpoint",
            value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            type="default",
            help="例: https://your-resource.openai.azure.com/"
        )
        azure_api_key = st.text_input(
            "Azure API Key",
            value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            type="password"
        )
        azure_api_version = st.text_input(
            "API Version",
            value=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        chat_deployment = st.text_input(
            "Chat Deployment Name",
            value=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4"),
            help="チャット用のデプロイメント名"
        )
        embedding_deployment = st.text_input(
            "Embedding Deployment Name",
            value=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
            help="埋め込み用のデプロイメント名"
        )

        # Set environment variables
        if azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        if azure_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
        if azure_api_version:
            os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version
        if chat_deployment:
            os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = chat_deployment
        if embedding_deployment:
            os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = embedding_deployment

        # Use "azure" to use .env settings
        llm_model = "azure"
        embedder_model = "azure"

    else:
        st.subheader("OpenAI 設定")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.environ.get("OPENAI_KEY", ""),
            type="password"
        )
        if openai_key:
            os.environ["OPENAI_KEY"] = openai_key

        llm_model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="使用するOpenAIモデル"
        )

        # Embedder selection for OpenAI
        st.subheader("Embedder 設定")
        embedder_option = st.selectbox(
            "Sentence Embedder",
            [
                "all-MiniLM-L6-v2 (軽量・推奨)",
                "all-mpnet-base-v2 (中量)",
                "intfloat/e5-mistral-7b-instruct (重い)"
            ]
        )

        embedder_map = {
            "all-MiniLM-L6-v2 (軽量・推奨)": "all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (中量)": "all-mpnet-base-v2",
            "intfloat/e5-mistral-7b-instruct (重い)": "intfloat/e5-mistral-7b-instruct"
        }
        embedder_model = embedder_map[embedder_option]

    st.divider()

    # PDF処理設定（完全ローカル: オンプレ PaddleX が既定）
    st.subheader("📄 PDF処理設定")
    pdf_processor = os.environ.get("PDF_PROCESSOR", "onprem")

    if pdf_processor == "onprem":
        st.success(f"✅ オンプレ処理: {os.environ.get('PDF_BACKEND', 'paddleocr_remote')}")
        st.caption(f"PaddleX: {os.environ.get('PADDLEX_ENDPOINT', '(未設定)')}")
    else:
        st.info(f"PDF_PROCESSOR={pdf_processor}")

    with st.expander("Azure Document Intelligence（ロールバック用）"):
        di_endpoint = st.text_input(
            "DI Endpoint",
            value=os.environ.get("AZURE_DI_ENDPOINT", ""),
            help="例: https://your-resource.cognitiveservices.azure.com/"
        )
        di_api_key = st.text_input(
            "DI API Key",
            value=os.environ.get("AZURE_DI_API_KEY", ""),
            type="password"
        )
        di_model = st.selectbox(
            "DI Model",
            ["prebuilt-layout", "prebuilt-read", "prebuilt-document"],
            index=0,
            help="prebuilt-layout推奨（テーブル・図対応）"
        )

        if di_endpoint:
            os.environ["AZURE_DI_ENDPOINT"] = di_endpoint
        if di_api_key:
            os.environ["AZURE_DI_API_KEY"] = di_api_key
        if di_model:
            os.environ["AZURE_DI_MODEL"] = di_model

    st.divider()

    # Advanced options
    with st.expander("詳細設定"):
        enrich_schema = st.checkbox(
            "スキーマ拡張",
            value=True,
            help="正規化できない関係を新しいスキーマとして追加（スキーマなしで使う場合は必須）"
        )
        refinement_iterations = st.number_input(
            "反復改善回数",
            min_value=0,
            max_value=5,
            value=0,
            help="Schema Retrieverによる反復改善（0=なし）"
        )

# Main content area
st.subheader("📁 ファイルアップロード")

uploaded_files = st.file_uploader(
    "テキスト/PDFファイル（複数選択可）",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    help="テキストファイル（1行1テキスト）またはPDFファイル（オンプレ PaddleX で処理）- 複数選択可"
)

# Schema options
st.subheader("ターゲットスキーマ（オプション）")
use_schema = st.checkbox("ターゲットスキーマを使用", value=False, help="スキーマなしで実行するとエッジを自動発見します")

uploaded_schema = None
if use_schema:
    uploaded_schema = st.file_uploader(
        "スキーマファイル（.csv）",
        type=["csv"],
        help="relation,definition形式のCSVファイル"
    )

# Run button
st.divider()

if st.button("🚀 トリプルを抽出", type="primary", use_container_width=True):
    # Validate configuration
    if api_provider == "vLLM (ローカル)":
        if not os.environ.get("VLLM_ENDPOINT"):
            st.error("VLLM_ENDPOINT を .env に設定してください")
            st.stop()
    elif api_provider == "Azure OpenAI":
        if not os.environ.get("AZURE_OPENAI_ENDPOINT") or not os.environ.get("AZURE_OPENAI_API_KEY"):
            st.error("Azure OpenAIのEndpointとAPI Keyを設定してください")
            st.stop()
    else:
        if not os.environ.get("OPENAI_KEY"):
            st.error("OpenAI API Keyを設定してください")
            st.stop()

    # Validate file upload
    if not uploaded_files:
        st.error("ファイルをアップロードしてください")
        st.stop()

    # Prepare input from all files
    input_texts = []
    file_boundaries = []  # [(start_idx, end_idx, filename), ...]

    with st.spinner("ファイルを処理中..."):
        for uploaded_file in uploaded_files:
            start_idx = len(input_texts)

            if uploaded_file.name.endswith('.pdf'):
                # PDF処理（オンプレ既定: PaddleX。PDF_PROCESSOR/PDF_BACKEND で切替）
                try:
                    texts = extract_text_from_pdf_upload(uploaded_file)
                    input_texts.extend(texts)
                    st.info(f"📄 {uploaded_file.name}: {len(texts)}ページを抽出")
                except Exception as e:
                    st.error(f"PDF処理エラー ({uploaded_file.name}): {str(e)}")
                    continue
            else:
                # テキストファイル処理
                texts = uploaded_file.read().decode("utf-8").strip().split("\n")
                input_texts.extend(texts)
                st.info(f"📄 {uploaded_file.name}: {len(texts)}行を読み込み")

            end_idx = len(input_texts)
            file_boundaries.append((start_idx, end_idx, uploaded_file.name))

    if not input_texts:
        st.error("処理可能なテキストがありません")
        st.stop()

    st.success(f"合計 {len(input_texts)} テキストを {len(uploaded_files)} ファイルから読み込みました")

    # Prepare schema
    schema_dict = {}
    if use_schema and uploaded_schema is not None:
        schema_content = uploaded_schema.read().decode("utf-8")
        for line in schema_content.strip().split("\n"):
            if "," in line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    schema_dict[parts[0].strip()] = parts[1].strip()

    # Run EDC
    with st.spinner("処理中... LLMを呼び出しています"):
        try:
            from edc.edc_framework import EDC
            import csv

            # Create temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write schema to temp file
                schema_path = os.path.join(tmpdir, "schema.csv")
                with open(schema_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for rel, defn in schema_dict.items():
                        writer.writerow([rel, defn])

                output_dir = os.path.join(tmpdir, "output")

                # EDC configuration
                edc_config = {
                    "oie_llm": llm_model,
                    "oie_prompt_template_file_path": "./prompt_templates/oie_template.txt",
                    "oie_few_shot_example_file_path": "./few_shot_examples/example/oie_few_shot_examples.txt",
                    "sd_llm": llm_model,
                    "sd_prompt_template_file_path": "./prompt_templates/sd_template.txt",
                    "sd_few_shot_example_file_path": "./few_shot_examples/example/sd_few_shot_examples.txt",
                    "sc_llm": llm_model,
                    "sc_embedder": embedder_model,
                    "sc_prompt_template_file_path": "./prompt_templates/sc_template.txt",
                    "sr_adapter_path": None,
                    "sr_embedder": embedder_model,
                    "oie_refine_prompt_template_file_path": "./prompt_templates/oie_r_template.txt",
                    "oie_refine_few_shot_example_file_path": "./few_shot_examples/example/oie_few_shot_refine_examples.txt",
                    "ee_llm": llm_model,
                    "ee_prompt_template_file_path": "./prompt_templates/ee_template.txt",
                    "ee_few_shot_example_file_path": "./few_shot_examples/example/ee_few_shot_examples.txt",
                    "em_prompt_template_file_path": "./prompt_templates/em_template.txt",
                    "target_schema_path": schema_path if schema_dict else None,
                    "enrich_schema": enrich_schema,
                    "loglevel": None,
                }

                # Change to edc directory for relative paths
                original_dir = os.getcwd()
                os.chdir(Path(__file__).parent)

                try:
                    edc = EDC(**edc_config)
                    results = edc.extract_kg(
                        input_texts,
                        output_dir,
                        refinement_iterations=refinement_iterations
                    )
                finally:
                    os.chdir(original_dir)

                # Display results
                st.success("抽出完了!")

                st.subheader("📊 抽出結果")

                # Display results grouped by file
                for start_idx, end_idx, filename in file_boundaries:
                    with st.expander(f"📄 {filename} ({end_idx - start_idx}テキスト)", expanded=True):
                        for idx in range(start_idx, end_idx):
                            text = input_texts[idx]
                            triplets = results[idx]
                            st.markdown(f"**テキスト {idx - start_idx + 1}:** {text[:100]}...")

                            if triplets:
                                data = []
                                for t in triplets:
                                    if t is not None and len(t) == 3:
                                        data.append({
                                            "Subject": t[0],
                                            "Relation": t[1],
                                            "Object": t[2]
                                        })
                                if data:
                                    st.table(data)
                                else:
                                    st.caption("正規化されたトリプルなし")
                            else:
                                st.caption("トリプルなし")
                            st.divider()

                # Edge summary (aggregate unique relations)
                st.subheader("📈 発見されたエッジ（スキーマ候補）")
                edge_summary = {}
                for triplets in results:
                    for t in triplets:
                        if t and len(t) == 3:
                            rel = t[1]
                            if rel not in edge_summary:
                                edge_summary[rel] = {"count": 0, "definition": "", "examples": []}
                            edge_summary[rel]["count"] += 1
                            if len(edge_summary[rel]["examples"]) < 3:
                                edge_summary[rel]["examples"].append((t[0], t[2]))

                # Get definitions from EDC schema
                for rel in edge_summary:
                    if rel in edc.schema:
                        edge_summary[rel]["definition"] = edc.schema[rel]

                if edge_summary:
                    edge_data = []
                    for rel, info in sorted(edge_summary.items(), key=lambda x: -x[1]["count"]):
                        examples_str = ", ".join([f"{s}→{o}" for s, o in info["examples"][:2]])
                        definition = info["definition"]
                        if len(definition) > 50:
                            definition = definition[:50] + "..."
                        edge_data.append({
                            "リレーション": rel,
                            "定義": definition,
                            "出現回数": info["count"],
                            "例": examples_str
                        })
                    st.table(edge_data)
                else:
                    st.info("エッジは抽出されませんでした")

                # Export section
                st.subheader("📥 エクスポート")

                # JSON export with file grouping
                export_data = []
                for start_idx, end_idx, filename in file_boundaries:
                    file_data = {
                        "file": filename,
                        "texts": []
                    }
                    for idx in range(start_idx, end_idx):
                        file_data["texts"].append({
                            "input_text": input_texts[idx],
                            "triplets": [t for t in results[idx] if t is not None]
                        })
                    export_data.append(file_data)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="JSONとしてダウンロード",
                        data=json.dumps(export_data, indent=2, ensure_ascii=False),
                        file_name="triplets.json",
                        mime="application/json"
                    )

                # Schema CSV export
                with col2:
                    csv_lines = ["relation,definition,count"]
                    for rel, info in sorted(edge_summary.items(), key=lambda x: -x[1]["count"]):
                        definition = info["definition"].replace('"', '""')
                        csv_lines.append(f'"{rel}","{definition}",{info["count"]}')
                    csv_content = "\n".join(csv_lines)

                    st.download_button(
                        label="スキーマCSVとしてダウンロード",
                        data=csv_content,
                        file_name="discovered_schema.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown("""
---
**EDC Framework** - [GitHub](https://github.com/clear-nus/edc) |
論文: [Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction](https://arxiv.org/abs/2404.03868)
""")
