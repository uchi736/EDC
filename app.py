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


def extract_text_from_pdf_azure_di(uploaded_file) -> list:
    """
    Azure Document Intelligence を使用してPDFからテキストをページごとに抽出

    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト

    Returns:
        ページごとのテキストリスト
    """
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    import re

    endpoint = os.environ.get("AZURE_DI_ENDPOINT")
    api_key = os.environ.get("AZURE_DI_API_KEY")
    model = os.environ.get("AZURE_DI_MODEL", "prebuilt-layout")

    if not endpoint or not api_key:
        raise ValueError("Azure Document Intelligence の設定が必要です（AZURE_DI_ENDPOINT, AZURE_DI_API_KEY）")

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )

    file_content = uploaded_file.read()
    poller = client.begin_analyze_document(
        model,
        AnalyzeDocumentRequest(bytes_source=file_content),
    )
    result = poller.result()

    # ページごとにテキストを抽出
    texts = []
    if result.pages:
        for page in result.pages:
            page_num = page.page_number
            # ページ範囲内のコンテンツを抽出
            page_content = []
            if result.paragraphs:
                for para in result.paragraphs:
                    if hasattr(para, 'bounding_regions') and para.bounding_regions:
                        for region in para.bounding_regions:
                            if region.page_number == page_num:
                                page_content.append(para.content)
                                break

            if page_content:
                # テキストをクリーンアップ
                text = "\n".join(page_content)
                # 日本語の不要なスペースを削除
                text = re.sub(r'[ ]+([ぁ-んァ-ヴー一-龠々〆〤])', r'\1', text)
                text = re.sub(r'([ぁ-んァ-ヴー一-龠々〆〤])[ ]+', r'\1', text)
                texts.append(text.strip())

    # ページごとの抽出が空の場合、全体コンテンツを使用
    if not texts and result.content:
        texts = [result.content]

    return texts


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # API Provider selection
    api_provider = st.selectbox(
        "API Provider",
        ["Azure OpenAI", "OpenAI"],
        index=0 if azure_configured else 1,
        help="使用するLLM APIを選択"
    )

    if api_provider == "Azure OpenAI":
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

    # Azure Document Intelligence settings
    st.subheader("📄 PDF処理設定")
    azure_di_configured = bool(
        os.environ.get("AZURE_DI_ENDPOINT") and
        os.environ.get("AZURE_DI_API_KEY")
    )

    if azure_di_configured:
        st.success("✅ Azure DI設定済み")
    else:
        st.warning("⚠️ Azure DI未設定（PDFアップロード不可）")

    with st.expander("Azure Document Intelligence"):
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
tab1, tab2, tab3 = st.tabs(["📝 テキスト入力", "📁 ファイルアップロード", "📊 スキーマ管理"])

with tab1:
    st.subheader("テキストから知識トリプルを抽出")

    input_text = st.text_area(
        "入力テキスト",
        value="John Doe is a student at National University of Singapore.",
        height=150,
        help="知識トリプルを抽出したいテキストを入力"
    )

    # Schema input
    st.subheader("ターゲットスキーマ（オプション）")
    use_schema = st.checkbox("ターゲットスキーマを使用", value=True)

    if use_schema:
        default_schema = """student,The subject receives education at the institute specified by the object entity.
country,The subject entity is located in the country specified by the object entity.
place of birth,The subject entity was born in the location specified by the object entity.
date of birth,The subject entity was born on the date specified by the object entity.
occupation,The subject entity has the occupation specified by the object entity."""

        schema_text = st.text_area(
            "スキーマ定義（CSV形式: relation,definition）",
            value=default_schema,
            height=150
        )

with tab2:
    st.subheader("ファイルからテキストを読み込み")

    uploaded_file = st.file_uploader(
        "テキスト/PDFファイル",
        type=["txt", "pdf"],
        help="テキストファイル（1行1テキスト）またはPDFファイル（Azure DIで処理）"
    )

    uploaded_schema = st.file_uploader(
        "スキーマファイル（.csv）",
        type=["csv"],
        help="relation,definition形式のCSVファイル"
    )

with tab3:
    st.subheader("スキーマ管理")

    # Load existing schemas
    schema_dir = Path(__file__).parent / "schemas"
    if schema_dir.exists():
        schema_files = list(schema_dir.glob("*.csv"))
        if schema_files:
            selected_schema = st.selectbox(
                "既存のスキーマを選択",
                [f.stem for f in schema_files]
            )

            if st.button("スキーマを読み込み"):
                schema_path = schema_dir / f"{selected_schema}.csv"
                with open(schema_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="csv")

# Run button
st.divider()

if st.button("🚀 トリプルを抽出", type="primary", use_container_width=True):
    # Validate configuration
    if api_provider == "Azure OpenAI":
        if not os.environ.get("AZURE_OPENAI_ENDPOINT") or not os.environ.get("AZURE_OPENAI_API_KEY"):
            st.error("Azure OpenAIのEndpointとAPI Keyを設定してください")
            st.stop()
    else:
        if not os.environ.get("OPENAI_KEY"):
            st.error("OpenAI API Keyを設定してください")
            st.stop()

    # Prepare input
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            # PDF処理（Azure Document Intelligence）
            with st.spinner("PDFを処理中... Azure Document Intelligence"):
                try:
                    input_texts = extract_text_from_pdf_azure_di(uploaded_file)
                    st.info(f"📄 {len(input_texts)}ページを抽出しました")
                except Exception as e:
                    st.error(f"PDF処理エラー: {str(e)}")
                    st.stop()
        else:
            # テキストファイル処理
            input_texts = uploaded_file.read().decode("utf-8").strip().split("\n")
    else:
        input_texts = [input_text.strip()]

    # Prepare schema
    schema_dict = {}
    if use_schema:
        if uploaded_schema is not None:
            schema_content = uploaded_schema.read().decode("utf-8")
        else:
            schema_content = schema_text

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

                for idx, (text, triplets) in enumerate(zip(input_texts, results)):
                    with st.expander(f"テキスト {idx + 1}: {text[:50]}...", expanded=True):
                        st.markdown(f"**入力:** {text}")
                        st.markdown("**抽出されたトリプル:**")

                        if triplets:
                            # Create table
                            data = []
                            for t in triplets:
                                if t is not None and len(t) == 3:
                                    data.append({
                                        "Subject (主語)": t[0],
                                        "Relation (関係)": t[1],
                                        "Object (目的語)": t[2]
                                    })

                            if data:
                                st.table(data)
                            else:
                                st.info("正規化されたトリプルはありません")
                        else:
                            st.info("トリプルは抽出されませんでした")

                # JSON export
                st.subheader("📥 エクスポート")
                export_data = []
                for idx, (text, triplets) in enumerate(zip(input_texts, results)):
                    export_data.append({
                        "input_text": text,
                        "triplets": [t for t in triplets if t is not None]
                    })

                st.download_button(
                    label="JSONとしてダウンロード",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name="triplets.json",
                    mime="application/json"
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
