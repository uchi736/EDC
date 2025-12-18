"""
Simple test script to run EDC with Azure OpenAI
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).parent / ".env")

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check Azure config
print("=== Azure OpenAI Configuration ===")
print(f"Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT', 'NOT SET')}")
print(f"API Version: {os.environ.get('AZURE_OPENAI_API_VERSION', 'NOT SET')}")
print(f"Chat Deployment: {os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME', 'NOT SET')}")
print(f"Embedding Deployment: {os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', 'NOT SET')}")
print(f"API Key: {'SET' if os.environ.get('AZURE_OPENAI_API_KEY') else 'NOT SET'}")
print()

# Test simple API call first
print("=== Testing Azure OpenAI Connection ===")
from edc.utils.llm_utils import get_azure_client, get_azure_chat_deployment_name

try:
    client = get_azure_client()
    deployment = get_azure_chat_deployment_name()
    print(f"Using deployment: {deployment}")

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Say 'Hello' in Japanese"}],
        max_tokens=10
    )
    print(f"API Response: {response.choices[0].message.content}")
    print("API connection successful!")
except Exception as e:
    print(f"API Error: {e}")
    sys.exit(1)

print()
print("=== Running EDC Sample ===")

# Run EDC
from edc.edc_framework import EDC
import tempfile

# Sample input
input_texts = ["John Doe is a student at National University of Singapore."]

# Sample schema
schema_content = """student,The subject receives education at the institute specified by the object entity.
country,The subject entity is located in the country specified by the object entity.
place of birth,The subject entity was born in the location specified by the object entity."""

# Create temp schema file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
    f.write(schema_content)
    schema_path = f.name

# Create temp output dir
output_dir = tempfile.mkdtemp()

# EDC config - use "azure" for both LLM and embedder
edc_config = {
    "oie_llm": "azure",
    "oie_prompt_template_file_path": "./prompt_templates/oie_template.txt",
    "oie_few_shot_example_file_path": "./few_shot_examples/example/oie_few_shot_examples.txt",
    "sd_llm": "azure",
    "sd_prompt_template_file_path": "./prompt_templates/sd_template.txt",
    "sd_few_shot_example_file_path": "./few_shot_examples/example/sd_few_shot_examples.txt",
    "sc_llm": "azure",
    "sc_embedder": "azure",
    "sc_prompt_template_file_path": "./prompt_templates/sc_template.txt",
    "sr_adapter_path": None,
    "sr_embedder": "azure",
    "oie_refine_prompt_template_file_path": "./prompt_templates/oie_r_template.txt",
    "oie_refine_few_shot_example_file_path": "./few_shot_examples/example/oie_few_shot_refine_examples.txt",
    "ee_llm": "azure",
    "ee_prompt_template_file_path": "./prompt_templates/ee_template.txt",
    "ee_few_shot_example_file_path": "./few_shot_examples/example/ee_few_shot_examples.txt",
    "em_prompt_template_file_path": "./prompt_templates/em_template.txt",
    "target_schema_path": schema_path,
    "enrich_schema": False,
    "loglevel": None,
}

# Change to edc directory
os.chdir(Path(__file__).parent)

print(f"Input: {input_texts[0]}")
print()

try:
    edc = EDC(**edc_config)
    results = edc.extract_kg(input_texts, output_dir, refinement_iterations=0)

    print("=== Results ===")
    for idx, (text, triplets) in enumerate(zip(input_texts, results)):
        print(f"Input: {text}")
        print(f"Extracted triplets:")
        for t in triplets:
            if t is not None:
                print(f"  - {t}")
        if not any(t is not None for t in triplets):
            print("  (No triplets extracted)")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
finally:
    # Cleanup
    import shutil
    os.unlink(schema_path)
    shutil.rmtree(output_dir, ignore_errors=True)

print()
print("=== Done ===")
