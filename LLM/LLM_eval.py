import os
import gc
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import spacy

from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util
from summac.model_summac import SummaCZS

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

warnings.filterwarnings("ignore")

print("=" * 80)
print("CHQ EVALUATION - 30 SAMPLES PER CONFIGURATION")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load metric models
print("Loading metric models...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
nlp = spacy.load("en_core_web_sm")
summac_model = SummaCZS(granularity="sentence", model_name="vitc", device=DEVICE)

# NLI model for entailment
nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(DEVICE).eval()

print("Metrics loaded\n")


class CHQDataset:
    """Dataset loader for CHQ-Summ XML format"""

    def __init__(self, xml_path):
        self.data = []
        self.load_data(xml_path)

    def load_data(self, xml_path):
        print(f"Loading: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for doc in root.findall(".//document"):
            try:
                human_summary = doc.find(".//test_metadata/human_summary")
                subject = doc.find(".//original_corpus_data/subject")

                if (
                    human_summary is None
                    or not human_summary.text
                    or subject is None
                    or not subject.text
                ):
                    continue

                question = subject.text.strip()
                content = doc.find(".//original_corpus_data/content")
                if content is not None and content.text:
                    question += " " + content.text.strip()

                self.data.append(
                    {"question": question, "gold_summary": human_summary.text.strip()}
                )
            except Exception:
                continue

        print(f"Loaded {len(self.data)} pairs\n")


class MetricsCalculator:
    """Calculate evaluation metrics"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def validate_text(self, text):
        return text and isinstance(text, str) and len(text.strip()) >= 3

    def calculate_rouge_l(self, pred, ref):
        if not self.validate_text(pred) or not self.validate_text(ref):
            return 0.0
        try:
            return self.rouge_scorer.score(ref, pred)["rougeL"].fmeasure
        except Exception:
            return 0.0

    def calculate_bertscore(self, preds, refs):
        if preds is None or refs is None:
            return {"f1": 0.0}

        n = min(len(preds), len(refs))
        preds, refs = preds[:n], refs[:n]

        valid_preds = [p if self.validate_text(p) else "" for p in preds]
        valid_refs = [r if self.validate_text(r) else "" for r in refs]

        try:
            _, _, F1 = bert_score_fn(
                valid_preds,
                valid_refs,
                lang="en",
                verbose=False,
                device=DEVICE,
            )
            return {"f1": max(float(F1.mean().item()), 0.0)}
        except Exception:
            return {"f1": 0.0}

    def calculate_semantic_coherence(self, pred, question):
        """Topic alignment between original question and generated summary (pred vs question)."""
        if not self.validate_text(pred) or not self.validate_text(question):
            return 0.0
        try:
            e1 = sentence_model.encode(pred, convert_to_tensor=True)
            e2 = sentence_model.encode(question[:512], convert_to_tensor=True)
            return max(float(util.cos_sim(e1, e2).item()), 0.0)
        except Exception:
            return 0.0

    def calculate_summac(self, pred, source):
        """Factual consistency proxy: summary vs source (here: question)."""
        if not self.validate_text(pred) or not self.validate_text(source):
            return 0.0
        try:
            s = summac_model.score([source], [pred])["scores"][0]
            return float(s) if s > -0.5 else 0.0
        except Exception:
            return 0.0

    def extract_entities(self, text):
        if not self.validate_text(text):
            return set()
        try:
            return set([e.text.lower() for e in nlp(text[:1000]).ents])
        except Exception:
            return set()

    def calculate_entity_preservation(self, pred, question):
        if not self.validate_text(pred) or not self.validate_text(question):
            return 0.0
        q_ents = self.extract_entities(question)
        p_ents = self.extract_entities(pred)
        return 1.0 if len(q_ents) == 0 else float(len(q_ents.intersection(p_ents)) / len(q_ents))

    def calculate_qe_overlap(self, pred, question):
        """Token overlap allowing alphanumerics (B12, HbA1c, COVID-19)."""
        if not self.validate_text(pred) or not self.validate_text(question):
            return 0.0

        def tokenize(t):
            return set(
                w.lower()
                for w in re.findall(r"[A-Za-z0-9\-]+", t)
                if len(w) >= 3
            )

        p_words = tokenize(pred)
        q_words = tokenize(question)
        return 0.0 if len(q_words) == 0 else float(len(p_words & q_words) / len(q_words))

    def calculate_entailment(self, pred, question):
        """NLI entailment probability: question entails summary."""
        if not self.validate_text(pred) or not self.validate_text(question):
            return 0.0
        try:
            inputs = nli_tokenizer(
                question[:512],
                pred[:512],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                logits = nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            # MNLI label order: contradiction, neutral, entailment
            return float(probs[0, 2].item())
        except Exception:
            return 0.0


class LLMSummarizer:
    """LLM-based summarization with multiple prompting strategies"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_registry = {
            "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
            "gemma-7b": "google/gemma-7b-it",
            "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat",
        }
        self.load_model()

    def load_model(self):
        self.hf_name = self.model_registry[self.model_name]

        print("=" * 80)
        print(f"Loading: {self.model_name} -> {self.hf_name}")
        print("=" * 80)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name, trust_remote_code=True)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # (kept from your code)
        if "gemma" in self.model_name.lower() and hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
        )

        self.model.eval()
        if torch.cuda.is_available():
            print(f"Loaded (GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB)\n")
        else:
            print("Loaded on CPU\n")

    def create_prompt(self, question, examples, strategy="standard"):
        instructions = {
            "standard": "Rewrite the question as ONE short medical question (10-15 words).\n",
            "chain-of-density": (
                "Step 1: List 3-8 key medical terms.\n"
                "Step 2: Write ONE short medical question (10-15 words) using those terms.\n"
            ),
            "hierarchical": (
                "Step 1: Identify the MAIN medical topic in 3-6 words.\n"
                "Step 2: Write ONE short medical question (10-15 words) about that topic.\n"
            ),
            "element-aware": (
                "Step 1: Extract important medical terms (diseases, drugs, tests).\n"
                "Step 2: Write ONE short medical question (10-15 words) that MUST include them.\n"
            ),
        }

        prompt = instructions[strategy]
        prompt += "Respond in the following format ONLY:\nFINAL_QUESTION: <question>\n\n"

        if examples:
            for ex in examples[:5]:
                q = ex["question"][:180]
                s = ex["summary"]
                prompt += f"Example:\nQuestion: {q}\nFINAL_QUESTION: {s}\n\n"

        prompt += f"Question: {question[:250]}\nFINAL_QUESTION:"
        return prompt

    def extract_summary(self, full_text):
        """Extract FINAL_QUESTION safely and deterministically."""
        if "FINAL_QUESTION:" not in full_text:
            return "What is the medical condition?"

        summary = full_text.split("FINAL_QUESTION:")[-1].strip()
        summary = summary.split("\n")[0].strip()

        words = summary.split()
        if 5 <= len(words) <= 20:
            if not summary.endswith("?"):
                summary += "?"
            return summary

        return "What is the medical condition?"

    # ----------------------------
    # FIXED: use chat templates
    # ----------------------------
    def generate_summary(self, question, examples=None, strategy="standard"):
        prompt_text = self.create_prompt(question, examples, strategy)

        # We treat your entire prompt as a user message
        messages = [{"role": "user", "content": prompt_text}]

        # Prefer chat template when available (crucial for Llama/Mistral/Qwen)
        try:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)
            else:
                raise RuntimeError("No chat_template available")
        except Exception:
            # Fallback: raw prompt
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=False,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

        # Keep your generation style deterministic
        max_new = 35 if "gemma" in self.model_name.lower() else 32
        rep = 1.03 if "gemma" in self.model_name.lower() else 1.02

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                min_new_tokens=7,
                do_sample=False,
                num_beams=1,
                repetition_penalty=rep,
                no_repeat_ngram_size=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode ONLY new tokens (not the prompt)
        gen_ids = outputs[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self.extract_summary(text)

    def cleanup(self):
        print(f"Cleaning {self.model_name}...")
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_evaluation():
    dataset_path = "merged_l6_found_only.xml"  # change if needed
    output_dir = "results_30samples"
    test_size = 150
    samples_per_config = 30

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Place merged_l6_found_only.xml in the same directory as this script")
        return

    os.makedirs(output_dir, exist_ok=True)
    dataset = CHQDataset(dataset_path)

    if len(dataset.data) == 0:
        print("No data loaded from dataset")
        return

    # SAFETY: don't exceed dataset size
    test_size = min(test_size, len(dataset.data))

    test_data = dataset.data[:test_size]
    train_data = (
        dataset.data[test_size : test_size + 10]
        if len(dataset.data) > test_size
        else dataset.data[:10]
    )

    print(f"Test: {len(test_data)} | Train: {len(train_data)}")
    print(f"Excel: {samples_per_config} samples per configuration")
    print("Total configs: 6 models x 4 strategies x 3 shots = 72")
    print(f"Total sample rows: 72 x {samples_per_config} = {72 * samples_per_config}\n")

    metrics_calc = MetricsCalculator()
    models = ["qwen2-7b", "mistral-7b", "llama3.1-8b", "llama3.2-3b", "gemma-7b", "deepseek-7b"]
    shots = [0, 2, 5]
    strategies = ["standard", "chain-of-density", "hierarchical", "element-aware"]

    all_results = []
    all_predictions = []
    total = len(models) * len(strategies) * len(shots)
    current = 0

    for model_name in models:
        try:
            summarizer = LLMSummarizer(model_name)

            for strategy in strategies:
                for n_shot in shots:
                    current += 1
                    config_name = f"{model_name}_{strategy}_{n_shot}shot"

                    print("=" * 80)
                    print(f"[{current}/{total}] {config_name}")
                    print("=" * 80)

                    examples = None
                    if n_shot > 0:
                        examples = [
                            {"question": ex["question"], "summary": ex["gold_summary"]}
                            for ex in train_data[:n_shot]
                        ]

                    preds, refs, questions = [], [], []
                    for item in tqdm(test_data, desc="Generating"):
                        try:
                            summary = summarizer.generate_summary(item["question"], examples, strategy)
                            preds.append(summary)
                        except Exception:
                            preds.append("What is the medical condition?")
                        refs.append(item["gold_summary"])
                        questions.append(item["question"])

                    preds = [p if metrics_calc.validate_text(p) else "What is the medical condition?" for p in preds]

                    print("Calculating metrics...")
                    rouge = [metrics_calc.calculate_rouge_l(p, r) for p, r in zip(preds, refs)]
                    bert = metrics_calc.calculate_bertscore(preds, refs)
                    semantic = [metrics_calc.calculate_semantic_coherence(p, q) for p, q in zip(preds, questions)]
                    summac = [metrics_calc.calculate_summac(p, q) for p, q in zip(preds, questions)]
                    qe = [metrics_calc.calculate_qe_overlap(p, q) for p, q in zip(preds, questions)]
                    ent = [metrics_calc.calculate_entailment(p, q) for p, q in zip(preds, questions)]
                    entity = [metrics_calc.calculate_entity_preservation(p, q) for p, q in zip(preds, questions)]

                    results = {
                        "model": model_name,
                        "strategy": strategy,
                        "n_shot": n_shot,
                        "metrics": {
                            "ROUGE-L": max(float(np.mean(rouge)), 0.0),
                            "BERTScore_F1": max(float(bert["f1"]), 0.0),
                            "Semantic_Coherence": max(float(np.mean(semantic)), 0.0),
                            "SummaC": max(float(np.mean(summac)), 0.0),
                            "QE_Overlap": max(float(np.mean(qe)), 0.0),
                            "Entailment": max(float(np.mean(ent)), 0.0),
                            "Entity_Preservation": max(float(np.mean(entity)), 0.0),
                        },
                    }

                    print("-" * 80)
                    print("ALL METRICS")
                    print("-" * 80)
                    for k, v in results["metrics"].items():
                        print(f"{k:.<30} {v:.6f}")
                    print("-" * 80)

                    all_results.append(results)
                    all_predictions.append(
                        {
                            "model": model_name,
                            "strategy": strategy,
                            "n_shot": n_shot,
                            "predictions": preds,
                            "references": refs,
                            "questions": questions,
                        }
                    )

            summarizer.cleanup()

        except Exception as e:
            print(f"Error in model {model_name}: {e}")
            continue

    print("=" * 80)
    print("CREATING EXCEL WITH 30 SAMPLES PER CONFIGURATION")
    print("=" * 80)

    excel_path = f"{output_dir}/CHQ_Results_30Samples.xlsx"

    summary_data = []
    for r in all_results:
        summary_data.append(
            {
                "Model": r["model"],
                "Strategy": r["strategy"],
                "N-Shot": r["n_shot"],
                "ROUGE-L": round(r["metrics"]["ROUGE-L"], 6),
                "BERTScore_F1": round(r["metrics"]["BERTScore_F1"], 6),
                "Semantic_Coherence": round(r["metrics"]["Semantic_Coherence"], 6),
                "SummaC": round(r["metrics"]["SummaC"], 6),
                "QE_Overlap": round(r["metrics"]["QE_Overlap"], 6),
                "Entailment": round(r["metrics"]["Entailment"], 6),
                "Entity_Preservation": round(r["metrics"]["Entity_Preservation"], 6),
            }
        )

    df_summary = pd.DataFrame(summary_data)

    pred_data = []
    for p in all_predictions:
        config_key = f"{p['model']}_{p['strategy']}_{p['n_shot']}shot"
        for i in range(min(30, len(p["questions"]))):  # always safe
            pred_data.append(
                {
                    "Config_ID": config_key,
                    "Sample_Number": i + 1,
                    "Model": p["model"],
                    "Strategy": p["strategy"],
                    "N-Shot": p["n_shot"],
                    "Original_Question": p["questions"][i][:400],
                    "Gold_Summary": p["references"][i],
                    "Model_Generated_Summary": p["predictions"][i],
                }
            )

    df_predictions = pd.DataFrame(pred_data)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="All_Results", index=False)
        df_predictions.to_excel(writer, sheet_name="All_Samples_30_Each", index=False)

        for model in df_summary["Model"].unique():
            model_data = df_summary[df_summary["Model"] == model].drop("Model", axis=1)
            sheet_name = model.replace(".", "_")[:31]
            model_data.to_excel(writer, sheet_name=sheet_name, index=False)

        for shot in shots:
            shot_df = df_predictions[df_predictions["N-Shot"] == shot]
            sheet_name = f"{shot}_shot_samples"
            shot_df.to_excel(writer, sheet_name=sheet_name, index=False)

    wb = load_workbook(excel_path)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except Exception:
                    pass
            ws.column_dimensions[column_letter].width = min(max_length + 2, 60)

    wb.save(excel_path)

    print("EVALUATION COMPLETE")
    if all_results:
        best_r = max(all_results, key=lambda x: x["metrics"]["ROUGE-L"])
        best_b = max(all_results, key=lambda x: x["metrics"]["BERTScore_F1"])

        print(f"Best ROUGE-L: {best_r['model']} | {best_r['strategy']} | {best_r['n_shot']}-shot")
        print(f"Score: {best_r['metrics']['ROUGE-L']:.6f}")

        print(f"Best BERTScore: {best_b['model']} | {best_b['strategy']} | {best_b['n_shot']}-shot")
        print(f"Score: {best_b['metrics']['BERTScore_F1']:.6f}")

    print(f"Excel Output: {excel_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("No GPU detected - using CPU\n")

    run_evaluation()
