"""Run only the gemini-3.5-flash figure judge.
"""
from pathlib import Path
import evaluate_single_mllm_judge as judge

if __name__ == "__main__":
    
    judge.BASE_URL = "http://xxxx/v1/"
    judge.API_KEY = "sk-xxxxx"
    judge.MODEL_NAME = "gemini-3.5-flash"
    
    TARGET_LANGUAGE = "latex"  # One of: python, Matlab, R, latex.
    judge.GT_DIR = Path("data_gt") / TARGET_LANGUAGE
    # Keep false unless this gemini gateway explicitly supports response_format.
    judge.FORCE_JSON_MODE = False
    
    test_list = ["gemini31p"]
    for TARGET_GENERATOR_MODEL in test_list:
        print(f"Evaluating {TARGET_GENERATOR_MODEL}...")
        judge.GEN_DIR = Path("results_final") / TARGET_GENERATOR_MODEL / TARGET_LANGUAGE
        judge.OUTPUT_RESULTS_FILE = Path("mllm_judge_results") / f"gemini35f__{TARGET_GENERATOR_MODEL}__{TARGET_LANGUAGE}__all.json"
        judge.main()
    
    
