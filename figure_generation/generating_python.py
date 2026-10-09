import re
import sys
import os
from pathlib import Path
import traceback
from tqdm import tqdm
import tempfile
import subprocess

# Define a dictionary mapping common "module names" to "pip package names" to solve the problem of inconsistent import names and installation names.
PACKAGE_MAPPING = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "mpl_toolkits": "matplotlib",
    "skimage": "scikit-image",
    "nx": "networkx"
}

def evaluate_python_generation(input_dir: str, output_dir: str, dpi: int = 300, timeout: int = 60, install_timeout: int = 120):
    """
    Evaluate Python scripts for figure generation.
    Traverses the specified folder for .py scripts, forces replacement of plt.show(), and outputs high-resolution PNGs at 300 DPI.
    Includes multiprocessing isolation, timeout control, and automatic detection and installation of missing Python libraries.

    :param timeout: Maximum execution time for a single Python script (seconds), default is 60 seconds
    :param install_timeout: Maximum time for pip to automatically install packages (seconds), default is 120 seconds
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    py_files = list(input_path.glob("*.py"))
    total_scripts = len(py_files)

    if total_scripts == 0:
        print(f"No .py scripts found in '{input_dir}'.")
        return

    print(f"🧪 Found {total_scripts} Python plotting scripts, starting evaluation...\n")

    success_count = 0
    fail_count = 0
    error_details = []

    for py_file in tqdm(py_files, desc="Python evaluation progress", unit="file"):
        output_image = output_path / f"{py_file.stem}.png"
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                original_code = f.read()
                
            out_img_str = output_image.as_posix()

            # Delete plt.show() 和 plt.savefig() 
            original_code = re.sub(r'plt\.show\s*\(.*?\)', '', original_code)
            original_code = re.sub(r'plt\.savefig\s*\(.*?\)', '', original_code)

            new_code = (
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n\n"
            ) + original_code + (
                f"\n\nplt.savefig('{out_img_str}', dpi={dpi}, bbox_inches='tight', facecolor='white', transparent=False)\n"
                "plt.close('all')\n"
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=py_file.parent, delete=False, encoding='utf-8') as temp_f:
                temp_f.write(new_code)
                temp_script_path = temp_f.name

            max_retries = 5
            retry_count = 0
            is_success = False
            final_err_msg = ""

            try:
                while retry_count <= max_retries:
                    try:
                        subprocess.run(
                            [sys.executable, temp_script_path],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=timeout
                        )

                        # Verify the final status of the image
                        if output_image.exists():
                            is_success = True
                        else:
                            final_err_msg = "The script ran without errors, but no plotting functions were called, and no image was generated."
                        break  # Success or logical failure (no image generated), break the retry loop

                    except subprocess.TimeoutExpired:
                        final_err_msg = f"⏳ Execution timeout (exceeded设定 threshold {timeout} seconds), forcefully terminated by the system."
                        break
                        
                    except subprocess.CalledProcessError as e:
                        err_output = e.stderr.strip()
                        
                        # Use regular expression to match Python missing package errors. Common ones are ModuleNotFoundError or ImportError.
                        match = re.search(r"(?:ModuleNotFoundError|ImportError): No module named [\'\"]([^\'\"]+)[\'\"]", err_output)
                        
                        if match and retry_count < max_retries:
                            module_name = match.group(1)
                            # Obtain the actual package names that need to be installed (handling cases such as cv2 -> opencv-python)
                            pkg_name = PACKAGE_MAPPING.get(module_name, module_name)
                            
                            tqdm.write(f"\n📦 Module missing has been detected. '{module_name}', Installing via pip '{pkg_name}' (automatic restoration {retry_count + 1}/{max_retries})...")
                            
                            try:
                                # pip install
                                subprocess.run(
                                    [sys.executable, "-m", "pip", "install", pkg_name],
                                    capture_output=True,
                                    text=True,
                                    check=True,
                                    timeout=install_timeout
                                )
                                tqdm.write(f"✅ Library '{pkg_name}' installed successfully, re-running the code...")
                                retry_count += 1
                                continue 
                                
                            except subprocess.TimeoutExpired:
                                final_err_msg = f"⏳ pip install '{pkg_name}' timeout (exceeded {install_timeout} seconds), giving up on repair."
                                break
                            except subprocess.CalledProcessError as install_e:
                                final_err_msg = f"❌ pip install '{pkg_name}' failed:\n{install_e.stderr.strip()}"
                                break
                        else:
                            if retry_count >= max_retries:
                                final_err_msg = f"Reached maximum automatic package installation retries ({max_retries} times), final error:\n{err_output}"
                            else:
                                final_err_msg = err_output
                            break

                # Record the final result of a single file
                if is_success:
                    success_count += 1
                else:
                    fail_count += 1
                    error_details.append((py_file.name, final_err_msg))

            finally:
                if os.path.exists(temp_script_path):
                    os.remove(temp_script_path)

        except Exception as e:
            fail_count += 1
            error_details.append((py_file.name, f"The main process encountered an error while processing the file:\n{traceback.format_exc()}"))

    actual_images = len(list(output_path.glob("*.png")))
    exec_success_rate = (success_count / total_scripts) * 100 if total_scripts else 0
    file_match_rate = (actual_images / total_scripts) * 100 if total_scripts else 0

    print("\n" + "="*60)
    print("📊 【FigCodeBench - Python Evaluation Final Statistics Report】")
    print("="*60)
    print(f"1. Total number of original .py scripts (expected):  {total_scripts} scripts")
    print(f"2. Number of successfully executed and image-generated scripts:    {success_count} scripts")
    print(f"3. Number of failed executions or scripts without images generated:  {fail_count} scripts")
    print(f"4. Total number of images actually generated locally:  {actual_images} images")
    print("-" * 60)
    print(f"📈 Core metrics:")
    print(f"   - Code execution success rate: {exec_success_rate:.2f}%")
    print(f"   - File count consistency: {file_match_rate:.2f}% ({'✅ Fully consistent' if total_scripts == actual_images else '⚠️ Inconsistent count'})")
    print("="*60)
    
    if fail_count > 0:
        print("\n📝 Failure/Timeout Details (Partial Display):")
        for idx, (fname, err) in enumerate(error_details[:10], 1):
            print(f"\n[{idx}] Script: {fname}")
            # Long error truncation display
            err_str = str(err)
            print(f"    Reason: {err_str[:500] + '... [Truncated]' if len(err_str) > 500 else err_str}")



if __name__ == "__main__":

    models = ['qwen35-122b']
    sets = ['exemplary','user_generated']
    types = ['Geospatial','Composition','Mathematical','Statistical','Temporal','Relational']
    for model in models:
        for s in sets:
            for t in types:
                SOURCE_PY_DIR = f"xxxx/results_final/{model}/python/{s}/code_base/{t}"    
                OUTPUT_IMG_DIR = f"xxxx/results_final/{model}/python/{s}/image_base/{t}"  
                
                evaluate_python_generation(SOURCE_PY_DIR, OUTPUT_IMG_DIR, dpi=300)
                
            for t in types:
                SOURCE_PY_DIR = f"xxxx/results_final/{model}/python/{s}/code_variant1/{t}"    
                OUTPUT_IMG_DIR = f"xxxx/results_final/{model}/python/{s}/image_variant1/{t}"  
                
                evaluate_python_generation(SOURCE_PY_DIR, OUTPUT_IMG_DIR, dpi=300)
                
            for t in types:
                SOURCE_PY_DIR = f"xxxx/results_final/{model}/python/{s}/code_variant2/{t}"    
                OUTPUT_IMG_DIR = f"xxxx/results_final/{model}/python/{s}/image_variant2/{t}"  
                
                evaluate_python_generation(SOURCE_PY_DIR, OUTPUT_IMG_DIR, dpi=300)