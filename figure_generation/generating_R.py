import subprocess
from pathlib import Path
from tqdm import tqdm
import re


def batch_run_r_and_save_plots(input_dir: str, output_dir: str, execute_timeout: int = 60, install_timeout: int = 300):
    """
    :param input_dir: Directory path where .R files are stored
    :param output_dir: Directory path for saving the generated images
    :param execute_timeout: Maximum duration (in seconds) for executing a single R script, default is 60 seconds
    :param install_timeout: Maximum duration (in seconds) for automatically installing R packages, default is 300 seconds (5 minutes)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    r_files = list(input_path.glob("*.R")) + list(input_path.glob("*.r"))
    total_scripts = len(r_files)

    if total_scripts == 0:
        print(f"No R scripts found in directory '{input_dir}'.")
        return

    print(f"🚀 Found {total_scripts} R scripts, starting execution...\n")

    success_count = 0
    fail_count = 0
    error_details = []  

    for r_file in tqdm(r_files, desc="Progress", unit="file"):
        output_image = output_path / f"{r_file.stem}.png"
        
        r_file_str = str(r_file).replace('\\', '/')
        output_image_str = str(output_image).replace('\\', '/')

        # R command
        r_cmd = (
            f"png('{output_image_str}', width=8, height=6, units='in', res=300); "
            f"source('{r_file_str}', print.eval=TRUE); "
            f"dev.off()"
        )
        
        max_retries = 5 
        retry_count = 0
        is_success = False
        final_err_msg = ""
        
        while retry_count <= max_retries:
            try:

                subprocess.run(
                    ["Rscript", "-e", r_cmd],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=execute_timeout  
                )
                

                if output_image.exists():
                    is_success = True
                    break  
                else:
                    final_err_msg = "Although the R script did not report any errors and ended normally, it failed to detect the generated image file (possibly because no drawing statements were triggered within the code)."
                    break
            
            except subprocess.TimeoutExpired as e:
                final_err_msg = f"⏳ Timeout ({execute_timeout} seconds) for executing the R script, it was forcibly terminated by the system."
                break 
                
            except subprocess.CalledProcessError as e:
                err_output = e.stderr.strip()
                
                # Using regular expressions to handle the package missing error in R language
                match = re.search(r"there is no package called [\'\"‘]([^\'\"’]+)[\'\"’]", err_output)
                
                if match and retry_count < max_retries:
                    pkg_name = match.group(1)
                    tqdm.write(f"\n📦 Lack of R package '{pkg_name}', attempting automatic installation (auto-fix {retry_count + 1}/{max_retries})...")
                    
                    install_cmd = f"install.packages('{pkg_name}', repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/')"
                    
                    try:
                        # ==========================
                        # Install R package (with timeout)
                        # ==========================
                        subprocess.run(
                            ["Rscript", "-e", install_cmd],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=install_timeout
                        )
                        tqdm.write(f"✅ Package '{pkg_name}' installed successfully, re-running the script...")
                        retry_count += 1
                        continue 
                        
                    except subprocess.TimeoutExpired:
                        final_err_msg = f"⏳ Timeout ({install_timeout} seconds) for installing the R package, it was forcibly terminated by the system."
                        break
                        
                    except subprocess.CalledProcessError as install_e:
                        final_err_msg = f"Failed to automatically install package '{pkg_name}':\n{install_e.stderr.strip()}"
                        break
                        
                else:
                    if retry_count >= max_retries:
                        final_err_msg = f"Reached maximum automatic package installation retries ({max_retries}), final error:\n{err_output}"
                    else:
                        final_err_msg = err_output
                    break
                    
            except FileNotFoundError:
                print("\n❌ cannot find 'Rscript' command. Please ensure R language is installed and the environment variables are configured.")
                return

        # record the final result of a single file
        if is_success:
            success_count += 1
        else:
            fail_count += 1
            error_details.append((r_file.name, final_err_msg))

    # print the summary report
    print("\n" + "="*50)
    print("📊 Evaluation Execution Report")
    print("="*50)
    print(f"✅ Successfully generated images: {success_count}")
    print(f"❌ Failed/Timeout: {fail_count}")
    print("="*50)

    if fail_count > 0:
        print("\n📝 failure list:")
        for idx, (fname, err) in enumerate(error_details, 1):
            print(f"\n[{idx}] script: {fname}")
            print(f"    reason: {err}")


if __name__ == "__main__":
    
    models = ['gpt52']
    sets = ['exemplary','user_generated']  # 'exemplary',
    types = ['Temporal','Relational','Geospatial','Composition','Mathematical','Statistical'] 
    
    for model in models:
            for s in sets:
                for t in types:
    
                    SOURCE_R_DIR = f"xxxx/results_final/{model}/R/{s}/code_base/{t}"    
                    OUTPUT_IMG_DIR = f"xxxx/results_final/{model}/R/{s}/image_base/{t}"  
    
                    batch_run_r_and_save_plots(SOURCE_R_DIR, OUTPUT_IMG_DIR)
                
                for t in types:
                    
                    SOURCE_R_DIR = f"xxxx/results_final/{model}/R/{s}/code_variant1/{t}"    
                    OUTPUT_IMG_DIR = f"xxxx/results_final/{model}/R/{s}/image_variant1/{t}"  
    
                    batch_run_r_and_save_plots(SOURCE_R_DIR, OUTPUT_IMG_DIR)
                    
                for t in types:
                    
                    SOURCE_R_DIR = f"xxxx/results_final/{model}/R/{s}/code_variant2/{t}"    
                    OUTPUT_IMG_DIR = f"xxxx/results_final/{model}/R/{s}/image_variant2/{t}"  
    
                    batch_run_r_and_save_plots(SOURCE_R_DIR, OUTPUT_IMG_DIR)