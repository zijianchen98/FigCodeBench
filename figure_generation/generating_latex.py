import pathlib
import subprocess
import tempfile
import shutil
from pdf2image import convert_from_path


LATEX_ENGINE = "xelatex"  
DPI = 300                         
# ===========================================
MAC_POPPLER_PATH = "/opt/homebrew/bin"


def compile_and_convert(tex_file_path, target_png_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        try:
            result = subprocess.run(
                [LATEX_ENGINE, "-interaction=nonstopmode", f"-output-directory={tmpdir}", str(tex_file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
  
                text=True, 
                encoding='utf-8',     
                errors='replace',  
                timeout=120 
            )
            
            if result.returncode != 0:
                print(f"   ❌ Compilation failed: {tex_file_path.name}")
                return False

            # 2. search PDF
            pdf_path = tmpdir_path / (tex_file_path.stem + ".pdf")
            if not pdf_path.exists():
                print(f"   ❌ Generated PDF not found: {tex_file_path.name}")
                return False

            # 3. PDF -> PNG
            images = convert_from_path(pdf_path, dpi=DPI, poppler_path=MAC_POPPLER_PATH)
            if images:

                target_png_path.parent.mkdir(parents=True, exist_ok=True)
                images[0].save(target_png_path, "PNG")
                return True
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Compilation timeout: {tex_file_path.name}")
        except Exception as e:
            print(f"   ❌ Processing error {tex_file_path.name}: {str(e)}")
            
    return False



if __name__ == "__main__":
    
    models = ['internvl35-38B']
    sets = ['exemplary','user_generated']  
    for model in models:
        for s in sets:
            # code_base
            SOURCE_CODE_DIR = f"xxxx/results_final/{model}/latex/{s}/code_base"          
            TARGET_FIGURE_DIR = f"xxxx/results_final/{model}/latex/{s}/image_base"    

            source_root = pathlib.Path(SOURCE_CODE_DIR)
            target_root = pathlib.Path(TARGET_FIGURE_DIR)

            tex_files = list(source_root.rglob("*.tex"))
            print(f"🚀 Begin task: A total of {len(tex_files)} files need to process...\n")

            success_count = 0
            for i, tex_file in enumerate(tex_files, 1):
                relative_path = tex_file.relative_to(source_root)
                target_png_path = target_root / relative_path.with_suffix(".png")
                
                print(f"[{i}/{len(tex_files)}] Processing: {relative_path}")
                
                if compile_and_convert(tex_file, target_png_path):
                    print(f"   ✅ Successfully converted")
                    success_count += 1
                else:
                    print(f"   ⚠️ Passed conversion")

            print(f"\n✨ Task completed!")
            print(f"📊 Success rate: {success_count}/{len(tex_files)}")
            print(f"🖼️ Results saved in: {target_root.absolute()}")
        
        for s in sets:
            # code_variant1
            SOURCE_CODE_DIR = f"xxxx/results_final/{model}/latex/{s}/code_variant1"         
            TARGET_FIGURE_DIR = f"xxxx/results_final/{model}/latex/{s}/image_variant1"     

            source_root = pathlib.Path(SOURCE_CODE_DIR)
            target_root = pathlib.Path(TARGET_FIGURE_DIR)

            tex_files = list(source_root.rglob("*.tex"))
            print(f"🚀 Begin task: A total of {len(tex_files)} files need to process...\n")

            success_count = 0
            for i, tex_file in enumerate(tex_files, 1):
                relative_path = tex_file.relative_to(source_root)
                target_png_path = target_root / relative_path.with_suffix(".png")
                
                print(f"[{i}/{len(tex_files)}] Processing: {relative_path}")
                
                if compile_and_convert(tex_file, target_png_path):
                    print(f"   ✅ Successfully converted")
                    success_count += 1
                else:
                    print(f"   ⚠️ Passed conversion")

            print(f"\n✨ Task completed!")
            print(f"📊 Success rate: {success_count}/{len(tex_files)}")
            print(f"🖼️ Results saved in: {target_root.absolute()}")
            
            
        for s in sets:
            # code_variant2
            SOURCE_CODE_DIR = f"xxxx/results_final/{model}/latex/{s}/code_variant2"         
            TARGET_FIGURE_DIR = f"xxxx/results_final/{model}/latex/{s}/image_variant2"     

            source_root = pathlib.Path(SOURCE_CODE_DIR)
            target_root = pathlib.Path(TARGET_FIGURE_DIR)

            tex_files = list(source_root.rglob("*.tex"))
            print(f"🚀 Begin task: A total of {len(tex_files)} files need to process...\n")

            success_count = 0
            for i, tex_file in enumerate(tex_files, 1):
 
                relative_path = tex_file.relative_to(source_root)
                target_png_path = target_root / relative_path.with_suffix(".png")
                
                print(f"[{i}/{len(tex_files)}] Processing: {relative_path}")
                
                if compile_and_convert(tex_file, target_png_path):
                    print(f"   ✅ Successfully converted")
                    success_count += 1
                else:
                    print(f"   ⚠️ Passed conversion")

            print(f"\n✨ Task completed!")
            print(f"📊 Success rate: {success_count}/{len(tex_files)}")
            print(f"🖼️ Results saved in: {target_root.absolute()}")