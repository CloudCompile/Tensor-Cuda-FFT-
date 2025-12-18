"""
Syntax validation script - checks if all Python files are valid
Runs without requiring PyTorch/CUDA installation
"""
import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    """Check all Python files for syntax errors."""
    root = Path(__file__).parent
    python_files = list(root.rglob('*.py'))
    
    # Exclude .letta directory
    python_files = [f for f in python_files if '.letta' not in str(f)]
    
    print(f"Checking {len(python_files)} Python files for syntax errors...\n")
    
    errors = []
    for file_path in python_files:
        is_valid, error = check_syntax(file_path)
        if is_valid:
            print(f"[OK] {file_path.relative_to(root)}")
        else:
            print(f"[FAIL] {file_path.relative_to(root)}")
            print(f"  Error: {error}\n")
            errors.append((file_path, error))
    
    print(f"\n{'='*60}")
    if errors:
        print(f"FAILED: {len(errors)} files with syntax errors")
        for file_path, error in errors:
            print(f"  - {file_path.name}: {error}")
        return 1
    else:
        print(f"SUCCESS: All {len(python_files)} Python files have valid syntax!")
        print(f"\nPackage structure validated successfully.")
        print(f"\nTo run full tests, install dependencies:")
        print(f"  1. Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print(f"  2. Install pytest: pip install pytest")
        print(f"  3. Build package: pip install -e .")
        print(f"  4. Run tests: pytest tests/ -v")
        return 0

if __name__ == '__main__':
    sys.exit(main())
