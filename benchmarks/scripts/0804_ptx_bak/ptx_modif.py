#!/usr/bin/env python3
"""
修复PTX中缺少@符号的问题
"""
import re
import sys

def fix_missing_at_symbols(input_file, output_file):
    """修复PTX中缺少的@符号"""
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    in_inline_asm = False
    fixed_count = 0
    
    for i, line in enumerate(lines):
        # 检测inline asm块的开始和结束
        if '// begin inline asm' in line:
            in_inline_asm = True
        elif '// end inline asm' in line:
            in_inline_asm = False
        
        # 在inline asm块中修复缺少@的条件执行指令
        if in_inline_asm:
            # 匹配没有@前缀的条件执行指令
            pattern = r'^(\s*)(%p\d+)\s+(ld|st)\.shared'
            match = re.match(pattern, line)
            if match:
                indent = match.group(1)
                pred = match.group(2)
                inst = match.group(3)
                # 添加@符号
                lines[i] = re.sub(pattern, rf'\1@\2 \3.shared', line)
                fixed_count += 1
                print(f"Fixed line {i+1}: Added @ before {pred}")
    
    # 写入修复后的文件
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    print(f"\nFixed {fixed_count} missing @ symbols")
    print(f"Output saved to {output_file}")

def check_syntax(ptx_file):
    """检查PTX文件中的潜在语法问题"""
    with open(ptx_file, 'r') as f:
        lines = f.readlines()
    
    issues = []
    in_inline_asm = False
    
    for i, line in enumerate(lines):
        if '// begin inline asm' in line:
            in_inline_asm = True
        elif '// end inline asm' in line:
            in_inline_asm = False
            
        # 检查可能的语法问题
        if in_inline_asm:
            # 检查条件执行指令是否缺少@
            if re.match(r'^\s*%p\d+\s+(ld|st)\.', line):
                issues.append(f"Line {i+1}: Missing @ before predicate")
            
            # 检查是否还有[symbol + register]形式
            if re.search(r'\[\w+_smem\s*\+\s*%r\d+\]', line):
                issues.append(f"Line {i+1}: Found [symbol + register] addressing")
    
    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious syntax issues found!")
    
    return len(issues) == 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_missing_at_symbol.py input.ptx output.ptx")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Fixing missing @ symbols in {input_file}...")
    fix_missing_at_symbols(input_file, output_file)
    
    print(f"\nChecking syntax in {output_file}...")
    check_syntax(output_file)