with open('inference.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('inference.py', 'w', encoding='utf-8', newline='\n') as f:
    for i, line in enumerate(lines):
        # specifically fix line 70 (index 69) and surrounding
        if i == 69:
             line = '    """Call LLM and parse action JSON. Returns a safe default on failure."""\n'
        f.write(line)
