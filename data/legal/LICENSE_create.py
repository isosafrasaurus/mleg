#!/usr/bin/env python3
import re
import sys
import json

def replace_match(match, values):
    key = match.group(1).strip()
    return values.get(key, match.group(0))

def process_template(template_content, values):
    pattern = r'!\s*\{\s*(.*?)\s*\}'
    return re.sub(pattern, lambda m: replace_match(m, values), template_content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python ./LICENSE_create.py mapping_file.json")
        sys.exit(1)
    
    mapping_file = sys.argv[1]
    with open(mapping_file, 'r') as f:
        config = json.load(f)

    template_path = config.get("template")
    if not template_path:
        print("Error: No 'template' field found in mapping file.")
        sys.exit(1)
    with open(template_path, 'r') as tf:
        template_content = tf.read()

    global_values = config.get("globals", {})

    mappings = config.get("mappings", [])
    if not mappings:
        print("Error: No 'mappings' found in mapping file.")
        sys.exit(1)
    
    for mapping in mappings:
        output_file = mapping.get("output")
        if not output_file:
            print("Warning: Skipping mapping with no output specified.")
            continue
        values = global_values.copy()
        values.update(mapping.get("values", {}))
        result = process_template(template_content, values)
        with open(output_file, 'w') as outf:
            outf.write(result)
        print(f"Processed template and saved output to '{output_file}'.")

if __name__ == '__main__':
    main()

# {
  # "template": "LICENSE.tex",
  # "globals": {
    # "PIERCE_ZHANG_EMAIL": "?",
    # "ALEX_BIYANOV_EMAIL": "?"
  # },
  # "mappings": [
    # {
      # "output": "Forms/LICENSE_jd.tex",
      # "values": {
        # "VOLUNTEER_NAME": "Jane Doe",
        # "VOLUNTEER_EMAIL": "\\texttt{jane@domain.com}",
        # "TERM_MONTHS": "6",
        # "NONRENEW_DAYS": "15",
        # "TERMINATION_DAYS": "15",
        # "DATA_TYPES": "Getty Images"
      # }
    # }
  # ]
# }