def convert_tabs(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    modified_lines = [line.replace('  ', '    ') if line.startswith('  ') else line for line in lines]

    with open(output_file, 'w') as f:
        f.writelines(modified_lines)

# Example usage:
input_file = 'wrappers.py'
output_file = '../robosuite_task/wrappers.py'

convert_tabs(input_file, output_file)