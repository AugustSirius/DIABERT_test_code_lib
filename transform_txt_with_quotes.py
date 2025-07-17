#!/usr/bin/env python3
"""
Transform a text file by wrapping each line in quotes.
Each line will be wrapped in double quotes.
"""

def transform_file_with_quotes(input_file, output_file):
    """
    Transform a text file by wrapping each line in quotes.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output text file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # Remove any trailing whitespace/newline, then add quotes and comma
                    line = line.rstrip()
                    if line:  # Only process non-empty lines
                        quoted_line = f'"{line}",'
                        outfile.write(quoted_line + '\n')
                    else:
                        outfile.write('\n')  # Preserve empty lines
        
        print(f"Successfully transformed {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    # Default file paths
    input_file = "report_diann_transition_group_id.txt"
    output_file = "report_diann_transition_group_id_quoted.txt"
    
    # You can customize these paths as needed
    print(f"Transforming {input_file} to {output_file}")
    print("Each line will be wrapped in double quotes.")
    
    transform_file_with_quotes(input_file, output_file)
    
    # Show a few examples of the transformation
    print("\nFirst few lines of the transformed file:")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 5:  # Show first 5 lines
                    print(f"  {line.rstrip()}")
                else:
                    break
    except:
        pass

if __name__ == "__main__":
    main() 