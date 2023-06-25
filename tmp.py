import re

def remove_env_var_pattern(input_string):
    # Pattern to match "VAR=VALUE" at the beginning of the string, with optional leading/trailing whitespaces
    pattern = r"^\s*\w+=.*?\s+"

    # re.sub() replaces the matched pattern with an empty string
    result = re.sub(r"^\s*\w+=.*?\s+", "", input_string)

    return result

s = "OMP_NUM_THREAD=1 CUDA_VISIBLE_DEVICES=0,1 python train.py -k --abc=d  --num 2"

# Remove "OMP_NUM_THREAD=1 "
s = remove_env_var_pattern(s)

# Remove "CUDA_VISIBLE_DEVICES=0,1 "
s = remove_env_var_pattern(s)

print(s)  # Should print: "python train.py -k --abc=d  --num 2"