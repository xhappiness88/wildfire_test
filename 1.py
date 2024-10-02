def next_permutation(num):
    digits = list(str(num))
    n = len(digits)
    
    # Step 1: Find the largest index k such that digits[k] < digits[k + 1]
    k = n - 2
    while k >= 0 and digits[k] >= digits[k + 1]:
        k -= 1
    
    if k == -1:
        return -1  # No greater permutation exists
    
    # Step 2: Find the largest index l greater than k such that digits[k] < digits[l]
    l = n - 1
    while digits[k] >= digits[l]:
        l -= 1
    
    # Step 3: Swap digits[k] and digits[l]
    digits[k], digits[l] = digits[l], digits[k]
    
    # Step 4: Reverse the sequence from digits[k + 1] to the end
    digits = digits[:k + 1] + digits[k + 1:][::-1]
    
    return int(''.join(digits))

def format_output(num):
    challenge_token = 'hxo74j2e3'
    next_perm = next_permutation(num)
    
    if next_perm == -1:
        return str(next_perm)  # Return -1 as a string for consistency
    
    final_string = str(next_perm) + challenge_token
    
    # Replace every third character with 'X'
    final_output = ''.join(c if (i + 1) % 3 != 0 else 'X' for i, c in enumerate(final_string))
    
    return final_output

# Example usage:|
print(format_output(11121))  # Output: 11X11XxoX4jXe3
print(format_output(41352))  # Output: 41X23XxoX4jXe3