def compare_strings(test, code):
    # Ensure both strings are of equal length
    if len(test) != len(code):
        return -1  # Return -1 if strings are not of equal length

    # Initialize counter for matching characters
    matching_characters = 0
    
    # Compare characters at each position
    for i in range(len(test)):
        if test[i] == code[i]:
            matching_characters += 1

    return matching_characters


# Example usage:
test = "Anurag"
code = "Anurag"

result = compare_strings(test, code)
if result == -1:
    print("Error: Strings are not of equal length")
else:
    print(f"Number of matching characters: {result}")
