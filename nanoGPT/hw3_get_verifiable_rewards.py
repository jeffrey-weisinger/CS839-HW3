def verifiable_rewards(input):
    white_space = ["\t", "\n", " "]

    reward = 0
    #checking first character. there is no opportunity to "end" a word here.
    #starting a new word
    if input[0] not in white_space:
        reward += 5
    #not starting a new word
    else:
        reward -= 2
    #i know that the input will have at least one character here, so we should always enter loop
    for i, char in enumerate(input[1:]):
        #starting a new word
        if char not in white_space and input[i] in white_space:
            reward += 5
        #ending a word
        elif char in white_space and input[i] not in white_space:
            reward += 3
        else:
            reward -= 2
    return reward

