def verifiable_rewards(input):
    input_list = input.split()
    total_words = len(input_list)

    total_letters = 0
    for val in input_list:
        if val == "":
            total_words -= 1
        else:
            total_letters += len(val)
    #this is average 
    if total_words == 0 or total_letters == 0:
        print(input)
        return -30
    else:
        return (total_letters-(input.count(" ") + input.count("\n")))/total_words 

