

import random


actions = ["Click", "Scroll", "Exit"]


sample_space = [(a1, a2) for a1 in actions for a2 in actions]

print("Sample Space:")
print(sample_space)
print("Total outcomes:", len(sample_space))



event_E = [outcome for outcome in sample_space if "Click" in outcome]

print("\nEvent E (At least one Click):")
print(event_E)

probability_E = len(event_E) / len(sample_space)
print("Theoretical Probability of at least one Click:", probability_E)


trials = 1000
count_sum_7 = 0

for _ in range(trials):
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)
    
    if die1 + die2 == 7:
        count_sum_7 += 1

experimental_probability = count_sum_7 / trials

print("\nExperimental Probability of Sum = 7 (1000 rolls):")
print(experimental_probability)




### task 2 ######
coin_tossed_head = 1/2
dice_rolling_6 = 1/6
print(f'P(A) * P(B) : {coin_tossed_head * dice_rolling_6 :.3f}')

#finding probability without Replacement
red = 5
blue = 5
picking_1st_marble = red / (red+blue)
picking_2st_marble = (red-1)/(red+blue-1)

prob = picking_1st_marble * picking_2st_marble
print(f'probability that both are Red without Replacement : {prob:.2f}')





#task 3  Given probabilities
P_spam = 0.1
P_ham = 0.9

P_free_given_spam = 0.9
P_free_given_ham = 0.05

P_free = (P_free_given_spam * P_spam) + \
         (P_free_given_ham * P_ham)

P_spam_given_free = (P_free_given_spam * P_spam) / P_free

print("P(Free) =", P_free)
print("P(Spam | Free) =", P_spam_given_free)










