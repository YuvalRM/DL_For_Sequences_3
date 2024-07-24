import random
import string


def generate_sequence(chars):
    return ''.join(random.choices(chars, k=random.randint(10, 20)))

def generate_positive_example():
    digits = string.digits[1:]
    rand_digit = random.choice(digits)
    return f"{rand_digit}" + \
            f"{generate_sequence(digits)}" + \
            f"{rand_digit}"

def generate_negative_example():
    digits = string.digits[1:]
    first_digit = random.choice(digits)
    last_digit = random.choice(digits)

    while last_digit == first_digit:
        last_digit = random.choice(digits)

    return f"{first_digit}" + \
            f"{generate_sequence(digits)}" + \
            f"{last_digit}"

def generate_examples(num_examples=500):

    with open('pos_examples1', 'w') as pos_file:
        for _ in range(num_examples):
            pos_file.write(generate_positive_example() + '\n')

    with open('neg_examples1', 'w') as neg_file:
        for _ in range(num_examples):
            neg_file.write(generate_negative_example() + '\n')






if __name__ == "__main__":
    generate_examples()
