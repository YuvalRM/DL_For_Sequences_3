import random
import string
import torch


def generate_sequence(chars):
    return ''.join(random.choices(chars, k=random.randint(1, 7)))

def generate_positive_example():
    digits = string.digits[1:]
    return f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['a'])}" + \
            f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['b'])}" + \
            f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['c'])}" + \
            f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['d'])}" + \
            f"{generate_sequence(digits)}"

def generate_negative_example():
    digits = string.digits[1:]
    return f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['a'])}" + \
            f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['c'])}" + \
            f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['b'])}" + \
            f"{generate_sequence(digits)}" + \
            f"{generate_sequence(['d'])}" + \
            f"{generate_sequence(digits)}"
def generate_examples(num_examples=500):
    positive_examples = [generate_positive_example() for _ in range(num_examples)]
    negative_examples = [generate_negative_example() for _ in range(num_examples)]

    with open('pos_examples', 'w') as pos_file:
        for example in positive_examples:
            pos_file.write(example + '\n')

    with open('neg_examples', 'w') as neg_file:
        for example in negative_examples:
            neg_file.write(example + '\n')






if __name__ == "__main__":
    generate_examples()
