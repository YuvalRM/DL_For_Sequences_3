import random
import string


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

    with open('pos_examples', 'w') as pos_file:
        for _ in range(num_examples):
            pos_file.write(generate_positive_example() + '\n')

    with open('neg_examples', 'w') as neg_file:
        for _ in range(num_examples):
            neg_file.write(generate_negative_example() + '\n')






if __name__ == "__main__":
    generate_examples()
