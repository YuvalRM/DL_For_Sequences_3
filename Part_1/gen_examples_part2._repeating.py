import random
import string


def generate_sequence(chars):
    return ''.join(random.choices(chars, k=random.randint(10, 20)))

def generate_positive_example():
    digits = string.digits[1:]
    rand_digits = random.choices(digits, k=5)
    rand_times = random.randint(3, 5)
    example = ''
    for i in range(rand_times):
        example += ''.join(rand_digits)

    return example

def generate_negative_example():
    digits = string.digits[1:]
    first_digit = random.choice(digits)
    return f'{first_digit}{generate_positive_example()}'

def generate_examples(num_examples=500):

    with open('pos_examples2', 'w') as pos_file:
        for _ in range(num_examples):
            pos_file.write(generate_positive_example() + '\n')

    with open('neg_examples2', 'w') as neg_file:
        for _ in range(num_examples):
            neg_file.write(generate_negative_example() + '\n')






if __name__ == "__main__":
    generate_examples()
