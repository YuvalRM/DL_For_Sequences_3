import random
import string


def generate_sequence(chars):
    return ''.join(random.choices(chars, k=random.randint(30, 40)))

def generate_positive_example():
    rand_times = random.randint(15,20)
    example = ['1'] * rand_times
    example += ['2'] * rand_times
    return ''.join(example)

def generate_negative_example():
    return f'1{generate_positive_example()}'

def generate_examples(num_examples=500):

    with open('pos_examples3', 'w') as pos_file:
        for _ in range(num_examples):
            pos_file.write(generate_positive_example() + '\n')

    with open('neg_examples3', 'w') as neg_file:
        for _ in range(num_examples):
            neg_file.write(generate_negative_example() + '\n')






if __name__ == "__main__":
    generate_examples()
