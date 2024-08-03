from Part_3.to_submit.bilstmTrain import main
import matplotlib.pyplot as plt

print("Starting POS a")
accs_a_POS = main('a', "pos/train",
                  "Models/model_pos_a.pt", "pos/dev", "POS")


print("Starting POS b")
accs_b_POS = main('b', 'pos/train', 'Models/model_pos_b.pt',
                  "pos/dev", "POS")

print("Starting POS c")
accs_c_POS = main('c', 'pos/train', 'Models/model_pos_c.pt',
                  "pos/dev", "POS")

print("Starting POS d")
accs_d_POS = main('d', 'pos/train', 'Models/model_pos_d.pt',
                  "pos/dev", "POS")
x_a, y_a = zip(*accs_a_POS)
x_b, y_b = zip(*accs_b_POS)
x_c, y_c = zip(*accs_c_POS)
x_d, y_d = zip(*accs_c_POS)
plt.figure(figsize=(10, 6))

plt.plot(y_a, x_a, label='a', marker='o', markersize=1)
plt.plot(y_b, x_b, label='b', marker='s', markersize=1)
plt.plot(y_c, x_c, label='c', marker='^', markersize=1)
plt.plot(y_d, x_d, label='d', marker='x', markersize=1)

# Labeling
plt.xlabel('Number of Sentences')
plt.ylabel('Accuracy')
plt.title('POS')
plt.legend()
plt.savefig('pos.jpg', format='jpg', dpi=300)
# Show the plot
plt.grid(True)
plt.show()

accs_a_NER = main('a', 'ner/train',
                  "Models/model_ner_a.pt", "ner/dev", "NER")
accs_b_NER = main('b', 'ner/train',
                  "Models/model_ner_b.pt", "ner/dev", "NER")
accs_c_NER = main('c', 'ner/train',
                  "Models/model_ner_c.pt", "ner/dev", "NER")
accs_d_NER = main('d', 'ner/train',
                  "Models/model_ner_d.pt", "ner/dev", "NER")


x_a, y_a = zip(*accs_a_NER)
x_b, y_b = zip(*accs_b_NER)
x_c, y_c = zip(*accs_c_NER)
x_d, y_d = zip(*accs_d_NER)
plt.figure(figsize=(10, 6))

plt.plot(y_a, x_a, label='a', marker='o', markersize=1)
plt.plot(y_b, x_b, label='b', marker='s', markersize=1)
plt.plot(y_c, x_c, label='c', marker='^', markersize=1)
plt.plot(y_d, x_d, label='d', marker='x', markersize=1)

# Labeling
plt.xlabel('Number of Sentences')
plt.ylabel('Accuracy')
plt.title('NER')
plt.legend()
plt.savefig('ner.jpg', format='jpg', dpi=300)
# Show the plot
plt.grid(True)
plt.show()