from Part_3.to_submit.bilstmPredict import main

main('c', 'Models/model_pos_c.pt',
     "pos/test", "test4.pos")

main('c', 'Models/model_ner_c.pt',
     "ner/test", "test4.ner")
