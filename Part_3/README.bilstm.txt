In bilstmTrain the options are:
dev_file -  which indicates what is the dev file we want to check acc on, if none is provided no acc check will run
task -  which indicates what task we are on (POS/NER) for acc purposes, if we are running ner it will skip 'O' predictions

In bilstmPredict the options are:
    output_file - indicating where we want to dump the results, a must