import random
import torch
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetConfig
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

class CustomDataset():
    def __init__(self, file, delimiter, header, names):
        # AG_NEWS : names = ['category', 'head', 'sentence']
        # IMDB_Dataset : names = ['catetory', 'sentence']
        file_type = file[file.rfind('.'):]
        if file_type == '.csv':
            self.df = pd.read_csv(file, delimiter=delimiter, header=header, names=names)
        elif file_type == '.json':
            pass

        self.sentence = self.df['sentence']
        self.category = self.df['category']
        #self.category = [1 if l == 'positive' else 0 for l in self.category]
        self.category = [c - min(self.category) for c in self.category]

        self.data = list(zip(self.sentence, self.category))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.df.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return list(self.sentence[idx]), list(self.category[idx])

def add_padding_and_truncate(input_ids):
    MAX_LEN = 64
    for index, input_id in enumerate(input_ids):
        for i in range(MAX_LEN - len(input_id)):
          input_id.insert(0, 0)
        if len(input_id) > MAX_LEN:
          input_ids[index] = input_id[:MAX_LEN]

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def run(bert_model):
    # Check if there is a GPU available.
    print(bert_model + " model will be used")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print(torch.cuda.get_device_name(0), 'will be used.')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    # Load data from custom dataset class
    print("Load Dataset")
    #dataset = CustomDataset(file="./Code/Dataset/IMDB_Dataset.csv", names=['sentence', 'category'], delimiter=',', header=None)
    dataset = pd.read_csv("./Code/Dataset/ag_news_train.csv", delimiter=',', header=None, names=['category', 'head', 'sentence'])
    sentences, labels = dataset[:]
    print("Training size : %d" % len(sentences))

    # Load BERT Tokenizer.
    if bert_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif bert_model == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
    elif bert_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    elif bert_model == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    print('Original : ', sentences[0])
    print('Tokenized : ', tokenizer.tokenize(sentences[0]))
    print('Token IDs : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    # Sentence to embedded ID.
    print("\nEncoding sentence to embedded ID...")
    input_ids = []
    for s in sentences:
        encoded_sentence = tokenizer.encode(s, max_length = 512, add_special_tokens=True)
        input_ids.append(encoded_sentence)

    print('original: ', sentences[0])
    print('id: ', input_ids[0])
    print('Max sentence length: ', max([len(sen) for sen in input_ids]))

    # Fit sentence's length to MAX_LEN(For gpu memory).
    add_padding_and_truncate(input_ids)
    print('After max question length: ', max([len(id) for id in input_ids]))

    attention_masks = []

    for id in input_ids:
        att_mask = [int(token_id) > 0 for token_id in id]
        attention_masks.append(att_mask)

    # Split data to train and validation.
    train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(input_ids, labels, random_state=2020, test_size=0.1)
    train_masks, valid_masks, _, _ = train_test_split(attention_masks, labels, random_state=2020, test_size=0.1)

    # Data to tensor
    train_inputs = torch.tensor(train_inputs)
    valid_inputs = torch.tensor(valid_inputs)

    train_labels = torch.tensor(train_labels)
    valid_labels = torch.tensor(valid_labels)

    train_masks = torch.tensor(train_masks)
    valid_masks = torch.tensor(valid_masks)

    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    ##############################
    ########## Training ##########
    ##############################
    if bert_model == 'bert':
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = 4, # depends on data.
            output_attentions = False,
            output_hidden_states = False
        )
    elif bert_model == 'albert':
        model = AlbertForSequenceClassification.from_pretrained(
            'albert-base-v2',
            num_labels = 4, # depends on data.
            output_attentions = False,
            output_hidden_states = False
        )
    elif bert_model == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels = 4, # depends on data.
            output_attentions = False,
            output_hidden_states = False
        )
    elif bert_model == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',
            num_labels = 4, # depends on data.
            output_attentions = False,
            output_hidden_states = False
        )


    if device.type == 'cuda':
        model.cuda()

    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    training_time = time.time()
    for epoch_i in range(0, epochs):
        #print("")
        #print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        #print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 30 batches.
            if step % 30 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                #print("    Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear calculated gradients before performing a backward pass.
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step by using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("    Average training loss: {0:.2f}".format(avg_train_loss))
        print("    Training epcoh took: {:}".format(format_time(time.time() - t0)))



        ##############################
        ######### Validation #########
        ##############################
        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:

            # Add batch to device
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # for saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("    Training Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("    Validation took: {:}".format(format_time(time.time() - t0)))

    print("\nTraining complete!")
    print("Training took: {:}".format(format_time(time.time() - training_time)))
    print()

    """
    plt.plot(loss_values, 'b-o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()
    """

    ##############################
    ############ Test ############
    ##############################
    #sentences, labels = dataset[:1000]

    dataset = pd.read_csv("./Code/Dataset/ag_news_test.csv", delimiter=',', header=None, names=['category', 'head', 'sentence'])
    sentences, labels = dataset[:]
    print("Test size : %d" % len(sentences))

    input_ids = []
    for s in sentences:
        encoded_sentence = tokenizer.encode(s, max_length = 512, add_special_tokens=True)
        input_ids.append(encoded_sentence)

    add_padding_and_truncate(input_ids)

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = 32

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()

    eval_accuracy = 0
    eval_steps = 0
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        eval_accuracy += flat_accuracy(logits, label_ids)
        eval_steps += 1

    print(bert_model + "Test Accuracy: {0:.2f}".format(eval_accuracy/eval_steps))
    print("\nDone")

bert_models = ['bert', 'albert', 'roberta', 'xlnet']
for b in bert_models:
    run(b)
