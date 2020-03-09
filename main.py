from torch import optim

from data_util import DataUtil

from models import *
from argument_parsers import *

import time
from time import strftime, localtime
import math
import random

import json

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from os.path import exists

import logging


class Experiment:
    def __init__(self, config, sequence_length=20, reload_data=True):
        # Hyper Parameters
        self.sequence_length = sequence_length
        self.hidden_size = 128
        self.num_layers = 1

        self.config = config
        self.data = DataUtil(data_dir=config.data_dir, vocab_dir=config.vocab_dir,
                             split_by_sentence=not config.split_by_section, skip_list=config.skip_list)

        if not self.config.filtered:
            self.data.make_dir(self.config.output_dir + "/models/")

        if reload_data:
            for ds in self.config.textbook_data_sets:
                self.data.load_textbook_train_dev_data(config.data_dir + 'medlit/train/' + ds,
                                                       config.data_dir + 'medlit/dev/' + ds)
            # train
            self.data.load_i2b2_train_data(train_base_dir=config.data_dir + '/i2b2_ehr/')
            # test
            self.data.load_test_data(ref_base_dir=config.data_dir + '/i2b2_ehr/')
            # dev
            self.data.load_test_data(ref_base_dir=config.data_dir + '/i2b2_ehr/', type='dev')

        else:
            self.data.load_split_data()

        self.data.make_dir(self.config.output_dir)

        log_file_name = strftime("log_%Y_%m_%d_%H_%M_%S", localtime())
        self.logger = self.setup_logger(self.config.output_dir + '/%s.txt' % log_file_name)

        if exists(config.vocab_dir + "/NaturalLang.pkl") and not reload_data:
            print("Loading vocab")
            self.data.load_vocab()
        else:
            print("Building vocab")
            self.data.build_vocab(self.data.textbook_train_data, pretrain=False)

        self.model = None
        self.use_cuda = torch.cuda.is_available()

        if not self.config.filtered:
            if self.config.model_type == 'gru_rnn':
                self.model = GRURNN(
                    self.config.embedding_size, self.hidden_size, self.data.input_lang,
                    self.data.pretrained_embeddings, self.num_layers, self.data.input_lang.n_words,
                    self.data.output_lang.n_words, self.config.dropout)
            elif self.config.model_type == 'attn_gru_rnn':
                self.model = AttentionGRURNN(
                    self.config.embedding_size, self.hidden_size, self.data.input_lang,
                    self.data.pretrained_embeddings, self.num_layers, self.data.input_lang.n_words,
                    self.data.output_lang.n_words, self.config.dropout)
            elif self.config.model_type == 'cnn':
                self.model = CNN(
                    self.data.input_lang.n_words, self.data.output_lang.n_words, self.config.embedding_size,
                    self.data.input_lang, self.data.pretrained_embeddings, self.config.dropout)

            self.epoch_start = 1

            if self.use_cuda:
                self.model = self.model.cuda()

    def setup_logger(self, log_file, level=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(level)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log(self, info):
        print(info)
        if self.logger is not None:
            self.logger.info(info)

    def as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(self, since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (self.as_minutes(s), self.as_minutes(rs))

    def train(self, data_setup, save_model_dir,
              print_every=20, plot_every=100, learning_rate=0.001):
        start = time.time()

        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        if self.config.model_type == 'cnn' and self.config.transfer_learning:
            self.model.output_size = self.data.output_lang.n_words
            if self.config.reuse_embedding_layer_only:
                self.model.init_conv1_layer()
                self.model.init_conv2_layer()
                self.model.init_fc_layers()
            if self.config.reuse_embedding_conv1_layers:
                self.model.init_conv2_layer()
                self.model.init_fc_layers()
            if self.use_cuda:
                self.model = self.model.cuda()
        elif self.config.transfer_learning:
            self.model.freeze_layer("fc1")

        if self.config.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr=learning_rate, momentum=0.9)
        elif self.config.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=learning_rate)

        self.log('data_setup:' + str(data_setup))

        train_data = []

        for data_set in data_setup:
            data_ratio = data_setup[data_set]
            data = self.data.get_dataset(data_set)
            train_data += self.data.get_data_subset(data, data_ratio)
        print('len train_data:', len(train_data))
        print('training data examples:', train_data[:5])

        if self.config.downsampling:
            train_data = self.data.downsampling(
                train_data, number_samples=self.config.downsampling_size)

        num_train_data = len(train_data)
        print('num_train_data:', num_train_data)
        print('train_data:', train_data[:10])
        num_batches = int(np.ceil(num_train_data / float(self.config.batch_size)))
        self.log('num_batches: ' + str(num_batches))

        if self.config.weighted_loss:
            loss_weight = self.data.get_label_weight(train_data)
            if self.use_cuda:
                loss_weight = loss_weight.cuda()
        else:
            loss_weight = None

        max_dev_acc = 0

        for epoch in range(self.epoch_start, self.config.num_train_epochs + 1):
            batch_start = time.time()
            correct = 0
            total = 0

            random.shuffle(train_data)

            self.model.train()

            for cnt, i in enumerate(random.sample(range(num_batches), num_batches), start=1):
                inputs, seq_lengths, targets, batch = self.data.construct_batch(
                    self.config.batch_size * i, self.config.batch_size * (i + 1),
                    train_data, fixed_length=True if self.config.model_type == 'cnn' else False)

                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                if self.config.model_type == 'cnn':
                    outputs = self.model(inputs)  # for CNN
                elif self.config.model_type == 'attn_gru_rnn':
                    outputs = self.model(inputs, self.data.input_lang, seq_lengths)
                else:
                    outputs = self.model(inputs, seq_lengths)

                _, predicted = torch.max(outputs.data, dim=1)

                total += targets.data.size(0)
                correct += (predicted == targets.data).sum()
                batch_train_acc = 100.0 * (predicted == targets.data).sum() / targets.data.size(0)

                loss = F.cross_entropy(outputs, targets, weight=loss_weight)
                loss.backward()
                optimizer.step()
                self.log("Epoch %d, batch %d / %d: train loss = %f, train accuracy = %f %%"
                         % (epoch, cnt, num_batches, loss.data.item(), batch_train_acc))

                print_loss_total += loss.data.item()
                plot_loss_total += loss.data.item()

                if cnt % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    self.log('Average batch loss: %s' % str(print_loss_avg))
                    self.log(self.time_since(batch_start, cnt * 1.0 / num_batches))

                if cnt % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
            self.log('Epoch %d is done' % epoch)
            self.log('Epoch %d Train Accuracy: %f %%' % (epoch, 100.0 * correct / total))
            self.log(self.time_since(start, epoch * 1.0 / self.config.num_train_epochs))

            datasets = []
            print("TUNING SET IS: " + str(self.config.tuning_set))
            if 'ALL' in self.config.tuning_set or 'MedLit' in self.config.tuning_set:
                self.log("Test on MedLit Dev: ")
                datasets.append(self.data.TEXTBOOK_DEV)
            if 'ALL' in self.config.tuning_set or 'i2b2' in self.config.tuning_set:
                self.log("Test on i2b2 EHR Dev: ")
                datasets.append(self.data.i2b2_DEV)

            self.log("Tuning on:")
            self.log(datasets)
            dev_acc = self.test(datasets=datasets, epoch=epoch, calc_confusion_matrix=True)

            # save intermediate training results
            if dev_acc > max_dev_acc:
                save_path = save_model_dir + "/models/best_model.pt"
                torch.save(self.model, save_path)
                self.log('Best Model saved in file: %s' % save_path)
                max_dev_acc = dev_acc

                if 'i2b2' in self.config.test_set:
                    self.log("Test on i2b2 Test:")
                    self.test(datasets=[self.data.i2b2_TEST], epoch=epoch, print_test_results=True)

            save_path = save_model_dir + "/models/epoch_" + str(epoch) + ".pt"
            torch.save(self.model, save_path)
            self.log('Model saved in file: %s' % save_path)

    def test(self, datasets, epoch=-1, calc_confusion_matrix=True,
             generate_reports=True, print_test_results=False, print_examples=False):
        if self.model is None:
            self.log('Restoring model from ' + self.config.reload_model_file)

            if torch.cuda.is_available():
                self.model = torch.load(self.config.reload_model_file)
            else:
                self.model = torch.load(self.config.reload_model_file, map_location='cpu')

            self.log('Model is restored')

        self.model.eval()

        start = time.time()

        data = []

        dataset_name = '_'.join(datasets)

        for dataset in datasets:
            data.extend(self.data.get_dataset(dataset))

        if self.config.downsampling:
            data = []
            for dataset in datasets:
                data.extend(self.data.get_dataset(dataset))
                data = self.data.downsampling(
                    data, number_samples=500)

        num_test_data = len(data)
        self.log("num_test_data: " + str(num_test_data))
        num_batches = int(np.ceil(num_test_data / float(self.config.batch_size)))
        self.log('num_batches: ' + str(num_batches))

        correct = 0
        total = 0
        loss = 0.0
        labels = []
        predictions = []
        examples = []

        for i in range(num_batches):

            inputs, seq_lengths, targets, batch = self.data.construct_batch(
                self.config.batch_size * i, self.config.batch_size * (i + 1),
                data,
                fixed_length=True if self.config.model_type == 'cnn' else False)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            if self.config.model_type == 'cnn':
                outputs = self.model(inputs)  # for CNN
            elif self.config.model_type == 'attn_gru_rnn':
                outputs = self.model(inputs, self.data.input_lang, seq_lengths)
            else:
                outputs = self.model(inputs, seq_lengths)
            
            _, predicted = torch.max(outputs.data, dim=1)

            ordered = torch.sort(outputs.data)

            total += targets.data.size(0)
            correct += (predicted == targets.data).sum()
            labels.extend(targets.cpu().data.numpy().tolist())
            predictions.extend(predicted.cpu().numpy().tolist())

            loss += F.cross_entropy(outputs, targets).data.item()

            if print_examples or print_test_results:
                for k, d in enumerate(batch):
                    examples.append([d[0], d[1].replace('\r', ' ').replace('\n', ' ').replace('\t', ' '),
                                     d[2], d[3], str(d[4]), str(d[5]),
                                     self.data.output_lang.get_word(predicted[k].cpu().data.item()),
                                     self.data.output_lang.get_word(int(ordered[1][k][outputs.data.shape[1]-2])),
                                     self.data.output_lang.get_word(int(ordered[1][k][outputs.data.shape[1]-3]))])

        if print_examples:
            self.data.make_dir(self.config.output_dir + '/test_saved')
            self.log("Save examples to: " + self.config.output_dir + '/test_saved')
            with open(self.config.output_dir + '/test_saved/' + dataset_name + 'epoch_%d.txt' % epoch, 'w') as f:
                f.write("#\tSentence\tTrue\tHeader String\tLocation\tLine\tPrediction 1\tPrediction 2\tPrediction 3\n")
                for e in examples:
                    f.write('\t'.join(e) + '\n')

        self.log('Epoch %d ' % epoch + 'Time used: ' + str(time.time() - start))
        self.log('Epoch %d ' % epoch + 'Test loss: %f' % loss)
        self.log('Epoch %d ' % epoch + 'Test Accuracy: %f %%' % (100.0 * correct / total))
        self.log('Epoch %d ' % epoch + 'Test Precision: %f %%' %
                 (100.0 * precision_score(labels, predictions, average='micro')))
        self.log('Epoch %d ' % epoch + 'Test Recall: %f %%' %
                 (100.0 * recall_score(labels, predictions, average='micro')))
        self.log('Epoch %d ' % epoch + 'Test F1 Score: %f %%' %
                 (100.0 * f1_score(labels, predictions, average='micro')))

        text_labels = [self.data.output_lang.get_word(l) for l in labels]
        text_preds = [self.data.output_lang.get_word(l) for l in predictions]
        label_set = sorted(list(set(text_labels)))
        if calc_confusion_matrix:
            cm = confusion_matrix(text_labels, text_preds, labels=label_set)
            self.log('confusion_matrix for epoch %d: ' % epoch)
            header = '\t'.join(label_set)
            self.log(header)
            for i, row in enumerate(list(cm)):
                row = [str(num) for num in row]
                self.log('\t'.join([label_set[i]] + row))
            np.savetxt(self.config.output_dir + '/' + dataset_name + '_confusion_matrix_epoch_%d.csv' % epoch,
                       cm, fmt='%d', header=header, delimiter=',')
            self.log('Saved confusion matrix!')

        if generate_reports:
            reports = classification_report(text_labels, text_preds, labels=label_set, target_names=label_set, digits=4)
            self.log(reports)
            with open(self.config.output_dir + '/' + dataset_name + '_report_epoch_%d.txt' % epoch, 'w') as f:
                f.write(reports)
            self.log('Saved report!')

        if print_test_results:
            with open(self.config.output_dir + '/' + dataset_name + '_predictions_epoch_%d.json' % epoch, 'w') as f:
                json.dump(examples, f, indent=4, sort_keys=True)
        return 100.0 * correct / total

    def test_one(self, header, text):
        if self.model is None:
            self.log('Restoring model from ' + self.config.reload_model_file)

            if torch.cuda.is_available():
                self.model = torch.load(self.config.reload_model_file)
            else:
                self.model = torch.load(self.config.reload_model_file, map_location='cpu')
            self.log('Model is restored')
            self.model.eval()
            if self.use_cuda:
                self.model = self.model.cuda()

        inputs, seq_lengths, targets = self.data.construct_one(
            header, text,
            fixed_length=True if self.config.model_type == 'cnn' else False)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        if self.config.model_type == 'cnn':
            outputs = self.model(inputs)  # for CNN
        elif self.config.model_type == 'attn_gru_rnn':
            outputs = self.model(inputs, self.data.input_lang, seq_lengths)
        else:
            outputs = self.model(inputs, seq_lengths)

        _, predicted = torch.max(outputs.data, dim=1)

        return predicted.cpu().numpy().tolist() == targets.cpu().data.numpy().tolist()

def main():
    parser = MainArgParser()
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    config = Config(vars(args))

    data_set_names = {
        'DynaMed': 'DM',
        'ElsevierCecil': 'EC',
        'ElsevierText': 'ET',
        'NGC': 'NGC',
        'WikipediaMedical': 'WM',
        'WileyBooks': 'WB',
        'WileyCochrane': 'WC'
    }

    reload_data = config.reload_data

    if config.data_sets == ['All_Textbooks']:
        config.textbook_data_sets = list(data_set_names.keys())
    elif config.data_sets == ['EHR']:
        config.textbook_data_sets = []
    else:
        config.textbook_data_sets = config.data_sets

    experiment = Experiment(config=config, reload_data=reload_data)
    data_setup = {
        experiment.data.TEXTBOOK_TRAIN: config.textbook_data_ratio,
        experiment.data.i2b2_TRAIN: config.i2b2_data_ratio
    }

    if config.do_train:
        if config.reload_model_file:
            print("CONTINUING TO TRAIN")
            if torch.cuda.is_available():
                experiment.model = torch.load(config.reload_model_file)
            else:
                experiment.model = torch.load(config.reload_model_file, map_location='cpu')
            experiment.model.use_cuda = torch.cuda.is_available()
            experiment.log('Model restored.')
            experiment.epoch_start = 1
        experiment.train(learning_rate=config.learning_rate,
                         data_setup=data_setup,
                         save_model_dir=config.output_dir)
    if config.do_eval:
        if "ALL" in config.test_set or "MedLit" in config.test_set:
            print("RUNNING MedLit Dev")
            experiment.test(datasets=[experiment.data.TEXTBOOK_DEV], print_examples=True)
            print("RUNNING DEV")
        if "ALL" in config.test_set or "i2b2" in config.test_set:
            print("RUNNING i2b2 DEV")
            experiment.test(datasets=[experiment.data.i2b2_DEV], print_examples=True, print_test_results=True)
            print("RUNNING TEST")
            print("RUNNING i2b2 TEST")
            experiment.test(datasets=[experiment.data.i2b2_TEST], print_examples=True, print_test_results=True)


if __name__ == "__main__":
    main()
