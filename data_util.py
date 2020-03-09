
from collections import Counter

from lang import *
from argument_parsers import DataUtilArgParser
import pickle

try:
    from sets import Set
except ImportError:
    Set = set

import torch
from torch.autograd import Variable
import random


class DataUtil:
    def __init__(self, data_dir, vocab_dir='./vocab', split_by_sentence=True, split_by_line=False,
                 train_headers=True, skip_list=['Health Maintenance']):
        self.data_dir = data_dir
        self.vocab_dir = vocab_dir
        self.textbook_train_data = []
        self.textbook_dev_data = []
        self.i2b2_train_data = []
        self.i2b2_dev_data = []
        self.i2b2_test_data = []
        self.textbook_train_data_source = []
        self.textbook_dev_data_source = []
        self.TEXTBOOK_TRAIN = 'textbook_train'
        self.TEXTBOOK_DEV = 'textbook_dev'
        self.i2b2_TRAIN = 'i2b2_train'
        self.i2b2_DEV = 'i2b2_dev'
        self.i2b2_TEST = 'i2b2_test'
        self.skip_list = skip_list

        self.split_by_sentence = split_by_sentence
        self.split_by_line = split_by_line
        self.train_headers = train_headers  # if or not including headers as training data

        if vocab_dir is not None:
            self.init_dirs()
            self.input_lang = NaturalLang('lang', self.data_dir, self.vocab_dir)
            self.output_lang = Lang('group', self.data_dir, self.vocab_dir)

            import spacy
            self.nlp = spacy.load('en')

        self.max_seq_length = 250
        self.pretrained_embeddings = None

    def init_dirs(self):
        self.make_dir(self.data_dir)

        if self.vocab_dir is not None:
            self.make_dir(self.vocab_dir)
        new_dirs = ['split']
        for d in new_dirs:
            self.make_dir(self.data_dir + '/' + d)

    def make_dir(self, new_dir):
        if not exists(new_dir):
            os.makedirs(new_dir)

    def load_split_data(self):
        if len(os.listdir(self.data_dir + '/split')) == 0:
            raise Exception(self.data_dir + '/split is empty. Please reload data first!')

        if os.path.exists(self.data_dir + '/split/textbook_train.pkl'):
            self.textbook_train_data = pickle.load(open(self.data_dir + '/split/textbook_train.pkl', 'rb'))
        if os.path.exists(self.data_dir + "/split/textbook_dev.pkl"):
            self.textbook_dev_data = pickle.load(open(self.data_dir + '/split/textbook_dev.pkl', 'rb'))
        if os.path.exists(self.data_dir + "/split/i2b2_test.pkl"):
            self.i2b2_test_data = pickle.load(open(self.data_dir + '/split/i2b2_test.pkl', 'rb'))
        if os.path.exists(self.data_dir + "/split/i2b2_train.pkl"):
            self.i2b2_train_data = pickle.load(open(self.data_dir + '/split/i2b2_train.pkl', 'rb'))
        if os.path.exists(self.data_dir + '/split/i2b2_dev.pkl'):
            self.i2b2_dev_data = pickle.load(open(self.data_dir + '/split/i2b2_dev.pkl', 'rb'))


    def parse_ref_line(self, line):
        id, label, text_header, text_path, line_num = tuple([t.strip() for t in line.split('\t')])
        return id, label, text_header, text_path, line_num

    def get_last_section(self, header):
        if '/' not in header:
            return header
        last_section = header.split('/')[-2]
        return last_section

    def parse_line(self, line):
        tokens = line.split('\t')
        if len(tokens) == 2:
            header_path, text = tokens
            return header_path, text
        else:
            return tokens[0], ''

    def load_ref_data(self, src_path):
        results = []
        with open(src_path, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                results.append(self.parse_ref_line(line.strip()))
        return results

    def build_vocab(self, dataset, pretrain=True, load_file_embeddings=False):
        print("Building vocabulary ...")
        total = len(dataset)
        max_seq_length = 0
        for i, example in enumerate(dataset):
            sentence = example[1]
            category = example[2]
            self.input_lang.add_sentence(sentence, normalize=True)
            input_len = len(self.input_lang.tokenize(sentence))
            if input_len > max_seq_length:
                max_seq_length = input_len
            self.output_lang.add_word(category)
            print('\r', end='')
            print(i, '/', total, end='')
        print("Counted words:")
        print("Input:")
        print(self.input_lang.name, self.input_lang.n_words)
        print("Output:")
        print(self.output_lang.name, self.output_lang.n_words)
        print('output vocab:', self.output_lang.word2count)
        print('max_seq_length:', max_seq_length)

        if pretrain:
            if load_file_embeddings:
                with open(self.vocab_dir + '/pretrained_embeddings.json') as f:
                    self.pretrained_embeddings = json.load(f)
            else:
                self.pretrained_embeddings = self.input_lang.get_vocab_embeddings(
                    output_dir=self.vocab_dir)
                with open(self.vocab_dir + '/pretrained_embeddings.json', 'w') as f:
                    json.dump(self.pretrained_embeddings, f)
        self.save_vocab()

    def load_vocab(self):
        self.input_lang.load()
        self.output_lang.load()

    def save_vocab(self):
        self.make_dir(self.vocab_dir)
        with open(self.vocab_dir + '/input_lang_index2word.json', 'w') as f:
            json.dump(self.input_lang.index2word, f)
        with open(self.vocab_dir + '/output_lang_index2word.json', 'w') as f:
            json.dump(self.output_lang.index2word, f)

        # Save lang objects
        self.input_lang.save()
        self.output_lang.save()

        # Save pretrained embeddings
        with open(self.vocab_dir + '/embeddings.pkl', 'wb') as f:
            pickle.dump(self.pretrained_embeddings, f, pickle.HIGHEST_PROTOCOL)

    # def one_hot(self, size, indices):
    #     mask = torch.LongTensor(*size).zero_()
    #     indices = torch.unsqueeze(indices, 1)
    #     ones = torch.LongTensor(indices.size()).fill_(1)
    #     return mask.scatter_(1, indices, ones)

    def get_category_counts(self, data):
        counter = Counter()
        for d in data:
            counter[d[2]] += 1
        for d in data:
            counter[self.get_source_label(d[4])] += 1
        counter_string = json.dumps(counter, indent=4, sort_keys=True)
        return counter, counter_string

    def get_source_label(self, file_path):
        if file_path.lower().find("wikipediamedical") >= 0:
            return 'WikipediaMedical'  # web reference
        if file_path.lower().find("i2b2") >= 0:
            return "i2b2"
        print("Unknown Source: " + file_path)
        return ""

    def get_data_subset(self, data, ratio, sampling=False):
        assert(0 <= ratio <= 1)
        if ratio == 0:
            return []
        if ratio == 1:
            return data

        data_subset = []
        category2data = {}
        for d in data:
            if d[2] not in category2data:
                category2data[d[2]] = []
            category2data[d[2]].append(d)

        for category in category2data:
            category_data = category2data[category]
            number_samples = int(len(category2data[category]) * ratio)
            if number_samples < 1:
                number_samples = 1

            if sampling:
                data_subset += [category_data[i] for i in random.sample(range(len(category_data)), number_samples)]
            else:
                data_subset += category_data[:number_samples]
        return data_subset

    def get_data_subsets(self, data, ratio, sampling=False):
        assert(0 <= ratio <= 1)
        if ratio == 0:
            return [], data
        if ratio == 1:
            return data, []

        data_subset = []
        rest = []
        category2data = {}
        for d in data:
            if d[2] not in category2data:
                category2data[d[2]] = []
            category2data[d[2]].append(d)

        for category in category2data:
            category_data = category2data[category]
            number_samples = int(len(category2data[category]) * ratio)
            if number_samples < 1:
                number_samples = 1

            if sampling:
                raise NotImplementedError
                data_subset += [category_data[i] for i in random.sample(range(len(category_data)), number_samples)]
            else:
                data_subset += category_data[:number_samples]
                rest += category_data[number_samples:]
        return data_subset, rest

    def downsampling(self, data, number_samples=0):
        counter, _ = self.get_category_counts(data)
        if number_samples == 0:
            key_min = min(counter.keys(), key=(lambda k: counter[k]))
            print('key_min:', key_min)
            print('amount:', counter[key_min])
            number_samples = counter[key_min]

        if number_samples == -1:
            number_samples = int(sum(counter.values()) / len(counter))

        train_data = []
        for category in counter:
            category_data = []
            for d in data:
                if d[2] == category:
                    category_data.append(d)
            if len(category_data) <= number_samples:
                train_data += category_data
            else:
                train_data += [category_data[i] for i in random.sample(range(len(category_data)), number_samples)]
        return train_data

    def get_label_weight(self, data):
        weight = np.zeros(len(self.output_lang.index2word), dtype=np.float)
        for d in data:
            weight[self.output_lang.get_index(d[2])] += 1
        weight = torch.FloatTensor(len(data) / weight / 100.0)
        print('weight:', weight)
        return weight

    def construct_batch(self, start, end, data, fixed_length=False):
        batch = data[start:min(end, len(data))]

        input_seqs = []
        target_seqs = []
        for i in range(len(batch)):
            input_seqs.append(self.input_lang.indexes_from_sentence(batch[i][1]))
            label_index = self.output_lang.get_index(batch[i][2])
            if label_index == self.output_lang.PAD_token:
                print("UNSEEN: " + batch[i][2])
            target_seqs.append(self.output_lang.get_index(batch[i][2]))

        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(input_seqs, zip(batch, target_seqs)), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs_batch = zip(*seq_pairs)
        batch, target_seqs = zip(*target_seqs_batch)

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_seq_lengths = [len(s) for s in input_seqs]
        if fixed_length:
            max_len = self.max_seq_length + 1
        else:
            max_len = max(input_seq_lengths)
        input_padded = [self.input_lang.pad_seq(s, max_len) for s in input_seqs]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        with torch.no_grad():
            input_var = Variable(torch.LongTensor(input_padded))
            target_var = Variable(torch.LongTensor(target_seqs))
            input_seq_lengths = Variable(torch.LongTensor(input_seq_lengths))
        return input_var, input_seq_lengths, target_var, batch

    def construct_one(self, header, text, fixed_length=False):
        input_seqs = []
        target_seqs = []

        input_seqs.append(self.input_lang.indexes_from_sentence(text))
        label_index = self.output_lang.get_index(header)

        if label_index == self.output_lang.PAD_token:
            print("UNSEEN: " + header)
        target_seqs.append(self.output_lang.get_index(header))

        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_seq_lengths = [len(s) for s in input_seqs]
        if fixed_length:
            max_len = self.max_seq_length + 1
        else:
            max_len = max(input_seq_lengths)
        input_padded = [self.input_lang.pad_seq(s, max_len) for s in input_seqs]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        with torch.no_grad():
            input_var = Variable(torch.LongTensor(input_padded))
            target_var = Variable(torch.LongTensor(target_seqs))
            input_seq_lengths = Variable(torch.LongTensor(input_seq_lengths))
        return input_var, input_seq_lengths, target_var

    def getXY(self, data):
        input_seqs = []  # X
        target_seqs = []  # Y
        for i in range(len(data)):
            input_seqs.append(data[i][1])
            target_seqs.append(data[i][2])

        return input_seqs, target_seqs

    def get_dataset(self, dataset):
        if dataset == self.TEXTBOOK_TRAIN:
            if len(self.textbook_train_data) == 0:
                if os.path.exists(self.data_dir + "/split/textbook_train.pkl"):
                    self.textbook_train_data = pickle.load(open(self.data_dir + "/split/textbook_train.pkl", 'rb'))
            return self.textbook_train_data

        if dataset == self.TEXTBOOK_DEV:
            if len(self.textbook_dev_data) == 0:
                if os.path.exists(self.data_dir + "/split/textbook_dev.pkl"):
                    self.textbook_dev_data = pickle.load(open(self.data_dir + "/split/textbook_dev.pkl", 'rb'))
            return self.textbook_dev_data
        if dataset == self.i2b2_TEST:
            if len(self.i2b2_test_data) == 0:
                if os.path.exists(self.data_dir + "/split/i2b2_test.pkl"):
                    self.i2b2_test_data = pickle.load(open(self.data_dir + "/split/i2b2_test.pkl", 'rb'))
            return self.i2b2_test_data
        if dataset == self.i2b2_TRAIN:
            if len(self.i2b2_train_data) == 0:
                if os.path.exists(self.data_dir + "/split/i2b2_train.pkl"):
                    self.i2b2_train_data = pickle.load(open(self.data_dir + "/split/i2b2_train.pkl", 'rb'))
            return self.i2b2_train_data
        if dataset == self.i2b2_DEV:
            if len(self.i2b2_dev_data) == 0:
                if os.path.exists(self.data_dir + "/split/i2b2_dev.pkl"):
                    self.i2b2_dev_data = pickle.load(open(self.data_dir + "/split/i2b2_dev.pkl", 'rb'))
            return self.i2b2_dev_data
        return None

    def get_dataset_size(self, dataset):
        dataset = self.get_dataset(dataset)
        if dataset is not None:
            return len(dataset)
        return 0

    # def view_data(self, data, label, output_dir):
    #     view_data = [d for d in data if d[2] == label]
    #     size_sampling = 10000
    #     if size_sampling >= len(view_data):
    #         choices = range(len(view_data))
    #     else:
    #         choices = random.sample(range(len(view_data)), size_sampling)
    #     view_data = [str(view_data[i]) for i in choices]
    #     path = output_dir + '/view_data_%s.txt' % label.replace(' ', '')
    #     with open(path, 'w') as f:
    #         f.write('\n\n\n\n\n'.join(view_data))
    #     print("Written data to " + path)

    def load_textbook_train_dev_data(self, train_base_dir, dev_base_dir, save=True):
        train_file_path = train_base_dir + '/train_ref.txt'
        dev_file_path = dev_base_dir + '/dev_ref.txt'
        train_data_ref = self.load_ref_data(src_path=train_file_path)
        dev_data_ref = self.load_ref_data(src_path=dev_file_path)

        train_d = self.load_data_from_ref(train_data_ref)
        dev_d = self.load_data_from_ref(dev_data_ref)
        self.textbook_train_data += train_d
        self.textbook_dev_data += dev_d

        if save:
            with open(self.data_dir + '/split/textbook_train.pkl', 'wb') as f:
                pickle.dump(self.textbook_train_data, f)
            with open(self.data_dir + '/split/textbook_dev.pkl', 'wb') as f:
                pickle.dump(self.textbook_dev_data, f)
        print('Finished loading ...')

    def load_i2b2_train_data(self, train_base_dir, save=True):
        train_file_path = train_base_dir + '/train/train_ref.txt'
        train_data_ref = self.load_ref_data(src_path=train_file_path)

        self.i2b2_train_data = self.load_test_data_from_ref(train_data_ref)
        if save:
            with open(self.data_dir + '/split/i2b2_train.pkl', 'wb') as f:
                pickle.dump(self.i2b2_train_data, f)
        print('Finished loading ...')

    def load_test_data(self, ref_base_dir, save=True, type="test"):
        test_file_path = ref_base_dir + '/' + type + '/' + type + '_ref.txt'
        test_data_ref = self.load_ref_data(src_path=test_file_path)
        data = None

        if type == "test":
            print("i2b2 test")
            self.i2b2_test_data = self.load_test_data_from_ref(test_data_ref)
            data = self.i2b2_test_data
        else:
            print("i2b2 dev")
            self.i2b2_dev_data = self.load_test_data_from_ref(test_data_ref)
            data = self.i2b2_dev_data

        if save:
            with open(self.data_dir + '/split/i2b2_' + type + '.pkl', 'wb') as f:
                pickle.dump(data, f)

        print('Finished loading ...')

    def load_data_from_ref(self, ref_data):
        data = []

        total = len(ref_data)
        for i, ref in enumerate(ref_data):
            id, label, text_header, file_path, line_num = ref

            # some decisions about data:
            # default is to skip health maintenance
            skip = False
            for cat in self.skip_list:
                if label.lower() == cat.lower():
                    print("Skipping: " + str(ref) + " , (" + cat + ")")
                    skip = True
                    break
            if skip:
                continue
            if label.lower() == "history of present illness":
                label = "Chief Complaint"
            elif label.lower() == "medical history":
                label = "Past Medical History"
            elif label.lower() == "laboratory tests":
                label = "Findings"
            elif label.lower() == "na":
                print("Skipping" + str(ref))
                continue

            header_passage = self.get_header_passage(file_path, int(line_num))
            if header_passage is None:
                print(" Skipping No HP: " + str(ref) + "," + header_passage)
                continue

            header, passage = header_passage
            if self.split_by_sentence:
                # print('passage:', passage)
                sentences = self.tokenize_sentences_by_text(passage)
                # print('sentences:', sentences)
                for s in sentences:
                    s = s.strip()
                    if len(s) > 15 and len(s) < 400:
                        data.append((str(id) + "." + str(i), s, label, text_header, file_path, line_num))

                if self.train_headers:
                    header = self.get_last_section(header)
                    data.append((id, header, label, text_header, file_path, line_num))

            # split by section - here we need to get all the
            # sentences that are part of the same section and append them.
            else:
                if i == 0:
                    print("label:" + label)
                    print("passage: " + passage)
                    print("text header: " + text_header)

                if self.train_headers:
                    header = self.get_last_section(header)
                    passage = header + " : " + passage

                data.append((str(id) + "." + str(i), passage, label, text_header, file_path, line_num))

            print('\r', end='')
            print('%d / %d' % (i, total), end='')
        print("Collected:")
        print(len(data))
        return data

    def load_test_data_from_ref(self, ref_data):
        data = []
        total = len(ref_data)

        cur_passage = ""
        cur_text_header = ""
        cur_label = None
        cur_line_num = 0
        header = True

        for i, ref in enumerate(ref_data):
            # id, label , ...
            id, label, text_header, file_path, line_num = ref

            # some decisions about data:
            # default is to skip health maintenance
            skip = False
            for cat in self.skip_list:
                if label.lower() == cat.lower():
                    print("Skipping: " + str(ref) + " , (" + cat + ")")
                    skip = True
                    break
            if skip:
                continue
            if label.lower() == "history of present illness":
                label = "Chief Complaint"
            elif label.lower() == "medical history":
                label = "Past Medical History"
            elif label.lower() == "laboratory tests":
                label = "Findings"
            elif label.lower() == "na":
                continue

            if cur_label is None:
                cur_label = label
                cur_text_header = text_header

            header_passage = self.get_ehr_header_passage(file_path, int(line_num))
            if header_passage is None or len(header_passage[1].strip()) == 0:
                print("Skipping Header None: " + str(file_path) + " " + line_num)
                continue

            _, passage = header_passage

            passage = passage.strip()

            if len(passage) == 0:
                continue

            if self.split_by_sentence:
                data.append((id, passage, label, text_header, file_path, line_num))
                if i == 0:
                    print('sentence file_path:', file_path)
                    print('sentence line_num:', line_num)
                    print('sentence passage:', passage + '\n')
            else:
                # new section
                if label != cur_label:
                    if cur_passage != "":
                        data.append((id, cur_passage, cur_label, cur_text_header, file_path, cur_line_num))

                    # reset
                    if cur_line_num == 0:
                        print('section file_path:', file_path)
                        print('section line_num:', cur_line_num)
                        print('section passage:', cur_passage + '\n')
                    cur_line_num = line_num
                    cur_passage = ""
                    cur_text_header = text_header
                    cur_label = label
                    header = True
                if not self.train_headers and header:
                    # it won't catch everything, but to check for the cases where the header may be separated by a colon
                    reg = re.compile("[-:]")
                    if len(reg.split(passage)) > 1:
                        h, p = reg.split(passage, 1)
                        if p != "":
                            cur_passage += p + " "
                # skip first line (this is the header)
                else:
                    cur_passage += passage + " "
                header = False
            print('\r', end='')
            print('%d / %d' % (i, total), end='')
        print("Collected:")
        print(len(data))
        return data

    def tokenize_sentences_by_text(self, passage):
        # return [sent.text for sent in self.nlp(passage).sents]
        if '|.|' in passage:
            return passage.split('|.|')
        return passage.split('.')

    def get_header_passage(self, path, line_num):
        with open(path, 'r') as f:
            cur_line_num = 0
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                cur_line_num += 1
                if cur_line_num == line_num:
                    header, passage = self.parse_line(line)
                    return header, passage
        return None

    def get_ehr_header_passage(self, path, line_num):
        with open(path, 'r') as f:
            if self.split_by_line:
                for i, line in enumerate(f.readlines()):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if i + 1 == line_num:
                        passage = line
                        return None, passage
            else:
                sections = f.read().split('|.|')
                if len(sections) >= line_num:
                    passage = sections[line_num - 1]
                    return None, passage
        return None


def main():
    parser = DataUtilArgParser()
    args = parser.parse_args()

    if args.vocab_dir == None:
        args.vocab_dir == args.output_dir + "/vocab/"

    helper = DataUtil(data_dir=args.output_dir,
                      vocab_dir=args.vocab_dir, split_by_sentence=not args.split_by_section)

    print("Datasets: " + str(args.data_set))

    if "ALL" in args.data_set or "MedLit" in args.data_set:
        print('processing medlit...')
        textbook_data_sets = args.textbook_data_sets

        for ds in textbook_data_sets:
            if ds is "None":
                continue
            helper.load_textbook_train_dev_data(args.ref_data_dir + '/medlit/train/' + ds,
                                                args.ref_data_dir + '/medlit/dev/' + ds)
            print('medlit training data stats:')
            _, counter_string = helper.get_category_counts(helper.textbook_train_data)
            print(counter_string)

            print('\nmedlit test/dev data stats:')
            _, counter_string = helper.get_category_counts(helper.textbook_dev_data)
            print(counter_string)

    if "ALL" in args.data_set or "i2b2" in args.data_set:
        print("processing i2b2")
        # train
        helper.load_i2b2_train_data(train_base_dir=args.ref_data_dir + '/i2b2_ehr/')
        # test
        helper.load_test_data(ref_base_dir=args.ref_data_dir + '/i2b2_ehr/')
        # dev
        helper.load_test_data(ref_base_dir=args.ref_data_dir + '/i2b2_ehr/', type='dev')

        print('\ni2b2 ehr dev data stats:')
        _, counter_string = helper.get_category_counts(helper.i2b2_dev_data)
        print(counter_string)

        helper.load_test_data(ref_base_dir=args.ref_data_dir + '/i2b2_ehr/', type='dev2')

        print('\ni2b2 ehr dev2 data stats:')
        _, counter_string = helper.get_category_counts(helper.i2b2_dev_data)
        print(counter_string)


        print('i2b2 training data stats:')
        _, counter_string = helper.get_category_counts(helper.i2b2_train_data)
        print(counter_string)

        print('\ni2b2 test data stats:')
        _, counter_string = helper.get_category_counts(helper.i2b2_test_data)
        print(counter_string)

    print('finished')

    helper.build_vocab(helper.textbook_train_data, pretrain=False)


if __name__ == '__main__':
    main()
