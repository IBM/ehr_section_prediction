
from os import listdir, makedirs, mkdir
from os.path import isfile, join, exists, dirname

from collections import Counter

try:
    from sets import Set
except ImportError:
    Set = set

import json

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from argument_parsers import *
from main import Experiment
import csv
import xml.etree.ElementTree


def get_last_section(header):
    if '/' not in header:
        return header
    last_section = header.split('/')[-2]
    return last_section


def parse_line(line):
    tokens = line.split('\t')
    if len(tokens) == 2:
        header_path, text = tokens
        return header_path, text
    else:
        return tokens[0], ''


def get_offsets(file):
    # open annotations
    with open(file) as f:
        annotations = f.readlines()
        annotations.reverse()
        offsets = []
        for line in annotations:
            _, annotation, category = line.strip().split('\t')
            _, start, end = annotation.split(" ")

            offsets.append(start + "\t" + category)
    return offsets


def make_dir(new_dir):
    if not exists(new_dir):
        makedirs(new_dir)


def read_header_vocab(path):
    headers = {}
    groups = set()
    acronyms = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            # skip commented out lines
            if line.startswith("#"):
                continue
            group, header, acronym = line.split('|')
            headers[header.strip()] = group.strip()
            if acronym.strip() == "Y":
                acronyms.add(acronym)
            groups.add(group.strip())
    return headers, groups, acronyms


def count_overlapped_tokens(seq1, seq2):
    return len(set(seq1) & set(seq2))


class HeaderDataUtil(object):
    def __init__(self, config, ref_output_dir=''):
        self.ref_output_dir = ref_output_dir
        self.config = config

        if exists(self.ref_output_dir):
            print("Warning! Output Directory already exists!")
        make_dir(self.ref_output_dir)

        self.data_cnt = 0
        self.nlp = spacy.load('en')
        self.stop_words = STOP_WORDS
        self.punctuations = string.punctuation
        self.header2group, self.headers, acronyms = read_header_vocab(config.medlit_headers_file_path)

        self.stemmer = PorterStemmer()
        self.stemmedheader2group = {}

        for header in self.header2group:
            if header in acronyms:
                self.stemmedheader2group[header] = [header]
            else:
                bow = word_tokenize(header.lower())
                bow = [self.stemmer.stem(token) for token in bow]
                self.stemmedheader2group[header] = bow
        print(self.stemmedheader2group)

    def filter_file(self, dataset, text_path, data_usage, threshold):
        with open(text_path, 'r') as text_file:
            text_lines = text_file.readlines()
            for k, line in enumerate(text_lines):
                groups = set()
                line = line.strip()
                if len(line) == 0:
                    continue
                text_header_path, text_passage = parse_line(line)
                text_last_section = get_last_section(text_header_path)
                if len(line) == 0:
                    continue
                header_path, _ = parse_line(line)
                last_section = get_last_section(header_path)
                if len(last_section.strip()) == 0:
                    print([header_path, last_section])

                tokens = word_tokenize(text_last_section.lower())
                tokens = [self.stemmer.stem(token) for token in tokens]
                for target_header in self.stemmedheader2group:
                    # target words
                    bow = self.stemmedheader2group[target_header]
                    # num words in header that match target
                    cnt = count_overlapped_tokens(tokens, bow)

                    # for categories with a lot of data, be more strict and require exact match.
                    # for categories with less data, allow extra words
                    rare_cats = {"Laboratory Tests", "Allergies", "History of Present Illness", "Family History",
                                 "Personal and Social history"}
                    group = self.header2group[target_header]

                    # already found!
                    if group in groups:
                        continue

                    # first is how many of words in header, second is how many words in target
                    # (second needs to be EXACT)
                    if ((group not in rare_cats and cnt / len(tokens) >= threshold) or
                        (group in rare_cats and cnt / len(tokens) >= .5)) and cnt / len(bow) >= threshold:
                        groups.add(group)
                        open_mode = 'w' if not exists(
                            self.ref_output_dir + '/' + dataset + '/' + data_usage + '_ref.txt') else 'a'
                        if open_mode == 'w':
                            mkdir(self.ref_output_dir + '/' + dataset + '/')
                        with open(self.ref_output_dir + '/' + dataset + '/' + data_usage + '_ref.txt',
                                  open_mode) as outfile:
                            label = group
                            data_line = '\t'.join(
                                [str(self.data_cnt), label, text_last_section, text_path, str(k+1)])
                            outfile.write(data_line + '\n')
                            self.data_cnt += 1

    def load_textbook_corpus(self, ds, config, data_usage):
        # for ds in datasets:
        print('Processing dataset:', ds)
        # path = self.base_dir + '/' + ds
        text_path = config.data_dir + "/" + ds + "/" + data_usage

        if not exists(text_path):
            print(text_path, 'does NOT exist!')
            return

        paths = [join(text_path, f) for f in listdir(text_path) if isfile(join(text_path, f))]
        total_files = len(paths)
        print("Processing: " + str(total_files) + " files")
        for i, text_path in enumerate(paths):
            print_every = 500
            if total_files > 2000:
                print_every = round(total_files / 10)
            if i % print_every == 0:
                print(str(i) + "/" + str(total_files))
            self.filter_file(ds, text_path, data_usage, config.threshold)

    def read_i2b2_annotation_file(self, target_ids, file_paths, output_path, output_dir, annotation_path):
        make_dir(dirname(output_path))
        count = 0

        labels_file = "./data/unique_labels.csv"

        labels = {}

        with open(labels_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                labels[row[0].strip()] = row[1]

        with open(output_path, 'w') as annotation_file:
            i = 0

            for patient_id in target_ids:
                for file_path in file_paths:
                    if os.path.isfile(file_path + '/' + patient_id):
                        file_name = file_path + "/" + patient_id

                e = xml.etree.ElementTree.parse(file_name).getroot()
                text = e.findall('TEXT')[0].text

                if os.path.isfile(annotation_path + '/Set1/' + patient_id[:-3] + "ann"):
                    annotation_path_dir = annotation_path + '/Set1'
                elif os.path.isfile(annotation_path + '/Set2/' + patient_id[:-3] + "ann"):
                    annotation_path_dir = annotation_path + '/Set2'
                else:
                    annotation_path_dir = annotation_path + '/Test'

                with open(output_dir + "/" + patient_id[:-3] + "txt", 'w') as outfile:
                    offsets = get_offsets(annotation_path_dir + "/" + patient_id[:-3] + "ann")
                    offsets.append(str(len(text)) + "\tNA")

                    j = 0

                    for i in range(len(offsets) - 1):
                        start, category = offsets[i].split("\t")
                        end, _ = offsets[i + 1].split("\t")

                        passage = text[int(start):int(end)]

                        sentences = sent_tokenize(passage)

                        for sentence in sentences:
                            outfile.write(sentence.replace("\n", " ") + "|.|")
                            if category.strip() not in labels:
                                print(category)
                            else:
                                annotation_file.write(str(count) + "\t" + labels[category.strip()].replace('\n', ' ') +
                                                      "\t" + category + "\t" + output_dir + "/" + patient_id[:-3] +
                                                      "txt" + "\t" + str(j + 1) + "\n")
                            j += 1
                            count += 1


def load_i2b2_corpus(config, load_test=False):
    data_path = config.data_dir
    annotation_path = config.annotation_dir

    util = HeaderDataUtil(config=config, ref_output_dir=config.output_dir + '/i2b2_ehr/')

    if not load_test:
        train = data_path + "/training-RiskFactors-Complete-Set1"
        dev = data_path + "/training-RiskFactors-Complete-Set2"
        paths = [train, dev]

        target_ids = [f for f in listdir(train)]
        target_ids.extend([f for f in listdir(dev)])

        length = round(len(target_ids) / 3)
        third = length
        two_third = length*2

        print(len(target_ids))

        print("0 : " + str(third))
        util.read_i2b2_annotation_file(
            target_ids=target_ids[0:third],
            file_paths=paths,
            output_path=config.output_dir + '/i2b2_ehr/train/train_ref.txt',
            output_dir=config.output_dir + '/i2b2_ehr/train/',
            annotation_path=annotation_path + "/Section-Heading Recognition Corpus/")

        print(str(third) + ":" + str(two_third))
        util.read_i2b2_annotation_file(
            target_ids=target_ids[third:two_third],
            file_paths=paths,
            output_path=config.output_dir + '/i2b2_ehr//dev/dev_ref.txt',
            output_dir=config.output_dir + '/i2b2_ehr/dev/',
            annotation_path=annotation_path + "/Section-Heading Recognition Corpus/")

        print(str(two_third) + ":" + str(len(target_ids)))
        util.read_i2b2_annotation_file(
            target_ids=target_ids[two_third:],
            file_paths=paths,
            output_path=config.output_dir + '/i2b2_ehr/dev2/dev2_ref.txt',
            output_dir=config.output_dir + '/i2b2_ehr/dev2/',
            annotation_path=annotation_path + "/Section-Heading Recognition Corpus/")
    else:
        test = data_path + "/testing-RiskFactors-Complete"

        paths = [test]

        target_ids = [f for f in listdir(test)]
        length = len(target_ids)
        print(len(target_ids))

        print("0 : " + str(length))
        util.read_i2b2_annotation_file(
            target_ids=target_ids[0:length],
            file_paths=paths,
            output_path=config.output_dir + '/i2b2_ehr/test/test_ref.txt',
            output_dir=config.output_dir + '/i2b2_ehr/test/',
            annotation_path=annotation_path + "/Section-Heading Recognition Corpus/")


def load_textbook_corpus(config, load_dev=False):
    textbook_data_sets = ['WikipediaMedical']
    print('textbook_data_sets:', textbook_data_sets)

    if load_dev:
        usage = "dev"
    else:
        usage = "train"

    output_dir = config.output_dir + '/medlit/' + usage + "/"

    util = HeaderDataUtil(config=config, ref_output_dir=output_dir)

    for ds in textbook_data_sets:
        util.load_textbook_corpus(ds, config, usage)


def main():
    parser = HeaderDataUtilArgParser()
    args = parser.parse_args()
    config = Config(vars(args))

    source_set = set(config.sources)

    load_textbooks = False
    load_i2b2_ehrs = False

    if 'MedLit' in source_set:
        load_textbooks = True
    if 'i2b2' in source_set:
        load_i2b2_ehrs = True

    if load_textbooks:
        load_textbook_corpus(config, load_dev=False)
        load_textbook_corpus(config, load_dev=True)

    if load_i2b2_ehrs:
        load_i2b2_corpus(config, load_test=False)
        load_i2b2_corpus(config, load_test=True)


if __name__ == "__main__":
    main()

