
import argparse
import os


class BaseArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(BaseArgParser, self).__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse_args(self):
        args = super(BaseArgParser, self).parse_args()
        # assert(args.global_output_dir != '')
        return args


class HeaderDataUtilArgParser(BaseArgParser):
    def __init__(self):
        super(HeaderDataUtilArgParser, self).__init__()
        self.add_argument('--output_dir', type=str, required=True,
                          help='Set output directory for formatted data')
        self.add_argument('--data_dir', type=str,
                          help='Path to data directory')
        self.add_argument('--annotation_dir', type=str,
                          help='Path to annotation directory (for i2b2)')
        self.add_argument('--medlit_headers_file_path', type=str,
                          default='./data/headers_MedLit.txt',
                          help='Absolute path of header file')
        self.add_argument('--textbook_data_sets', type=str, nargs='+', default=['WikipediaMedical'],
                          choices=['WikipediaMedical'],
                          help='data sets, one or more from [WikipediaMedical]')
        self.add_argument('--threshold', type=float, default=1,
                          help='threshold to adjust matching overlap for finding headers. 1 indicates exact match')
        self.add_argument('--sources', type=str, nargs='+', default=['MedLit,i2b2'],
                          choices=['MedLit', 'i2b2'], help='choose which sources to generate')

class DataUtilArgParser(BaseArgParser):
    def __init__(self):
        super(DataUtilArgParser, self).__init__()
        self.add_argument('--ref_data_dir', type=str, required=True,
                          help='Absolute dir path of reference data')
        self.add_argument('--output_dir', type=str, default='data/', required=True,
                          help='Absolute dir to output data.')
        self.add_argument('--vocab_dir', type=str, required=True,
                          help='Absolute dir to vocabulary files')
        self.add_argument('--textbook_data_sets', type=str, nargs='+', default=['WikipediaMedical'],
                          choices=['WikipediaMedical', 'None'],
                          help='data sets, one or more from [WikipediaMedical, None]'),
        self.add_argument('--split_by_section', action='store_true', default=False,
                          help="Split the data by sentence (default) or section"),
        self.add_argument('--data_set', type=str,  nargs='+', default=['ALL'], choices=['ALL', 'i2b2', 'MedLit'], help="Data to process: ALL , MedLit, i2b2")

class MainArgParser(BaseArgParser):
    def __init__(self):
        super(MainArgParser, self).__init__()
        self.add_argument('--output_dir', type=str, required=True,
                          help='Set global output directory')
        self.add_argument('--data_dir', type=str,
                          default=None,
                          help='Absolute dir path to ref (formatted) data dir')
        self.add_argument('--vocab_dir', type=str, required=True,
                          help='Absolute dir to vocabulary files')
        self.add_argument('--reload_data', action='store_true', default=False,
                          help="Reload the training data and store in pkl files")
        self.add_argument("--do_train",
                          action='store_true',
                          help="Whether to run training.")
        self.add_argument("--do_eval",
                          action='store_true',
                          help="Whether to run eval on the dev set.")
        self.add_argument('--reload_model_file', type=str, default='',
                          help='Absolute path of model file to reload for training')
        self.add_argument('--model_type', type=str, default='attn_gru_rnn',
                          choices=['cnn', 'lstm_rnn', 'attn_lstm_rnn', 'gru_rnn', 'attn_gru_rnn'],
                          help='Model type (cnn, lstm_rnn, attn_lstm_rnn, gru_rnn, attn_gru_rnn)')
        self.add_argument('--embedding_size', type=int, default=300,
                          help='Embedding size of embedding matrix')
        self.add_argument('--optimizer', type=str, default='adam',
                          choices=['sgd', 'adam'],
                          help='Optimizer (sgd, adam)')
        self.add_argument('--data_sets', type=str, nargs='+', default=['WikipediaMedical'],
                          choices=['WikipediaMedical', 'EHR'],
                          help='data sets, one or more from [WikipediaMedical, EHR]')
        self.add_argument('--textbook_data_ratio', type=float, default=1.0,
                          help='Ratio of selected textbook data to train')
        self.add_argument('--i2b2_data_ratio', type=float, default=1.0,
                          help='Ratio of i2b2 data to train')
        self.add_argument('--batch_size', type=int, default=32,
                          help='Batch_size for training')
        self.add_argument('--dropout', type=float, default=0.05,
                          help='Dropout probability')
        self.add_argument('--learning_rate', type=float, default=0.001,
                          help='Learning rate')
        self.add_argument('--weighted_loss', action='store_true', default=False,
                          help='whether or not train on weighted loss')
        self.add_argument('--downsampling', action='store_true', default=False,
                          help='whether or not do downsampling. If true, set downsampling_size accordingly')
        self.add_argument('--downsampling_size', type=int, default=5000,
                          help='Downsampling size. If 0, use the min size of all categories as upper bound. '
                               'If -1, use the averaged size of all categories as upper bound.'
                               'If others, use the set number as upper bound.')
        self.add_argument('--transfer_learning', action='store_true', default=False,
                          help='Transfer Learning for CNN')
        self.add_argument('--reuse_embedding_layer_only', action='store_true', default=False,
                          help='Transfer Learning for CNN: whether or not reuse embedding layer only')
        self.add_argument('--reuse_embedding_conv1_layers', action='store_true', default=False,
                          help='Transfer Learning for CNN: whether or not reuse embedding and conv1 layer')
        self.add_argument('--save_model', action='store_true', default=False,
                          help='whether or not save trained model every epoch')
        self.add_argument('--num_train_epochs', type=int, default=50,
                          help='Number of training epochs')
        self.add_argument('--filtered', action='store_true', default=False,
                         help='whether to filter the data based on ones where source is easily predictable')
        self.add_argument('--tuning_set', type=str,  nargs='+', default=['ALL'], choices=['ALL', 'i2b2', 'MedLit'], help="Data to tune against: ALL , MedLit, i2b2")
        self.add_argument('--test_set', type=str, nargs='+', default=['ALL'], choices=['ALL', 'i2b2', 'MedLit'],
                          help="Data to test against: ALL , MedLit, i2b2")
        self.add_argument('--split_by_section', action='store_true', default=False,
                          help="Split the data by sentence (default) or section")
        self.add_argument('--skip_list', type=str, nargs='+', default=['Health Maintenance'],
                          choices=['Health Maintenance', 'Personal and Social History', 'Past Medical History',
                                   'Medications', 'Examination', 'History of Present Illness', 'Findings', 'Procedures',
                                   'Family History', 'Review of Systems', 'Allergies', 'Assessment and Plan',
                                   'Laboratory Tests'], help="Categories to skip: 'Health Maintenance', 'Personal and"
                                                             " Social History', 'Past Medical History', ...")
        self.add_argument('--cuda', type=int, default=0, help="pick which cuda to use (0 or 1)")


class BertArgParser(BaseArgParser):
    def __init__(self):
        super(BertArgParser, self).__init__()

        ## Required parameters
        self.add_argument("--bert_model", default=None, type=str, required=True,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                                 "bert-base-multilingual-cased, bert-base-chinese.")
        self.add_argument("--output_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        self.add_argument("--data_dir",
                          default=None,
                          type=str,
                          required=True,
                          help="The directory where the data is stored.")
        self.add_argument("--vocab_dir",
                          default=None,
                          type=str,
                          required=True,
                          help="The directory where the vocabs are stored.")
        self.add_argument("--ref_data_dir",
                          default=None,
                          type=str,
                          help="The directory where the ref data is stored.")

        ## Other parameters
        self.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        self.add_argument("--do_train",
                            action='store_true',
                            help="Whether to run training.")
        self.add_argument("--do_eval",
                            action='store_true',
                            help="Whether to run eval on the dev set.")
        self.add_argument("--do_lower_case",
                            action='store_true',
                            help="Set this flag if you are using an uncased model.")
        self.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Total batch size for training.")
        self.add_argument("--eval_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for eval.")
        self.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        self.add_argument("--num_train_epochs",
                            default=3.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        self.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        self.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether not to use CUDA when available")
        self.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        self.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        self.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        self.add_argument('--fp16',
                            action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        self.add_argument('--loss_scale',
                            type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        self.add_argument('--data_sets', type=str, nargs='+', default=['WikipediaMedical'],
                          choices=['WikipediaMedical', 'EHR'],
                          help='data sets, one or more from [WikipediaMedical, EHR]')
        self.add_argument('--split_by_section', action='store_true', default=False,
                          help="Split the data by sentence (default) or section")
        self.add_argument('--skip_list', type=str, nargs='+', default=['Health Maintenance'],
                          choices=['Health Maintenance', 'Personal and Social History', 'Past Medical History',
                                   'Medications', 'Examination', 'History of Present Illness', 'Findings', 'Procedures',
                                   'Family History', 'Review of Systems', 'Allergies', 'Assessment and Plan',
                                   'Laboratory Tests'], help="Categories to skip: 'Health Maintenance', 'Personal and"
                                                             " Social History', 'Past Medical History', ...")
        self.add_argument('--reload_data', action='store_true', default=False,
                          help="Reload the training data and store in pkl files")
        self.add_argument('--train_data', type=str,  nargs='+', default=['ALL'], choices=['ALL', 'i2b2', 'MedLit'], help="Data to use for training: ALL , MedLit, i2b2")
        self.add_argument('--tuning_set', type=str,  nargs='+', default=['ALL'], choices=['ALL', 'i2b2', 'MedLit'], help="Data to tune against: ALL , MedLit, i2b2")
        self.add_argument('--i2b2_data_ratio', type=float, default=1.0,
                          help='Ratio of i2b2 data to train')
        self.add_argument('--test_set', type=str, nargs='+', default=['ALL'], choices=['ALL', 'i2b2', 'MedLit'],
                          help="Data to test against: ALL , MedLit, i2b2")


class Config(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, 'option.txt'), 'w') as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write('%s, %s\n' % (key, str(value)))


