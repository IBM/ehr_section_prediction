
from argument_parsers import *
import torch
import logging
import random
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from data_util import DataUtil
from time import strftime, gmtime, localtime

parser = BertArgParser()
args = parser.parse_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
os.makedirs(args.output_dir, exist_ok=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
log_file_name = strftime("log_%Y_%m_%d_%H_%M_%S", localtime())
fh = logging.FileHandler(args.output_dir + '/%s.txt' % log_file_name)
logger = logging.getLogger(__name__)
logger.addHandler(fh)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example[1])

        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)
        #     # Modifies `tokens_a` and `tokens_b` in place so that the total
        #     # length is less than the specified length.
        #     # Account for [CLS], [SEP], [SEP] with "- 3"
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # else:
            # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example[2]]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("source: %s" % (example[4]))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example[2], label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def prf(out, labels, label_set):
    outputs = np.argmax(out, axis=1)
    prf = precision_recall_fscore_support(labels, outputs)
    micro = precision_recall_fscore_support(labels, outputs, average='micro')
    report = classification_report(labels, outputs, target_names=label_set, labels=[0,1,2,3,4,5,6,7,8,9,10,11], digits=4)
    # logger.info(confusion_matrix(labels, outputs, labels=[0,1,2,3,4,5,6,7,8,9,10,11]))
    return prf, micro, report

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def run_eval(args, model, device, eval_examples, eval_features, loss, global_step, name, label_list,
             print_examples=False, save_results=True):
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits = []
    all_label_ids = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
        all_logits.extend(logits)
        all_label_ids.extend(label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    if print_examples:
        examples = []
        for idx, d in enumerate(eval_examples):
            sorted_logits = np.argsort(all_logits[idx])

            examples.append([d[0], d[1].replace('\r', ' ').replace('\n', ' ').replace('\t', ' '), d[2], d[3], str(d[4]),
                             str(d[5]), label_list[sorted_logits[len(label_list) - 1]],
                             label_list[sorted_logits[len(label_list) - 2]],
                             label_list[sorted_logits[len(label_list) - 3]]])
            example_file = os.path.join(args.output_dir, name + "_examples.txt")
        with open(example_file, "w") as writer:
            logger.info("Save examples to: " + example_file)
            with open(example_file, 'w') as f:
                f.write(
                    "#\tSentence\tTrue\tHeader String\tLocation\tLine\tPrediction 1\tPrediction 2\tPrediction 3\n")
                for e in examples:
                    f.write('\t'.join(e) + '\n')

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    eval_prf, micro, report = prf(all_logits, all_label_ids, label_list)

    if save_results:
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_prf': eval_prf,
                  'micro_prf': micro,
                  'report': report,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(args.output_dir, name + "_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    else:
        logger.info(report)

    return micro[2]

def main():

    data_set_names = {
        'WikipediaMedical': 'WM',
    }

    if args.data_sets == ['EHR']:
        args.textbook_data_sets = []
    else:
        args.textbook_data_sets = args.data_sets

    # if len(args.textbook_data_sets) == 0:
    #     base_dir = args.global_dir + '/' + '_'.join(args.data_sets)
    # else:
    #     base_dir = args.global_dir + '/' + '_'.join([data_set_names[ds] for ds in args.textbook_data_sets])

    data = DataUtil(data_dir=args.data_dir, vocab_dir=args.vocab_dir,
                    split_by_sentence=not args.split_by_section, skip_list=args.skip_list)
    if args.reload_data:
        # if self.config.textbook_data_ratio > 0:
        for ds in args.textbook_data_sets:
            data.load_textbook_train_dev_data(
                args.ref_data_dir + 'medlit/' + args.data_type + '/train/' + ds,
                args.ref_data_dir + 'medlit/' + args.data_type + '/dev/' + ds)
        # train
        data.load_i2b2_train_data(train_base_dir=args.ref_data_dir + 'i2b2_ehr/' + args.data_type)
        # test
        data.load_test_data(ref_base_dir=args.ref_data_dir + 'i2b2_ehr/' + args.data_type, i2b2=True)
        # dev
        data.load_test_data(ref_base_dir=args.ref_data_dir + 'i2b2_ehr/' + args.data_type, i2b2=True,
                                 type='dev')

    else:
        data.load_split_data()

    logger.info("MedLit Training data: " + str(len(data.textbook_train_data)))
    logger.info("MedLit Dev data: " + str(len(data.textbook_dev_data)))
    logger.info("i2b2 Training data: " + str(len(data.i2b2_train_data)))
    logger.info("i2b2 Dev data: " + str(len(data.i2b2_dev_data)))
    logger.info("i2b2 Test data: " + str(len(data.i2b2_test_data)))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # task_name = args.task_name.lower()
    #
    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % (task_name))

    #processor = processors[task_name]()
    num_labels = 11
    label_list = ['Allergies', 'Assessment and Plan', 'Chief Complaint', 'Examination', 'Family History', 'Findings',
                  'Medications', 'Past Medical History', 'Personal and Social history', 'Procedures',
                  'Review of Systems']

    logger.info("Num Labels: " + str(num_labels))
    logger.info("Labels: " + str(label_list))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = []
    num_train_steps = None
    if args.do_train:
        for data_name in args.train_data:
            if data_name == "i2b2" or data_name == "ALL":
                if args.i2b2_data_ratio != 1:
                    train_examples.extend(data.get_data_subset(data.i2b2_train_data, args.i2b2_data_ratio))
                else:
                    train_examples.extend(data.i2b2_train_data)
            if data_name == "MedLit" or data_name == "ALL":
                train_examples.extend(data.textbook_train_data)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    logger.info("Combined Train data: " + str(len(train_examples)))

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank),
                                                          num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    best_f1 = 0
    best_model = model
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num train examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        dev_examples = []
        name = ""
        for data_name in args.tuning_set:
            name += data_name + "_"
            if data_name == "i2b2" or data_name == "ALL":
                random.shuffle(data.i2b2_dev_data)
                dev_examples.extend(data.i2b2_dev_data[:500])
            if data_name == "MedLit" or data_name == "ALL":
                random.shuffle(data.textbook_dev_data)
                dev_examples.extend(data.textbook_dev_data[:500])
        dev_features = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer)

        logger.info(" Num dev examples: " + str(len(dev_examples)))

        logger.info("EVAL on Pretrained model only: " + args.bert_model)
        run_eval(args, model, device, dev_examples, dev_features, 0, global_step,
                 name, label_list, save_results=False)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            loss = tr_loss / nb_tr_steps
            f1 = run_eval(args, model, device, dev_examples, dev_features, loss, global_step,
                          name, label_list, save_results=False)
            logger.info(str(epoch) + "/" + str(args.num_train_epochs) + ". loss: " + str(loss) + ", F1: " + str(f1))
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                output_model_file = os.path.join(args.output_dir, "pytorch_model" + str(epoch) + ".bin")
                logger.info("Saving best model with F1: " + str(best_f1))
                model_to_save = best_model.module if hasattr(best_model,
                                                        'module') else best_model  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)

    # Save a trained model
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    if args.do_train:
        logger.info("Saving best model with F1: " + str(best_f1))
        model_to_save = best_model.module if hasattr(best_model, 'module') else best_model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
    else:
        model_state_dict = torch.load(os.path.join(args.bert_model, "pytorch_model.bin"))
        best_model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                              num_labels=num_labels)
    best_model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        loss = tr_loss / nb_tr_steps if args.do_train else None

        if "ALL" in args.test_set or "MedLit" in args.test_set:
            eval_examples = data.textbook_dev_data
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)

            run_eval(args, best_model, device, eval_examples, eval_features, loss, global_step, "medlit_dev",
                     label_list, print_examples=True)

        if "ALL" in args.test_set or "i2b2" in args.test_set:
            eval_examples = data.i2b2_test_data
            eval_features = convert_examples_to_features(
               eval_examples, label_list, args.max_seq_length, tokenizer)

            run_eval(args, best_model, device, eval_examples, eval_features, loss, global_step, "i2b2_test",
                    label_list, print_examples=True)

if __name__ == "__main__":
    main()
