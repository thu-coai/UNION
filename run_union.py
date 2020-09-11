# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import sys
import union_modeling as modeling
# import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
import time
import operator
from functools import reduce
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
# flags.DEFINE_string("data_dir", "./Data/ROCStories","The input data dir.")
flags.DEFINE_string("data_dir", "./Data/WritingPrompts","The input data dir.")
# flags.DEFINE_string("task_name", "train", "The name of the task.")
flags.DEFINE_string("task_name", "pred", "The name of the task.")
flags.DEFINE_string("output_dir", "./model/output","The output directory where the model checkpoints will be written.")
flags.DEFINE_boolean("use_reconstruction", False, "Whether to use original bert")
# flags.DEFINE_string("init_checkpoint", "./model/uncased_L-12_H-768_A-12/bert_model.ckpt","Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("init_checkpoint", "./model/union_wp/union_wp","Initial checkpoint (usually from a pre-trained BERT model).")
if FLAGS.task_name == "train":
  flags.DEFINE_boolean("do_train", True, "Whether to run training.")
  flags.DEFINE_boolean("do_eval", True, "Whether to run eval on the dev set.")
  flags.DEFINE_boolean("do_predict", False, "Whether to run the model in inference mode on the test set.")
else:
  flags.DEFINE_boolean("do_train", False, "Whether to run training.")
  flags.DEFINE_boolean("do_eval", False, "Whether to run eval on the dev set.")
  flags.DEFINE_boolean("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 10, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

## Other parameters
flags.DEFINE_string("bert_config_file", "./model/uncased_L-12_H-768_A-12/bert_config.json", "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
flags.DEFINE_string("vocab_file", "./model/uncased_L-12_H-768_A-12/vocab.txt","The vocabulary file that the BERT model was trained on.")
flags.DEFINE_boolean("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
flags.DEFINE_integer("max_seq_length", 200, "The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 100.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
tf.flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
tf.flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, ref=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
      ref: (Optional) string. The reference for reconstruction task.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.ref = ref if FLAGS.use_reconstruction else None

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               ref_input_ids,
               ref_input_mask,               
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.ref_input_ids = ref_input_ids
    self.ref_input_mask = ref_input_mask    
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

class GenStoryProcessor(DataProcessor):
  def get_test_examples(self, data_dir):
    """See base class."""
    eval_story = []
    with tf.gfile.Open(os.path.join(data_dir, "ant_data/ant_data.txt"), "r") as fin:
        for line in fin:
            tmp = line.strip().split("|||")
            true_st_id = list(map(int, tmp[0].strip().split("///")))
            eval_st = tmp[1].strip()
            score = list(map(float, tmp[2].strip().split()))
            eval_story.append({"true_st": true_st_id, "st":eval_st, "score":score, "human":np.mean(score)})
    example = self._create_examples([st["st"] for st in eval_story], "test")
    return example
  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the testing sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line)
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=0, ref=None))
    return examples

class StoryClassiferProcessor(DataProcessor):
  def __init__(self):
    self.name_list = ["human"] +["negative"]
  def _read_st(self, input_file):
    def _read_line(fin):
      story, tmp = [], []
      for k, line in enumerate(fin):
          i = k + 1
          if i % 6 == 0:
            story.append(tmp)
            tmp = []
          else:
            tmp.append(line.strip())
      return story
    with tf.gfile.Open(input_file+".txt", "r") as fin:
      if "ROCStories" in FLAGS.data_dir:
        st = _read_line(fin)
      else:
        st = [[s.strip()] for s in fin.readlines()]
    if "human" in input_file:
      st_label = [1 for _ in range(len(st))]
    else:
      st_label = [0 for _ in range(len(st))]
    if FLAGS.use_reconstruction:
      ref_file = "_".join(input_file.split("_")[:-1] + ["human"])
      with tf.gfile.Open(ref_file+".txt", "r") as fin:
        if "ROCStories" in FLAGS.data_dir:
          st_ref_0 = _read_line(fin)
        else:
          st_ref_0 = [[s.strip()] for s in fin.readlines()]
        st_ref = []
        for sref in st_ref_0:
          for _ in range(int(len(st)/len(st_ref_0))):
            st_ref.append(sref)
    else:
      st_ref = [[None] for _ in range(len(st))]
    return [{"story":s, "label":l, "ref": ref} for s, l, ref in zip(st, st_label, st_ref)]

  def get_train_examples(self, data_dir):
    """See base class."""
    example = []
    for f in self.name_list:
      example += self._create_examples(
            self._read_st(os.path.join(data_dir, "train_data/train_%s"%f)), "train-%s"%f)
    np.random.shuffle(example)
    print("train:", len(example))
    return example

  def get_dev_examples(self, data_dir):
    """See base class."""
    example = []
    for f in self.name_list:
      example += self._create_examples(
            self._read_st(os.path.join(data_dir, "train_data/dev_%s"%f)), "dev-%s"%f)
    np.random.shuffle(example)
    print("dev:", len(example))
    return example

  def get_test_examples(self, data_dir):
    """See base class."""
    example = []
    for f in self.name_list:
      example += self._create_examples(
            self._read_st(os.path.join(data_dir, "train_data/test_%s"%f)), "test-%s"%f)
    print("test", len(example))
    return example

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = [tokenization.convert_to_unicode(s) for s in line["story"]]
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=line["label"], ref=line["ref"] if FLAGS.use_reconstruction else None))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        ref_input_ids=[0] * max_seq_length,
        ref_input_mask=[0] * max_seq_length,        
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)


  def token(text):
    if isinstance(text, str):
      token_text = tokenizer.tokenize(text)
      return token_text, [len(token_text)]
    elif isinstance(text, list):
      token_text = [tokenizer.tokenize(t) for t in text]
      token_text_mask = [len(t) for t in token_text]
      return reduce(operator.add, token_text), token_text_mask
    else:
      print(text)
      print("TOKEN ERROR")
      exit(-1)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  tokens_a, _ = token(example.text_a)
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]

  if FLAGS.use_reconstruction:
    tokens_a_ref, tokens_a_ref_length = token(example.ref)
    if len(tokens_a_ref) > max_seq_length - 2:
      tokens_a_ref = tokens_a_ref[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  ref_tokens = []
  if FLAGS.use_reconstruction:
    ref_tokens.append("[CLS]")
    for token in tokens_a_ref:
      ref_tokens.append(token)
    ref_tokens.append("[SEP]")
  
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  ref_input_ids, ref_input_mask = None, None
  if FLAGS.use_reconstruction:
    ref_input_ids = tokenizer.convert_tokens_to_ids(ref_tokens)
    ref_input_mask = [0] * (tokens_a_ref_length[0]+1) + [1] * (len(ref_input_ids)-tokens_a_ref_length[0]-1)
    while len(ref_input_mask) < max_seq_length:
      ref_input_mask.append(0)
      ref_input_ids.append(0)
    if len(ref_input_ids) > max_seq_length:
      ref_input_ids = ref_input_ids[:max_seq_length]
      ref_input_mask = ref_input_mask[:max_seq_length]

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    if FLAGS.use_reconstruction:
      tf.logging.info("ref_tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in ref_tokens]))
      tf.logging.info("ref_input_ids: %s" % " ".join([str(x) for x in ref_input_ids]))
      tf.logging.info("ref_input_mask: %s" % " ".join([str(x) for x in ref_input_mask]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      ref_input_ids=ref_input_ids,
      ref_input_mask=ref_input_mask,      
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def get_ref_lm_output(bert_config, input_tensor, output_weights,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  # input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.

    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_weights = tf.cast(label_weights, tf.float32)

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    predict_token = tf.argmax(log_probs, axis=-1)
    # [batch_size, seq_len]
    per_position_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_position_loss, 1)
    denominator = tf.reduce_sum(tf.cast(tf.not_equal(label_weights, 0.0), tf.float32), 1) + 1e-5
    per_example_loss = numerator / denominator
    loss = tf.reduce_mean(per_example_loss)

  return (loss, per_example_loss, log_probs, predict_token)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, ref_input_ids=None, ref_input_mask=None):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    if FLAGS.use_reconstruction:
      ref_hidden = model.get_all_encoder_layers()[-1]
      (ref_loss,
          ref_per_example_loss, ref_log_probs, predict_token) = get_ref_lm_output(
         bert_config, ref_hidden, model.get_embedding_table(),
         ref_input_ids, ref_input_mask)
      total_loss = loss + 0.1 * ref_loss
      return (total_loss, per_example_loss, ref_per_example_loss, predict_token, logits, probabilities, output_layer)
    return (loss, per_example_loss, logits, probabilities, output_layer)

output_param = True
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  global output_param
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    global output_param
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]

    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    ref_input_ids, ref_input_mask = None, None
    if "ref_input_ids" in features:
      ref_input_ids = features["ref_input_ids"]
      ref_input_mask = features["ref_input_mask"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model_output = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings, ref_input_ids=ref_input_ids, ref_input_mask=ref_input_mask)
    ref_per_example_loss, predict_token = None, None
    if FLAGS.use_reconstruction:
      (total_loss, per_example_loss, ref_per_example_loss, predict_token, logits, probabilities, output_layer) = model_output
    else:
      (total_loss, per_example_loss, logits, probabilities, output_layer) = model_output

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if output_param:
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
        output_param = False
    def metric_fn(per_example_loss, label_ids, logits, is_real_example, ref_per_example_loss=None):
      predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      accuracy = tf.metrics.accuracy(
        labels=label_ids, predictions=predictions, weights=is_real_example)
      loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
      result_map = {
          "result_accuracy": accuracy,
          "result_loss": loss,
        }
      if ref_per_example_loss is not None:
        ref_loss = tf.metrics.mean(values=ref_per_example_loss, weights=is_real_example)
        result_map["result_ref_loss"] = ref_loss
      return result_map
    predictions = {"probabilities": probabilities, 
                    "output_layer": output_layer, 
                    "input_mask": input_mask,
                    "label": label_ids}
    output_spec = None
    if FLAGS.use_reconstruction:
      eval_metrics = (metric_fn,
                  [per_example_loss, label_ids, logits, is_real_example, ref_per_example_loss])
      predictions["predict_token"] = predict_token
    else:    
      eval_metrics = (metric_fn,
                  [per_example_loss, label_ids, logits, is_real_example, None])

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          eval_metrics=eval_metrics,
          predictions=predictions,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          predictions=predictions,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predictions,
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  if FLAGS.use_reconstruction:
    all_ref_input_ids = []
    all_ref_input_mask = []    
  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)
    all_input_mask.append(feature.input_mask)    
    if FLAGS.use_reconstruction:
      all_ref_input_mask.append(feature.ref_input_mask)
      all_ref_input_ids.append(feature.ref_input_ids)      
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d_dict = {
        "input_ids":
            tf.constant(
                all_input_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),          
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    }
    if FLAGS.use_reconstruction:
      d_dict["ref_input_ids"] = tf.constant(
                all_ref_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32)
      d_dict["ref_input_mask"] = tf.constant(
                all_ref_input_mask, shape=[num_examples, seq_length],
                dtype=tf.int32)

    d = tf.data.Dataset.from_tensor_slices(d_dict)

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


dataset_feature = {}
def train(estimator, train_examples, label_list, tokenizer, num_train_steps, num=0):
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = input_fn_builder(
        features=dataset_feature["train"],
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True
      )
  estimator.train(input_fn=train_input_fn, steps=num_train_steps)

first = {"dev":True, "test":True}
def eval(processor, estimator, label_list, tokenizer, num=0, name_list=None, map_name_func=None):
  for name in name_list:
    eval_examples = map_name_func[name](FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = input_fn_builder(
          features=dataset_feature[name],
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=eval_drop_remainder,
        )

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "%s_results.txt"%name)
    with tf.gfile.GFile(output_eval_file, "w" if first[name] else "a+") as writer:
      tf.logging.info("***** %s results *****"%name)
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
      writer.write("=====\n")
      first[name] = False

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "train": StoryClassiferProcessor,
      "pred": GenStoryProcessor,
  }

  # tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  map_name_func = {
    "train":processor.get_train_examples,
    "dev":processor.get_dev_examples,
    "test":processor.get_test_examples,
  }
  label_list = processor.get_labels()
  
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  name_list = ["train", "dev", "test"] if FLAGS.do_train else ["test"]
  for name in name_list:
    print("beginning checking %s"%name)
    eval_examples = map_name_func[name](FLAGS.data_dir)
    print("finish reading %s"%name)
    dataset_feature[name] = convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer)
  tpu_cluster_resolver = None

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=50,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  tf.logging.info("***** building model function *****")
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
  if FLAGS.do_train:
    for i in range(int(num_train_steps / FLAGS.save_checkpoints_steps)):
      train(estimator, train_examples, label_list, tokenizer, FLAGS.save_checkpoints_steps, num=i)
      if FLAGS.do_eval:
        eval(processor, estimator, label_list, tokenizer, num=i, name_list=["dev", "test"], map_name_func=map_name_func)
  elif FLAGS.do_eval:
    eval(processor, estimator, label_list, tokenizer, num=0, name_list=["test"], map_name_func=map_name_func)

  if FLAGS.do_predict:
    for name in ["test"]:
      predict_examples = map_name_func[name](FLAGS.data_dir)
      num_actual_predict_examples = len(predict_examples)

      tf.logging.info("***** Running prediction*****")
      tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                      len(predict_examples), num_actual_predict_examples,
                      len(predict_examples) - num_actual_predict_examples)
      tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
      predict_drop_remainder = True if FLAGS.use_tpu else False
      predict_input_fn = input_fn_builder(
            features=dataset_feature[name],
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder
          )

      result = [r for r in estimator.predict(input_fn=predict_input_fn)]
      opt_name_list = ["probabilities"]
      if FLAGS.use_reconstruction: opt_name_list += ["predict_token"]
      for opt_name in opt_name_list:
        output_name = "%s_results_%s.txt"%(name, opt_name)
        output_predict_file = os.path.join(FLAGS.output_dir, output_name)
        with tf.gfile.GFile(output_predict_file, "w") as writer:
          tf.logging.info("***** Predict results %s *****" % opt_name)
          for (i, prediction) in enumerate(result):
            opt = prediction[opt_name]
            if i % 1000 == 0:
              print(i)
            if i >= num_actual_predict_examples:
              break
            if opt_name == "predict_token":
              output_line = " ".join(
                tokenizer.convert_ids_to_tokens(opt, trunct=False)
              ) + "\n"
            else:
              output_line = str(opt[1]) + "\n"
            writer.write(output_line)

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
