from __future__ import print_function

import numpy
import random
from sklearn import preprocessing


class DataSetViews(object):

  def __init__(self, texts, topologys, urls, demos, labels, fake_data=False):
      self._num_examples = labels.shape[0]
      self._texts = texts
      self._topologys = topologys
      self._labels = labels
      self._urls = urls
      self._demos = demos
      self._epochs_completed = 0
      self._index_in_epoch = 0

  @property
  def texts(self):
    return self._texts

  @property
  def topologys(self):
    return self._topologys

  @property
  def urls(self):
    return self._urls

  @property
  def demos(self):
    return self._demos

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._texts = self._texts[perm]
      self._topologys = self._topologys[perm]
      self._urls = self._urls[perm]
      self._demos = self._demos[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._texts[start:end], self._topologys[start:end], self._urls[start:end], self._demos[start:end], self._labels[start:end]

class SemiDataSetViews(object):
    def __init__(self, texts, topologys, urls, demos, labels, n_labeled):
        self.n_labeled = n_labeled

        # Unlabled DataSet
        self.unlabeled_ds = DataSetViews(texts, topologys, urls, demos, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        texts = texts[shuffled_indices]
        topologys = topologys[shuffled_indices]
        urls = urls[shuffled_indices]
        demos = demos[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(2)[l==1][0] for l in labels])

        n_classes = y.max() + 1
        n_from_each_class = n_labeled / n_classes
        i_labeled = []
        for c in range(n_classes):
            i = indices[y==c][:n_from_each_class]
            i_labeled += list(i)
        l_texts = texts[i_labeled]
        l_topologys = topologys[i_labeled]
        l_urls = urls[i_labeled]
        l_demos = demos[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSetViews(l_texts, l_topologys, l_urls, l_demos, l_labels)

    def next_batch(self, batch_size):
        unlabeled_texts, unlabeled_topologys, unlabeled_urls, unlabeled_demos, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_texts, labeled_topologys, labeled_urls, labeled_demos, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_texts, labeled_topologys, labeled_urls, labeled_demos, labels = self.labeled_ds.next_batch(batch_size)
        texts = numpy.vstack([labeled_texts, unlabeled_texts])
        topologys = numpy.vstack([labeled_topologys, unlabeled_topologys])
        urls = numpy.vstack([labeled_urls, unlabeled_urls])
        demos = numpy.vstack([labeled_demos, unlabeled_demos])
        return texts, topologys, urls, demos, labels

def read_data_sets_views(n_labeled = 100, TEST_SIZE = 8000):
    class DataSets(object):
        pass
    data_sets = DataSets()

    spammers = set()
    reader = open('ContentPolluter_Data/content_polluters.txt')
    for line in reader.readlines():
        spammers.add(line.strip().split('\t')[0])
    reader.close()

    legimate = set()
    reader = open('ContentPolluter_Data/legitimate_users.txt')
    for line in reader.readlines():
        legimate.add(line.strip().split('\t')[0])
    reader.close()

    nids = []
    nids.extend(spammers)
    nids.extend(legimate)

    reader = open('ContentPolluter_Data/T_0.0.emb', 'r')
    index = 0
    nid_texts = {}
    doc_colmns = 0
    for line in reader.readlines():
        if index == 0:
            ss = line.split(' ')
            doc_colmns = int(ss[1])
            index += 1
            continue
        ss = line.strip().split(' ')
        nodeid = ss[0]
        if nodeid in spammers or nodeid in legimate:
            sample = [float(ss[i]) for i in range(1, doc_colmns + 1)]
            nid_texts[nodeid] = sample
            index += 1
    reader.close()

    reader = open('ContentPolluter_Data/R_0.0.emb', 'r')
    index = 0
    nid_topologys = {}
    topo_colmns = 0
    for line in reader.readlines():
        if index == 0:
            ss = line.split(' ')
            topo_colmns = int(ss[1])
            index += 1
            continue
        ss = line.strip().split(' ')
        nodeid = ss[0]
        if nodeid in spammers or nodeid in legimate:
            sample = [float(ss[i]) for i in range(1, topo_colmns + 1)]
            nid_topologys[nodeid] = sample
            index += 1
    reader.close()

    reader = open('ContentPolluter_Data/S_0.0.emb', 'r')
    index = 0
    nid_urls = {}
    url_colmns = 0
    for line in reader.readlines():
        if index == 0:
            ss = line.split(' ')
            url_colmns = int(ss[1])
            index += 1
            continue
        ss = line.strip().split(' ')
        nodeid = ss[0]
        if nodeid in spammers or nodeid in legimate:
            sample = [float(ss[i]) for i in range(1, url_colmns + 1)]
            nid_urls[nodeid] = sample
            index += 1
    reader.close()

    reader = open('ContentPolluter_Data/D_0.0.emb', 'r')
    index = 0
    nid_demos = {}
    d_colmns = 0
    for line in reader.readlines():
        if index == 0:
            ss = line.split(' ')
            d_colmns = int(ss[1])
            index += 1
            continue
        ss = line.strip().split(' ')
        nodeid = ss[0]
        if nodeid in spammers or nodeid in legimate:
            sample = [float(ss[i]) for i in range(1, d_colmns + 1)]
            nid_demos[nodeid] = sample
            index += 1
    reader.close()

    random.shuffle(nids)

    texts_samples = []
    topologys_samples = []
    urls_samples = []
    demos_samples = []
    labels = []


    for nid in nids:
        if nid in spammers:
            labels.append(0)
        elif nid in legimate:
            labels.append(1)
        else:
            continue
        if nid in nid_texts:
            texts_samples.append(nid_texts[nid])
        else:
            texts_samples.append(numpy.ones(shape=(doc_colmns)))
        if nid in nid_topologys:
            topologys_samples.append(nid_topologys[nid])
        else:
            topologys_samples.append(numpy.ones(shape=(topo_colmns)))
        if nid in nid_urls:
            urls_samples.append(nid_urls[nid])
        else:
            urls_samples.append(numpy.ones(shape=(url_colmns)))
        if nid in nid_demos:
            demos_samples.append(nid_demos[nid])
        else:
            demos_samples.append(numpy.ones(shape=(d_colmns)))

    samples_count = len(labels)
    nb_classes = 2
    targets = numpy.array(labels).reshape(-1)
    labels = numpy.eye(nb_classes)[targets]
    demos_samples = preprocessing.normalize(demos_samples, norm='l2')

    train_texts = numpy.asarray(texts_samples[:samples_count - TEST_SIZE])
    train_topologys = numpy.asarray(topologys_samples[:samples_count - TEST_SIZE])
    train_urls = numpy.asarray(urls_samples[:samples_count - TEST_SIZE])
    train_demos = numpy.asarray(demos_samples[:samples_count - TEST_SIZE])
    train_labels = labels[:samples_count - TEST_SIZE]


    test_texts = numpy.asarray(texts_samples[samples_count - TEST_SIZE:])
    test_topologys = numpy.asarray(topologys_samples[samples_count - TEST_SIZE:])
    test_urls = numpy.asarray(urls_samples[samples_count - TEST_SIZE:])
    test_demos = numpy.asarray(demos_samples[samples_count - TEST_SIZE:])
    test_labels = labels[samples_count - TEST_SIZE:]


    data_sets.train = SemiDataSetViews(train_texts, train_topologys, train_urls, train_demos, train_labels, n_labeled)
    data_sets.test = DataSetViews(test_texts, test_topologys, test_urls, test_demos, test_labels)

    return data_sets