"""Functions for downloading and reading MNIST data."""
""" 
Edited by GLU to NOT download the images and modify a set of images so 
that they can be read by tensorflow.
They have to be converted to numpy arrays from matlab created images. 
First, let's try saving images from matlab directly into python
"""


# SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'  # Commented to avoid using by error
imagePath = '/Users/nupic/soft/nupic.vision/nupic/vision/mnist/isetmnist012/small_training'
extensions = ['.png']

import os
import xml.etree.cElementTree as ET

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin



def _walk(top):
  """
  Directory tree generator lifted from python 2.6 and then
  stripped down.  It improves on the 2.5 os.walk() by adding
  the 'followlinks' capability.
  GLU: copied from image sensor
  """

  try:
    # Note that listdir and error are globals in this module due
    # to earlier import-*.
    names = os.listdir(top)
  except OSError, e:
    raise RuntimeError("Unable to get a list of files due to an OS error.\nDirectory: "+top+"\nThis may be due to an issue with Snow Leopard.")
    #raise
  except:
    return

  dirs, nondirs = [], []
  for name in names:
    if os.path.isdir(os.path.join(top, name)):
      dirs.append(name)
    else:
      nondirs.append(name)

  yield top, dirs, nondirs
  for name in dirs:
    path = os.path.join(top, name)
    for x in _walk(path):
      yield x


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)

def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
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
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets


def createImageList(imagePath, extensions):
  pattern = None
  maskPath = None
  start = None
  stop = None
  step = None
  images = []
  categoryList = [None]
  # Assume each directory in imagePath is its own category
  categoryList = [c for c in sorted(os.listdir(imagePath))
                  if c[0] != '.' and
                  os.path.isdir(os.path.join(imagePath, c))]
  sepImages = dict()
  for category in categoryList:
    categoryFilenames = []
    if category:
      walkPath = os.path.join(imagePath, category)
    else:
      walkPath = imagePath
      category = os.path.split(imagePath)[1]

    w = _walk(walkPath)
    while True:
      try:
        dirpath, dirnames, filenames = w.next()
      except StopIteration:
        break
      # Don't enter directories that begin with '.'
      for d in dirnames[:]:
        if d.startswith('.'):
          dirnames.remove(d)
      dirnames.sort()
      # Ignore files that begin with '.'
      filenames = [f for f in filenames if not f.startswith('.')]
      # Only load images with the right extension
      filenames = [f for f in filenames
        if os.path.splitext(f)[1].lower() in extensions]
      if pattern:
        # Filter images with regular expression
        filenames = [f for f in filenames
                     if re.search(pattern, os.path.join(dirpath, f))]
      filenames.sort()
      imageFilenames = [os.path.join(dirpath, f) for f in filenames]

      # Get the corresponding path to the masks
      if maskPath:
        maskdirpath = os.path.join(maskPath, dirpath[len(imagePath)+1:])
        maskFilenames = [os.path.join(maskdirpath, f) for f in filenames]
        if strictMaskLocations:
          # Only allow masks with parallel filenames
          for i, filename in enumerate(maskFilenames):
            if not os.path.exists(filename):
              maskFilenames[i] = None
        else:
          # Find the masks even if the path does not match exactly
          for i, filename in enumerate(maskFilenames):
            while True:
              if os.path.exists(filename):
                maskFilenames[i] = filename
                break
              if os.path.split(filename)[0] == maskPath:
                # Failed to find the mask
                maskFilenames[i] = None
                break
              # Try to find the mask by eliminating subdirectories
              body, tail = os.path.split(filename)
              head, body = os.path.split(body)
              while not os.path.exists(head):
                tail = os.path.join(body, tail)
                head, body = os.path.split(head)
              filename = os.path.join(head, tail)
      else:
        maskFilenames = [None for f in filenames]
      # Add our new images and masks to the list for this category
      categoryFilenames.extend(zip(imageFilenames, maskFilenames))
    # We have the full list of filenames for this category
    sepImages[category] = []
    for f in categoryFilenames:
      images.append((f[0], f[1], category))
      sepImages[category].append((f[0], f[1], category))  
    
  return sepImages



# if __name__ == "__main__":   # Convertirlo en una funcion
def saveAsXmlList(imagePath, extensions):
  sepImageList = createImageList(imagePath, extensions)
  imagelist = ET.Element("imagelist")

  for ind in range(len(sepImageList[sepImageList.keys()[0]])):
    for category in sepImageList.keys():
        filename = sepImageList[category][ind][0]
        ET.SubElement(imagelist, 'image', file=filename, tag=category)

  tree = ET.ElementTree(imagelist)
  tree.write("isetmnist012XML.xml")




