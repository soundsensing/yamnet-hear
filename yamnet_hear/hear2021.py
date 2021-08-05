
"""
HEAR2021 API implementation

As per specifications in
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html
"""

HOP_SIZE_TIMESTAMPS = 0.050 # <50 ms recommended
HOP_SIZE_SCENE = 0.5
EMBEDDING_SIZE = 1024

import openl3
import numpy
import tensorflow as tf
import tensorflow_hub as hub

#import tensorflow_datasets
#from tensorflow_datasets.typing import Tensor
#from tensorflow.types.experimental import Tensor
from typing import NewType, Tuple
Tensor = NewType('Tensor', object)

class Model(tf.Module):
    def __init__(self, model, sample_rate=16000, embedding_size=EMBEDDING_SIZE):
        self.sample_rate = sample_rate
        self.scene_embedding_size = embedding_size
        self.timestamp_embedding_size = embedding_size

        self.yamnet_model = model # the YAMNET model instance    


def load_model(model_file_path: str) -> Model:
    # FIXME: respect model_file_path

    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    model = Model(model=yamnet_model)
    return model

TimestampedEmbeddings = Tuple[Tensor, Tensor]

def get_timestamp_embeddings(
    audio: Tensor,
    model: Model,
    hop_size=HOP_SIZE_TIMESTAMPS,
) -> TimestampedEmbeddings:
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]
    model: Loaded Model. 

    Returns:

        embedding: A float32 Tensor with shape (n_sounds, n_timestamp, model.timestamp_embedding_size).
        timestamps: Tensor. Centered timestamps in milliseconds corresponding to each embedding in the output.
     """
    # pre-conditions
    assert len(audio.shape) == 2

    # get embeddings for a single audio clip
    def get_embedding(samples):
        scores, embeddings, spectrogram = model.yamnet_model(samples)

        ts = numpy.arange(embeddings.shape[0])*hop_size
        return embeddings, ts

    # Compute embeddings for each clip
    embeddings = []
    timestamps = []
    for sound_no in range(audio.shape[0]):
        samples = numpy.array(audio[sound_no, :])
        emb, ts = get_embedding(samples)
        embeddings.append(emb)
        timestamps.append(ts)
    emb = numpy.stack(embeddings)
    ts = numpy.stack(timestamps)
    emb = tf.convert_to_tensor(emb)
    ts = tf.convert_to_tensor(ts)
    
    # post-conditions
    assert len(ts.shape) == 2 
    assert len(ts) >= 1
    assert emb.shape[0] == audio.shape[0]
    assert len(emb.shape) == 3, emb.shape
    assert ts.shape[0] == audio.shape[0]
    assert emb.shape[1] == ts.shape[1], (emb.shape, ts.shape)
    assert emb.shape[2] == model.timestamp_embedding_size
    if len(ts) >= 2:
        assert ts[0,1] == ts[0,0] + hop_size

    # XXX: are timestampes centered?
    # first results seems to be 0.0, which would indicate that window
    # starts at -window/2 ?
    #assert ts[0] > 0.0 and ts[0] < hop_size, ts
    return (emb, ts)


def get_scene_embeddings(
    audio: Tensor,
    model: Model,
    hop_size=HOP_SIZE_SCENE,
) -> Tensor:

    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1].
    model: Loaded Model.

    Returns:

        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    assert len(audio.shape) == 2 

    embeddings, ts = get_timestamp_embeddings(audio, model, hop_size=hop_size)

    # FIXME: use TensorFlow Tensor instead. Using tf.constant ?
    emb = numpy.mean(embeddings, axis=1)
    emb = tf.convert_to_tensor(emb)

    assert len(emb.shape) == 2, emb.shape
    assert emb.shape[0] == audio.shape[0], (emb.shape, audio.shape)
    assert emb.shape[1] == model.scene_embedding_size, (emb.shape, audio.shape)

    return emb


