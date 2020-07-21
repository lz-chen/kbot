import json
import os
from typing import Optional

import numpy as np
import tensorflow as tf
import model, sample, encoder
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class ConditionalModel:
    model_name: str = '345M'
    seed: Optional[int] = None
    nsamples: int = 1
    batch_size: int = 1
    length: Optional[int] = None
    temperature: int = 1
    top_k: int = 40
    top_p: int = 0
    models_dir: str = '/media/liah/DATA/docker/trained_models'

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = 1
        assert self.nsamples % self.batch_size == 0

        self.enc = encoder.get_encoder(self.models_dir, self.model_name)
        self.hparams = model.default_hparams()
        with open(
                os.path.join(self.models_dir, self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
        elif self.length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # with tf.Session(graph=tf.Graph()) as sess:
        self.context = tf.placeholder(tf.int32, [self.batch_size, None])
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.output = sample.sample_sequence(
            hparams=self.hparams, length=self.length,
            context=self.context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
        )

        saver = tf.train.Saver()
        print(f"MODEL DIR {self.models_dir}")
        print(f"MODEL NAME {self.model_name}")
        print(f"PWD {os.getcwd()}")
        print(f"MODEL DIR ABS {Path(self.models_dir).absolute()}")
        ckpt = tf.train.latest_checkpoint(
            os.path.join(self.models_dir, self.model_name))
        saver.restore(self.sess, ckpt)

    def generate(self, sentences):
        if sentences == None:
            raise ValueError('Sentences cannot be None')

        listy = []
        n = 0
        if isinstance(sentences, list):
            for i in sentences:
                context_tokens = self.enc.encode(i)
                for _ in range(self.nsamples // self.batch_size):
                    out = self.sess.run(self.output, feed_dict={
                        self.context: [context_tokens for _ in range(self.batch_size)]
                    })[:, len(context_tokens):]
                text = i + self.enc.decode(out[0])
                listy.append(text)
                n += 1
                print(n)
            return dict(zip(sentences, listy))
        else:
            context_tokens = self.enc.encode(sentences)
            for _ in range(self.nsamples // self.batch_size):
                out = self.sess.run(self.output, feed_dict={
                    self.context: [context_tokens for _ in range(self.batch_size)]
                })[:, len(context_tokens):]
            text = sentences + self.enc.decode(out[0])

            return {sentences: text}


if __name__ == '__main__':
    cond_model = ConditionalModel(model_name='plato', seed=155)
    reply = cond_model.generate(sentences=['Is the love of God divine and better than all other '
                                           'types of love?',
                                           'What is freedom?'])
    for k, v in reply.items():
        print(k)
        print(v)
