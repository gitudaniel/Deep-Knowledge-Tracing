import pandas as pd
import tensorflow as tf
import numpy as np


MASK_VALUE = -1  # The masking value cannot be zero.


def load_dataset(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn)
    # count number of visually_impaired students to justify dropping
    # add PCA (principal component analysis) to justify dropping
    df.drop(['date_of_evaluation', 'taxonomy_id_1', 'taxonomy_id_2', 'number_of_taxonomies', 'taxonomy_id_1_class_level', 'taxonomy_id_1_specific_objective', 'taxonomy_id_1_subject', 'taxonomy_id_1_unit', 'taxonomy_id_2_class_level', 'taxonomy_id_2_specific_objective', 'taxonomy_id_2_subject', 'taxonomy_id_2_unit', 'student_first_name', 'student_last_name', 'student_updated_at', 'student_created_at', 'visually_impaired', 'class_name', 'class_grade', 'school_name', 'school_details'], axis=1, inplace=True)

    # df = df.sample(n=100000)
    df = df[df['subject'] == 'math']

    if "taxonomy_id_0" not in df.columns:
        raise KeyError(f"The column 'taxonomy_id_0' was not found on {fn}")
    if "answer_selection_correct" not in df.columns:
        raise KeyError(f"The column 'answer_selection_correct' was not found on {fn}")
    if "student_id" not in df.columns:
        raise KeyError(f"The column 'student_id' was not found on {fn}")

    if not (df['answer_selection_correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'answer_selection_correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without taxonomy
    df.dropna(subset=['taxonomy_id_0'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('student_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['factorized_taxonomy_code'], _ = pd.factorize(df['taxonomy_id_0'], sort=True)

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # feature crossing: https://developers.google.com/machine-learning/crash-course/feature-crosses/crossing-one-hot-vectors
    df['taxonomy_with_answer'] = df['factorized_taxonomy_code'] * 2 + df['answer_selection_correct']

    df['answer_selection_correct'] = pd.to_numeric(df['answer_selection_correct'])

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    # Create a series for each student followed by a tuple of arrays for their
    # taxonomy_with_answer, factorized_taxonomy_code, answer_selection_correct
    seq = df.groupby('student_id').apply(
        lambda r: (
            r['taxonomy_with_answer'].values[:-1],
            r['factorized_taxonomy_code'].values[1:],
            r['answer_selection_correct'].values[1:],
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    # create a tensorflow Dataset from our series
    # Dataset potentially large set of elements in a format
    # easy for Tensorflow to work with and manipulate
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
        # output_types=(tf.int32, tf.int32, tf.int32)
    )

    # fill a buffer with nb_users elements (4047 elements)
    # randomly sample from this buffer, replacing the
    # selected elements with new elements.
    # our buffer_size is equal to the number of elements
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Step 6 - Encode categorical features and merge taxonomies with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    features_depth = df['taxonomy_with_answer'].max() + 1
    taxonomy_depth = df['factorized_taxonomy_code'].max() + 1

    # expand_dims -> add a dimension to the data
    dataset = dataset.map(
        lambda feat, factorized_taxonomy_code, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(factorized_taxonomy_code, depth=taxonomy_depth),
                    tf.expand_dims(label, -1)
                ],
                axis = -1
            )
        )
    )
    # dataset = dataset.map(
    #     lambda feat, skill: (
    #         tf.one_hot(feat, depth=features_depth),
    #         tf.one_hot(skill, depth=taxonomy_depth),
    #     )
    # )

    # Step 7 - Pad sequences per batch
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch
    # https://keras.io/guides/understanding_masking_and_padding/
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        # padding_values=(MASK_VALUE, MASK_VALUE),
        padding_values=(
            tf.constant(-1, dtype=tf.float32),
            tf.constant(-1, dtype=tf.float32)
        ),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size
    return dataset, length, features_depth, taxonomy_depth


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set


def get_target(y_true, y_pred):
    # Get taxonomies and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    taxonomies, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each taxonomy
    y_pred = tf.reduce_sum(y_pred * taxonomies, axis=-1, keepdims=True)

    return y_true, y_pred
