import numpy as np
import tensorflow as tf


def roi_pooling_tf(fmap, coords, output_shape):
    """
    Description:
    fmap 을 지정된 크기만큼 잘라서 가져옵니다.
    지정된 크기가 소수점이라면 내림(floor)을 수행합니다.
    Example)
     fmap = (3.5, 3.5, 7, 4.2) 라면 x1, y1, x2, y2 로 치환하면 (0, 1.3, 7, 5.6)가 됩니다.
     위 치환됟 좌표에서 내림해 (0, 1, 7, 5)가 됩니다.

    최종 출력 output 이 2,2 라면 x, y축을 각각 2등분 합니다.
    각 4개의 cell 을 나누는 교차점의 좌표가  (3.5, 3) 가 됩니다.

    +---+---+
    | a | b |
    +---+---+
    | c | d |
    +---+---+

    각 좌표를 나타내면 아래와 같이 나타낼 수 있습니다.
    이때 소수점을 다시 내림 합니다.

    |   | x1  | y1 | x2 | y2 |        |   | x1 | y1 | x2 | y2 |
    +---+-----+----+----+----+        +---+----+----+----+----+
    | a | 0   | 1  | 3  | 3  |        | a | 0  | 1  | 3  | 3  |
    +---+-----+----+----+----+        +---+----+----+----+----+
    | b | 3.5 | 1  | 7  | 3  |        | b | 3  | 1  | 7  | 3  |
    +---+-----+----+----+----+  ->    +---+----+----+----+----+
    | c | 0   | 3  | 3  | 5  |        | c | 0  | 3  | 3  | 5  |
    +---+-----+----+----+----+        +---+----+----+----+----+
    | d | 3.5 | 3  | 7  | 6  |        | d | 3  | 3  | 7  | 6  |
    +---+-----+----+----+----+        +---+----+----+----+----+

    이렇게 나눠진 Cell 들의 값에서 max pooling 을 수행합니다.

    +----+----+----+----+----+----+----+
    | *1 | *2 | *3 | ※4 | ※5 | ※6 | ※7 |
    +----+----+----+----+----+----+----+
    | *8 | *9 | *1 | ※2 | ※3 | ※4 | ※5 |    +----+----+
    +----+----+----+----+----+----+----+    | *9 | ※6 |
    | ▲6 | ▲7 | ▲8 | ☐9 | ☐0 | ☐1 | ☐2| -> +----+----+
    +----+----+----+----+----+----+----+    | ▲8 | ☐9 |
    | ▲3 | ▲4 | ▲5 | ☐6 | ☐7 | ☐8 | ☐9|    +----+----+
    +----+----+----+----+----+----+----+
    | ▲0 | ▲1 | ▲2 | ☐3 | ☐4 | ☐5 | ☐6|
    +----+----+----+----+----+----+----+


    :param fmap: ndarray, shape=(N_data, w, h, ch)
    :param coord: ndarray, shape=(N_anchor, 4=(cx cy w h)):
    :return:
    """

    # 좌표값을 n 등분 합니다.
    # shape = (N_anchor, n_output)
    coords = tf.cast(tf.floor(coords), tf.int32)

    cropped_fmap = fmap[coords[1]:coords[3] + 1, coords[0]:coords[2] + 1]

    xs_range = tf.linspace(0, cropped_fmap.shape[1], num=output_shape[1] + 1, axis=-1)
    xs_range = tf.floor(xs_range)
    xs_index = tf.cast(tf.concat([xs_range[1:-1], [cropped_fmap.shape[1] - 1]], axis=-1), tf.int32)

    # shape N_anchor = (N_anchor, n_output)
    ys_range = tf.linspace(0, cropped_fmap.shape[0], num=output_shape[0] + 1, axis=-1)
    ys_range = tf.floor(ys_range)
    ys_index = tf.cast(tf.concat([ys_range[1:-1], [cropped_fmap.shape[0] - 1]], axis=-1), tf.int32)

    vertical_splits = tf.split(cropped_fmap, num_or_size_splits=ys_index, axis=0)
    # ※ for 구문으로 처리해도 가능한 이유는 항상 나눠지는 개수가 동일하기 때문이다.
    max_roi = []
    for vertical_split in vertical_splits:
        values = tf.split(vertical_split, xs_index, axis=1)
        for value in values:
            max_value = tf.reduce_max(value, axis=[0, 1])
            max_roi.append(max_value)

    return max_roi


def roi_pooling_np(fmap, coords, output_shape):
    """
    Description:
    fmap 을 지정된 크기만큼 잘라서 가져옵니다.
    지정된 크기가 소수점이라면 내림(floor)을 수행합니다.
    Example)
     fmap = (3.5, 3.5, 7, 4.2) 라면 x1, y1, x2, y2 로 치환하면 (0, 1.3, 7, 5.6)가 됩니다.
     위 치환됟 좌표에서 내림해 (0, 1, 7, 5)가 됩니다.

    최종 출력 output 이 2,2 라면 x, y축을 각각 2등분 합니다.
    각 4개의 cell 을 나누는 교차점의 좌표가  (3.5, 3) 가 됩니다.

    +---+---+
    | a | b |
    +---+---+
    | c | d |
    +---+---+

    각 좌표를 나타내면 아래와 같이 나타낼 수 있습니다.
    이때 소수점을 다시 내림 합니다.

    |   | x1  | y1 | x2 | y2 |        |   | x1 | y1 | x2 | y2 |
    +---+-----+----+----+----+        +---+----+----+----+----+
    | a | 0   | 1  | 3  | 3  |        | a | 0  | 1  | 3  | 3  |
    +---+-----+----+----+----+        +---+----+----+----+----+
    | b | 3.5 | 1  | 7  | 3  |        | b | 3  | 1  | 7  | 3  |
    +---+-----+----+----+----+  ->    +---+----+----+----+----+
    | c | 0   | 3  | 3  | 5  |        | c | 0  | 3  | 3  | 5  |
    +---+-----+----+----+----+        +---+----+----+----+----+
    | d | 3.5 | 3  | 7  | 6  |        | d | 3  | 3  | 7  | 6  |
    +---+-----+----+----+----+        +---+----+----+----+----+

    이렇게 나눠진 Cell 들의 값에서 max pooling 을 수행합니다.

    +----+----+----+----+----+----+----+
    | *1 | *2 | *3 | ※4 | ※5 | ※6 | ※7 |
    +----+----+----+----+----+----+----+
    | *8 | *9 | *1 | ※2 | ※3 | ※4 | ※5 |    +----+----+
    +----+----+----+----+----+----+----+    | *9 | ※6 |
    | ▲6 | ▲7 | ▲8 | ☐9 | ☐0 | ☐1 | ☐2| -> +----+----+
    +----+----+----+----+----+----+----+    | ▲8 | ☐9 |
    | ▲3 | ▲4 | ▲5 | ☐6 | ☐7 | ☐8 | ☐9|    +----+----+
    +----+----+----+----+----+----+----+
    | ▲0 | ▲1 | ▲2 | ☐3 | ☐4 | ☐5 | ☐6|
    +----+----+----+----+----+----+----+


    :param fmap: ndarray, shape=(N_data, w, h, ch)
    :param coord: ndarray, shape=(N_anchor, 4=(cx cy w h)):
    :return:
    """

    # 좌표값을 n 등분 합니다.
    # shape = (N_anchor, n_output)
    coords = np.floor(coords).astype(int)
    fmap = fmap[coords[1]:coords[3] + 1, coords[0]:coords[2] + 1]

    xs_range = np.linspace(0, fmap.shape[1], num=output_shape[1] + 1, axis=-1)
    xs_range = np.floor(xs_range)
    xs_index = xs_range[1:-1].astype(int)

    # shape N_anchor = (N_anchor, n_output)
    ys_range = np.linspace(0, fmap.shape[0], num=output_shape[0] + 1, axis=-1)
    ys_range = np.floor(ys_range)
    ys_index = ys_range[1:-1]
    ys_index = list(ys_index.astype(int))

    vertical_splits = np.split(fmap, indices_or_sections=ys_index, axis=0)

    max_roi = []
    for vertical_split in vertical_splits:
        values = np.split(vertical_split, xs_index, axis=1)
        for value in values:
            max_roi.append(np.max(value, axis=(0, 1)))
    max_roi = np.array(max_roi)
    max_roi = np.reshape(max_roi, newshape=output_shape)

    return max_roi


if __name__ == '__main__':
    a = np.zeros((2, 2))
    b = np.ones((2, 3))
    c = np.ones((3, 2)) * 2
    d = np.ones((3, 3)) * 3

    ab = np.hstack([a, b])
    cd = np.hstack([c, d])
    abcd = np.vstack([ab, cd])
    coord = (1.1, 2.4, 3.3, 5.7)
    # roi_pooling_np(abcd, coords, output_shape=(2, 2))
    abcds = np.stack([abcd, abcd, abcd], axis=-1)
    roi_pool = roi_pooling_np(abcds, coord, output_shape=(2, 2, 3))

    abcds_tf = tf.constant(abcds)
    coords_tf = tf.constant(coord)

    roi_pool = roi_pooling_tf(abcds_tf, coords_tf, output_shape=(2, 2))
    print(roi_pool)
