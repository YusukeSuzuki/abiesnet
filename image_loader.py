import tensorflow as tf

def build_full_network(path_strings, height, width, channels, batch_size, reader,
    flip_up_down=False, flip_left_right=False,
    random_brightness=False, braightness_range=0.4,
    random_contrast=False, contrast_min=0.6, contrast_max=1.4):

    ''' create image read operations '''

    filename_queue = tf.train.string_input_producer(path_strings, shuffle=True)
    _, raw = reader.read(filename_queue)

    read_image = tf.image.decode_jpeg(raw, channels=channels)

    if flip_up_down:
        read_image = tf.image.random_flip_up_down(read_image)
    if flip_left_right:
        read_image = tf.image.random_flip_left_right(read_image)
    if random_brightness:
        read_image = tf.image.random_brightness(read_image, braightness_range)
    if random_contrast:
        read_image = tf.image.random_contrast(
            read_image, contrast_min, contrast_max)

    read_image = tf.to_float(read_image)
    read_image = tf.image.resize_images(read_image, height,width)

    batch_images = tf.train.shuffle_batch([read_image], batch_size=batch_size,
        capacity=20000,num_threads=8,min_after_dequeue=6000)

    return batch_images


