from collections import Counter
import random

from boltons.iterutils import chunked
import numpy as np
from pysaliency.datasets import create_subset
from pysaliency.utils import remove_trailing_nans
import torch
from tqdm import tqdm


def ensure_color_image(image):
    if len(image.shape) == 2:
        return np.dstack([image, image, image])
    return image


def x_y_to_sparse_indices(xs, ys):
    # Converts list of x and y coordinates into indices and values for sparse mask
    x_inds = []
    y_inds = []
    values = []
    pair_inds = {}

    for x, y in zip(xs, ys):
        key = (x, y)
        if key not in pair_inds:
            x_inds.append(x)
            y_inds.append(y)
            pair_inds[key] = len(x_inds) - 1
            values.append(1)
        else:
            values[pair_inds[key]] += 1

    return np.array([y_inds, x_inds]), values


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli,
        fixations,
        centerbias_model=None,
        transform=None,
        cached=True,
        average="fixation",
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.transform = transform
        self.average = average

        self.cached = cached
        if cached:
            self._cache = {}
            print("Populating fixations cache")
            self._xs_cache = {}
            self._ys_cache = {}

            for x, y, n in zip(
                self.fixations.x_int, self.fixations.y_int, tqdm(self.fixations.n)
            ):
                self._xs_cache.setdefault(n, []).append(x)
                self._ys_cache.setdefault(n, []).append(y)

            for key in list(self._xs_cache):
                self._xs_cache[key] = np.array(self._xs_cache[key], dtype=np.long)
            for key in list(self._ys_cache):
                self._ys_cache[key] = np.array(self._ys_cache[key], dtype=np.long)

    def get_shapes(self):
        return list(self.stimuli.sizes)

    def __getitem__(self, key):
        if not self.cached or key not in self._cache:
            image = np.array(self.stimuli.stimuli[key])
            centerbias_prediction = self.centerbias_model.log_density(image)

            image = ensure_color_image(image).astype(np.float32)
            image = image.transpose(2, 0, 1)

            if self.cached:
                xs = self._xs_cache.pop(key)
                ys = self._ys_cache.pop(key)
            else:
                inds = self.fixations.n == key
                xs = np.array(self.fixations.x_int[inds], dtype=np.long)
                ys = np.array(self.fixations.y_int[inds], dtype=np.long)

            data = {
                "image": image,
                "x": xs,
                "y": ys,
                "centerbias": centerbias_prediction,
            }

            if self.average == "image":
                data["weight"] = 1.0
            else:
                data["weight"] = float(len(xs))

            if self.cached:
                self._cache[key] = data
        else:
            data = self._cache[key]

        if self.transform is not None:
            return self.transform(dict(data))

        return data

    def __len__(self):
        return len(self.stimuli)


class FixationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli,
        fixations,
        centerbias_model=None,
        transform=None,
        included_fixations=-2,
        average="fixation",
        cache_image_data=False,
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.transform = transform
        self.average = average

        self._shapes = None

        if isinstance(included_fixations, int):
            if included_fixations < 0:
                included_fixations = [-1 - i for i in range(-included_fixations)]
            else:
                raise NotImplementedError()

        self.included_fixations = included_fixations
        self.fixation_counts = Counter(fixations.n)

        self.cache_image_data = cache_image_data

        if self.cache_image_data:
            self.image_data_cache = {}

            print("Populating image cache")
            for n in tqdm(range(len(self.stimuli))):
                self.image_data_cache[n] = self._get_image_data(n)

    def get_shapes(self):
        if self._shapes is None:
            shapes = list(self.stimuli.sizes)
            self._shapes = [shapes[n] for n in self.fixations.n]

        return self._shapes

    def _get_image_data(self, n):
        image = np.array(self.stimuli.stimuli[n])
        centerbias_prediction = self.centerbias_model.log_density(image)

        image = ensure_color_image(image).astype(np.float32)
        image = image.transpose(2, 0, 1)

        return image, centerbias_prediction

    def __getitem__(self, key):
        n = self.fixations.n[key]

        if self.cache_image_data:
            image, centerbias_prediction = self.image_data_cache[n]
        else:
            image, centerbias_prediction = self._get_image_data(n)

        x_hist = remove_trailing_nans(self.fixations.x_hist[key])
        y_hist = remove_trailing_nans(self.fixations.y_hist[key])

        data = {
            "image": image,
            "x": np.array([self.fixations.x_int[key]], dtype=np.long),
            "y": np.array([self.fixations.y_int[key]], dtype=np.long),
            "x_hist": x_hist[self.included_fixations],
            "y_hist": y_hist[self.included_fixations],
            "centerbias": centerbias_prediction,
        }

        if self.average == "image":
            data["weight"] = 1.0 / self.fixation_counts[n]
        else:
            data["weight"] = 1.0

        if self.transform is not None:
            return self.transform(data)

        return data

    def __len__(self):
        return len(self.fixations)


class FixationMaskTransform(object):
    def __call__(self, item):
        shape = torch.Size([item["image"].shape[1], item["image"].shape[2]])
        x = item.pop("x")
        y = item.pop("y")

        # inds, values = x_y_to_sparse_indices(x, y)
        inds = np.array([y, x])
        values = np.ones(len(y), dtype=np.int)

        mask = torch.sparse.IntTensor(torch.tensor(inds), torch.tensor(values), shape)
        mask = mask.coalesce()

        item["fixation_mask"] = mask

        return item


def collate_fn(batch):
    batch_data = {
        "image": torch.tensor([item["image"] for item in batch]),
        "fixations": torch.sparse.LongTensor(),
    }
    return batch_data


class ImageDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=1, ratio_used=1.0, shuffle=True):
        self.ratio_used = ratio_used
        self.shuffle = shuffle

        shapes = data_source.get_shapes()
        unique_shapes = sorted(set(shapes))

        shape_indices = [[] for shape in unique_shapes]

        for k, shape in enumerate(shapes):
            shape_indices[unique_shapes.index(shape)].append(k)

        if self.shuffle:
            for indices in shape_indices:
                random.shuffle(indices)

        self.batches = sum(
            [chunked(indices, size=batch_size) for indices in shape_indices], []
        )

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.batches))
        else:
            indices = range(len(self.batches))

        if self.ratio_used < 1.0:
            indices = indices[: int(self.ratio_used * len(indices))]

        return iter(self.batches[i] for i in indices)

    def __len__(self):
        return int(self.ratio_used * len(self.batches))


# TODO: The following helper functions should go into pysaliency or be replaced by
# functions already contained therein.


def create_train_folds(crossval_folds, val_folds, test_folds):
    all_folds = list(range(crossval_folds))
    if isinstance(val_folds, int):
        val_folds = [val_folds]
    if isinstance(test_folds, int):
        test_folds = [test_folds]

    train_folds = [f for f in all_folds if not (f in val_folds or f in test_folds)]

    return train_folds, val_folds, test_folds


def get_crossval_folds(crossval_folds, crossval_no, test_folds=1, val_folds=1):
    assert test_folds <= 1
    if test_folds:
        _test_folds = [crossval_no]
        _val_folds = [(crossval_no - i - 1) % crossval_folds for i in range(val_folds)]

    else:
        assert val_folds == 1

        _test_folds = [crossval_no]
        _val_folds = [crossval_no]

    _train_folds, _val_folds, _test_folds = create_train_folds(
        crossval_folds, _val_folds, _test_folds
    )

    return _train_folds, _val_folds, _test_folds


def get_crossval_split(stimuli, fixations, split_count, included_splits, random=True):
    inds = list(range(len(stimuli)))
    if random:
        print("Using random shuffles for crossvalidation")
        rst = np.random.RandomState(seed=42)
        rst.shuffle(inds)
        inds = list(inds)
    size = int(np.ceil(len(inds) / split_count))
    chunks = chunked(inds, size=size)

    inds = []
    for split_nr in included_splits:
        inds.extend(chunks[split_nr])

    stimuli, fixations = create_subset(stimuli, fixations, inds)
    return stimuli, fixations
