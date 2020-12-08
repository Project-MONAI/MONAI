import numpy as np
import skimage
import skimage.measure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_cdt


# TODO:: Fix Base Class
# TODO:: Fix class names (for interactions)
# TODO:: Fix param names for each class init
# TODO:: Add 3D support for each of the following transforms
# TODO:: Unit Test
#####################################################################################
# FOR 2D
# Following are used while training
#####################################################################################
class AddInitialSeedPoint(object):
    def __init__(self, label_field, positive_guidance_field, negative_guidance_field):
        self.label_field = label_field
        self.positive_guidance_field = positive_guidance_field
        self.negative_guidance_field = negative_guidance_field

    def __call__(self, data):
        positive_guidance = data[self.positive_guidance_field]
        negative_guidance = data[self.negative_guidance_field]

        if type(positive_guidance) is np.ndarray:
            positive_guidance = positive_guidance.tolist()

        if type(negative_guidance) is np.ndarray:
            negative_guidance = negative_guidance.tolist()

        curr_label = data[self.label_field]
        curr_label = (curr_label > 0.5).astype(np.float32)

        blobs_labels = skimage.measure.label(curr_label.astype(int), background=0)
        assert np.max(blobs_labels) > 0

        for ridx in range(1, 6):
            curr_label = (blobs_labels == ridx).astype(np.float32)
            if np.sum(curr_label) == 0:
                positive_guidance.append([-1, -1, -1])
                negative_guidance.append([-1, -1, -1])
                continue

            distance = distance_transform_cdt(curr_label).flatten()
            probability = np.exp(distance) - 1.0

            idx = np.where(curr_label.flatten() > 0)[0]
            seed = np.random.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            pg = np.asarray(np.unravel_index(seed, curr_label.shape)).transpose().tolist()[0]
            pg[0] = dst[0]

            assert curr_label[..., pg[-2], pg[-1]] == 1
            assert len(pg) == 3

            positive_guidance.append(pg)
            negative_guidance.append([-1, -1, -1])

        data[self.positive_guidance_field] = np.asarray(positive_guidance)
        data[self.negative_guidance_field] = np.asarray(negative_guidance)
        return data


class AddGuidanceSignal(object):
    def __init__(self, field, positive_guidance_field, negative_guidance_field, sigma=2):
        self.field = field
        self.sigma = sigma
        self.positive_guidance_field = positive_guidance_field
        self.negative_guidance_field = negative_guidance_field

    def signal(self, img, pos, neg):
        signal = np.zeros((2, img.shape[-2], img.shape[-1]), dtype=np.float32)
        for p in pos:
            if np.any(np.asarray(p) < 0):
                continue
            signal[0, int(p[-2]), int(p[-1])] = 1.0

        signal[0] = gaussian_filter(signal[0], sigma=self.sigma)
        signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))

        assert np.max(signal[0]) == 1

        for n in neg:
            if np.any(np.asarray(n) < 0):
                continue
            signal[1, int(n[-2]), int(n[-1])] = 1.0

        if np.max(signal[1]) > 0:
            signal[1] = gaussian_filter(signal[1], sigma=self.sigma)
            signal[1] = (signal[1] - np.min(signal[1])) / (np.max(signal[1]) - np.min(signal[1]))

        return signal

    def __call__(self, data):
        img = data[self.field]
        pos = data[self.positive_guidance_field]
        neg = data[self.negative_guidance_field]

        sig = self.signal(img, pos, neg)
        data[self.field] = np.concatenate([img, sig], axis=0)
        return data


#####################################################################################
# FOR 2D
# Following are click-transforms used by batch training/validation step
#####################################################################################
# NOTE:: All the Interaction* Works on batch Data

class InteractionFindDiscrepancyRegions(object):
    def __init__(self, prediction_field, label_field, positive_disparity_field, negative_disparity_field):
        self.prediction_field = prediction_field
        self.label_field = label_field
        self.positive_disparity_field = positive_disparity_field
        self.negative_disparity_field = negative_disparity_field

    def __call__(self, data):
        positive_disparity = []
        negative_disparity = []

        for pred, gt in zip(data[self.prediction_field], data[self.label_field]):
            pred = (pred > 0.5).astype(np.float32)
            gt = (gt > 0.5).astype(np.float32)

            disparity = gt - pred

            negative_disparity.append((disparity < 0).astype(np.float32))
            positive_disparity.append((disparity > 0).astype(np.float32))

        data[self.positive_disparity_field] = positive_disparity
        data[self.negative_disparity_field] = negative_disparity
        return data


class InteractionAddRandomGuidance(object):
    def __init__(self, label_field, positive_guidance_field, negative_guidance_field, positive_disparity_field,
                 negative_disparity_field, p_interact_field):
        self.label_field = label_field
        self.positive_guidance_field = positive_guidance_field
        self.negative_guidance_field = negative_guidance_field
        self.positive_disparity_field = positive_disparity_field
        self.negative_disparity_field = negative_disparity_field
        self.p_interact_field = p_interact_field

    def __call__(self, data):
        positive_guidance = data[self.positive_guidance_field]
        negative_guidance = data[self.negative_guidance_field]

        if type(positive_guidance) is np.ndarray:
            positive_guidance = positive_guidance.tolist()

        if type(negative_guidance) is np.ndarray:
            negative_guidance = negative_guidance.tolist()

        for i, pos_discr, neg_discr, curr_label, probability in zip(
                range(len(data[self.label_field])),
                data[self.positive_disparity_field],
                data[self.negative_disparity_field],
                data[self.label_field],
                data[self.p_interact_field]
        ):
            will_interact = np.random.choice([True, False], p=[probability, 1.0 - probability])
            if not will_interact:
                continue

            can_be_negative = np.sum(neg_discr) > 0
            can_be_positive = np.sum(pos_discr) > 0
            correct_pos = np.sum(pos_discr) >= np.sum(neg_discr)

            if correct_pos and can_be_positive:
                distance = distance_transform_cdt(pos_discr).flatten()
                probability = np.exp(distance) - 1.0
                idx = np.where(pos_discr.flatten() > 0)[0]

                if np.sum(pos_discr > 0) > 0:
                    seed = np.random.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                    dst = distance[seed]

                    pg = np.asarray(np.unravel_index(seed, pos_discr.shape)).transpose().tolist()[0]
                    pg[0] = dst[0]

                    assert curr_label[..., pg[-2], pg[-1]] == 1
                    assert len(pg) == 3

                    negative_guidance[i].append([-1, -1, -1])
                    positive_guidance[i].append(pg)

            if not correct_pos and can_be_negative:
                distance = distance_transform_cdt(neg_discr).flatten()

                probability = np.exp(distance) - 1.0
                idx = np.where(neg_discr.flatten() > 0)[0]

                if np.sum(neg_discr > 0) > 0:
                    seed = np.random.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                    dst = distance[seed]

                    ng = np.asarray(np.unravel_index(seed, neg_discr.shape)).transpose().tolist()[0]
                    ng[0] = dst[0]

                    assert curr_label[..., ng[-2], ng[-1]] == 0
                    assert len(ng) == 3

                    negative_guidance[i].append(ng)
                    positive_guidance[i].append([-1, -1, -1])
            else:
                positive_guidance[i].append([-1, -1, -1])
                negative_guidance[i].append([-1, -1, -1])

        data[self.positive_guidance_field] = np.asarray(positive_guidance)
        data[self.negative_guidance_field] = np.asarray(negative_guidance)
        return data


class InteractionAddGuidanceSignal(AddGuidanceSignal):
    def __init__(self, field, positive_guidance_field, negative_guidance_field, sigma=2, number_intensity_ch=1):
        super().__init__(field, positive_guidance_field, negative_guidance_field, sigma)
        self.number_intensity_ch = number_intensity_ch

    def __call__(self, data):
        images = []
        for img, pos, neg in zip(
                data[self.field],
                data[self.positive_guidance_field],
                data[self.negative_guidance_field]
        ):
            img = img[0:0 + self.number_intensity_ch, ...]  # image can have only number_intensity_ch channels
            signal = self.signal(img, pos, neg)
            images.append(np.concatenate([img, signal], axis=0))

        data[self.field] = images
        return data


class InteractionAddGuidanceSignalClickSize(InteractionAddGuidanceSignal):
    def __init__(self, bins, sigma_multiplier, normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize = normalize
        self.bins = bins
        self.sigma_multiplier = sigma_multiplier

        assert len(self.bins) == len(self.sigma_multiplier)

    @staticmethod
    def digitize(array, bins):
        array = np.asarray(array)
        bins = np.asarray(bins)

        distances = np.sqrt((array[:, np.newaxis] - bins[np.newaxis]) ** 2)
        assignment = np.argmin(distances, axis=1)
        return assignment

    def __call__(self, data):
        img = data[self.field]
        pos = data[self.positive_guidance_field]
        neg = data[self.negative_guidance_field]

        img = img[0:0 + self.number_intensity_ch, ...]  # image can have only number_intensity_ch channels
        signal = np.zeros((2, img.shape[-2], img.shape[-1]), dtype=np.float32)

        positive_interactions = np.asarray(pos)
        positive_interactions = positive_interactions[positive_interactions[:, 0] > 0]

        if len(positive_interactions) > 0:
            level = self.digitize(positive_interactions[:, 0], self.bins)

            for l in range(np.max(level), -1, -1):
                curr_interactions = positive_interactions[level == l]
                for location in curr_interactions:
                    if np.any(location < 0):
                        continue
                    signal[0, int(location[-2]), int(location[-1])] = 1.0

                signal[0] = gaussian_filter(signal[0], sigma=self.sigma * self.sigma_multiplier[l])
                if self.normalize:
                    signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))

            signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))
            assert np.isclose(np.max(signal[0]), 1)

        negative_interactions = np.asarray(neg)
        negative_interactions = negative_interactions[negative_interactions[:, 0] > 0]

        if len(negative_interactions) > 0:
            level = self.digitize(negative_interactions[:, 0], self.bins)

            for l in range(np.max(level), -1, -1):
                current_locations = negative_interactions[level == l]
                for location in current_locations:
                    if np.any(location < 0):
                        continue
                    signal[1, int(location[-2]), int(location[-1])] = 1.0

                signal[1] = gaussian_filter(signal[1], sigma=self.sigma * self.sigma_multiplier[l])
                if self.normalize:
                    signal[1] = (signal[1] - np.min(signal[1])) / (np.max(signal[1]) - np.min(signal[1]))

            signal[1] = (signal[1] - np.min(signal[1])) / (np.max(signal[1]) - np.min(signal[1]))
            assert np.isclose(np.max(signal[1]), 1)

        data[self.field] = np.concatenate([img, signal], axis=0)
        return data
