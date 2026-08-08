import random

import torch


class MultiLabelEpisodeSampler:
    def __init__(
        self,
        dataset,
        num_episode_labels,
        num_support_per_label,
        num_query,
        seed=None,
    ):
        if num_episode_labels <= 0:
            raise ValueError("num_episode_labels must be positive.")
        if num_support_per_label <= 0:
            raise ValueError("num_support_per_label must be positive.")
        if num_query <= 0:
            raise ValueError("num_query must be positive.")

        self.dataset = dataset
        self.num_episode_labels = num_episode_labels
        self.num_support_per_label = num_support_per_label
        self.num_query = num_query
        self.rng = random.Random(seed)

        labels = torch.as_tensor(dataset.labels, dtype=torch.float32)
        if labels.dim() != 2:
            raise ValueError("dataset.labels must have shape [num_samples, num_labels].")

        self.labels = labels
        self.num_samples, self.num_labels = labels.shape
        self.label_to_indices = []
        for label_idx in range(self.num_labels):
            indices = torch.nonzero(labels[:, label_idx] > 0, as_tuple=False)
            self.label_to_indices.append(indices.flatten().tolist())

        self.available_labels = [
            label_idx
            for label_idx, indices in enumerate(self.label_to_indices)
            if len(indices) >= num_support_per_label
        ]
        if len(self.available_labels) < num_episode_labels:
            raise ValueError("Not enough labels with sufficient positive samples.")

    def sample_episode(self):
        episode_labels = self.rng.sample(
            self.available_labels,
            self.num_episode_labels,
        )
        support_indices = self._sample_support_indices(episode_labels)
        query_indices = self._sample_query_indices(episode_labels, support_indices)

        support = self._stack_samples(support_indices)
        query = self._stack_samples(query_indices)

        return {
            "support_features": support["features"],
            "support_labels": support["labels"],
            "support_sample_ids": support["sample_ids"],
            "query_features": query["features"],
            "query_labels": query["labels"],
            "query_sample_ids": query["sample_ids"],
            "episode_labels": torch.tensor(episode_labels, dtype=torch.long),
        }

    def _sample_support_indices(self, episode_labels):
        support_indices = []
        seen = set()
        for label_idx in episode_labels:
            positives = self.label_to_indices[label_idx]
            candidates = [idx for idx in positives if idx not in seen]
            if len(candidates) < self.num_support_per_label:
                candidates = positives
            selected = self.rng.sample(candidates, self.num_support_per_label)
            for sample_idx in selected:
                if sample_idx not in seen:
                    support_indices.append(sample_idx)
                    seen.add(sample_idx)
        return support_indices

    def _sample_query_indices(self, episode_labels, support_indices):
        support_set = set(support_indices)
        episode_label_tensor = torch.tensor(episode_labels, dtype=torch.long)
        has_episode_label = self.labels[:, episode_label_tensor].sum(dim=1) > 0
        candidates = torch.nonzero(has_episode_label, as_tuple=False).flatten().tolist()
        candidates = [idx for idx in candidates if idx not in support_set]

        if len(candidates) < self.num_query:
            candidates = [idx for idx in range(self.num_samples) if idx not in support_set]
        if len(candidates) < self.num_query:
            raise ValueError("Not enough query candidates for this episode.")

        return self.rng.sample(candidates, self.num_query)

    def _stack_samples(self, indices):
        features = []
        labels = []
        sample_ids = []
        for index in indices:
            sample = self.dataset[index]
            features.append(sample["feature"])
            labels.append(sample["label"])
            sample_ids.append(sample["sample_id"])

        return {
            "features": torch.stack(features, dim=0),
            "labels": torch.stack(labels, dim=0),
            "sample_ids": torch.tensor(sample_ids, dtype=torch.long),
        }
