import json
import torch
import math
import torch.distributed as dist
from typing import TypeVar, Optional, Iterator
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from PIL import Image
import random


def txt_parse(f):
    result = []
    with open(f) as fp:
        line = fp.readline()
        result.append(line)
        while line:
            line = fp.readline()
            result.append(line)
    return result


class OLIVES_HierarchihcalDataset(Dataset):
    def __init__(self, list_file, class_map_file, transform=None):
        with open(list_file, 'r') as f:
            self.data_dict = json.load(f)
        # assert len(data_dict['images']) == len(data_dict['categories'])
        assert len(self.data_dict['images']) == len(self.data_dict['patient_ids'])
        num_data = len(self.data_dict['images'])
        self.transform = transform
        self.augment_transform = transforms.RandomChoice([
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.)),
            transforms.RandomHorizontalFlip(1),
            transforms.ColorJitter(0.4, 0.4, 0.4)])

        with open(class_map_file, 'r') as f:
            self.class_map = json.load(f)
        # self.repeating_product_ids = txt_parse(repeating_product_ids_file)
        self.filenames = []
        # self.category = []
        self.patient_id = []
        self.bcva_list = []
        self.cst_list = []
        self.labels = {}
        for i in range(num_data):
            
            filename = self.data_dict['images'][i]
            # category = self.class_map[data_dict['categories'][i]]
            patient_id = self.class_map[self.data_dict['patient_ids'][i]]

            # product, variation, image = self.get_label_split(filename)
            bcva = self.data_dict['bcva_indexes'][i]
            self.bcva_list.append(bcva)

            cst = self.data_dict['cst_indexes'][i]
            self.cst_list.append(cst)

            week, image = self.get_label_split_OLIVES(filename)

            if patient_id not in self.labels:
                self.labels[patient_id] = {}
            if bcva not in self.labels[patient_id]:
                self.labels[patient_id][bcva] = {}
            if week not in self.labels[patient_id][bcva]:
                self.labels[patient_id][bcva][week] = {}
            
            self.labels[patient_id][bcva][week][image] = i
            self.patient_id.append(patient_id)
            root_data = "/content/drive/MyDrive/IEEE_2023_Ophthalmic_Biomarker_Det"
            self.filenames.append(root_data+filename)
            
        print(self.filenames)

    # def get_label_split(self, filename):
    #     split = filename.split('/')
    #     image_split = split[-1].split('.')[0].split('_')
    #     return int(split[-2][3:]), int(image_split[0]), int(image_split[1])

    def get_label_split_OLIVES(self, filename):
        elements = filename.split("/")
        week = elements[-3][1:]
        image_split = elements[-1].split('.')[0].split('_')[1]
        return int(week), int(image_split)

    # def get_label_split_by_index(self, index):
    #     filename = self.filenames[index]
    #     category = self.category[index]
    #     product, variation, image = self.get_label_split(filename)

    #     return category, product, variation, image

    def get_label_split_by_index_OLIVES(self, index):
        filename = self.filenames[index]
        patient_id = self.patient_id[index]
        bcva = self.data_dict['bcva_indexes'][index]
        week, image = self.get_label_split_OLIVES(filename)

        return patient_id, bcva, week, image

    def __getitem__(self, index):
        images0, images1, labels = [], [], []
        for i in index:
            image = Image.open(self.filenames[i])
            # label = list(self.get_label_split_by_index(i))
            label = list(self.get_label_split_by_index_OLIVES(i))
            if self.transform:
                image0, image1 = self.transform(image)
            images0.append(image0)
            images1.append(image1)
            labels.append(label)

        return [torch.stack(images0), torch.stack(images1)], torch.tensor(labels)

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        #all sub trees end with an int index
        while type(curr_dict) is not int:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while (random_label == label):
                        random_label = random.sample(curr_dict.keys(), 1)[0]
            else:
                random_label = random.sample(curr_dict.keys(), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict

    def __len__(self):
        return len(self.filenames)


class OLIVES_HierarchihcalDatasetEval(Dataset):
    def __init__(self, list_file, class_map_file, transform=None):
        with open(list_file, 'r') as f:
            self.data_dict = json.load(f)

        assert len(self.data_dict['images']) == len(self.data_dict['patient_ids'])

        num_data = len(self.data_dict['images'])

        self.transform = transform
        self.augment_transform = transforms.RandomChoice([
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.)),
            transforms.RandomHorizontalFlip(1),
            transforms.ColorJitter(0.4, 0.4, 0.4)])

        with open(class_map_file, 'r') as f:
            self.class_map = json.load(f)

        self.filenames = []
        # self.category = []
        self.patient_id = []
        self.bcva_list = []
        self.cst_list = []
        self.labels = {}
        for i in range(num_data):
            
            filename = self.data_dict['images'][i]
            # category = self.class_map[data_dict['categories'][i]]
            patient_id = self.class_map[self.data_dict['patient_ids'][i]]

            # product, variation, image = self.get_label_split(filename)
            bcva = self.data_dict['bcva_indexes'][i]
            self.bcva_list.append(bcva)

            cst = self.data_dict['cst_indexes'][i]
            self.cst_list.append(cst)

            week, image = self.get_label_split_OLIVES(filename)

            if patient_id not in self.labels:
                self.labels[patient_id] = {}
            if bcva not in self.labels[patient_id]:
                self.labels[patient_id][bcva] = {}
            if week not in self.labels[patient_id][bcva]:
                self.labels[patient_id][bcva][week] = {}
            
            self.labels[patient_id][bcva][week][image] = i
            self.patient_id.append(patient_id)
            root_data = "/content/drive/MyDrive/IEEE_2023_Ophthalmic_Biomarker_Det"
            self.filenames.append(root_data+filename)
                    

    def get_label_split_OLIVES(self, filename):
        elements = filename.split("/")
        week = elements[-3][1:]
        image_split = elements[-1].split('.')[0].split('_')[1]
        return int(week), int(image_split)

    def get_label_split_by_index_OLIVES(self, index):
        filename = self.filenames[index]
        patient_id = self.patient_id[index]
        bcva = self.data_dict['bcva_indexes'][index]
        week, image = self.get_label_split_OLIVES(filename)

        return patient_id, bcva, week, image


    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = list(self.get_label_split_by_index_OLIVES(index))
        if self.transform:
            image = self.transform(image)

        return image, label

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        #all sub trees end with an int index
        while type(curr_dict) is not int:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while (random_label == label):
                        random_label = random.sample(curr_dict.keys(), 1)[0]
            else:
                random_label = random.sample(curr_dict.keys(), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict

    def __len__(self):
        return len(self.filenames)

class OLIVES_HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: OLIVES_HierarchihcalDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank)

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(
                label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        remaining = list(set(indices).difference(visited))
        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            
            patient_id, bcva, week, image = self.dataset.get_label_split_by_index_OLIVES(
                idx)
            image_index = self.random_unvisited_sample(
                # [category][product][variation]
                image, self.dataset.labels[patient_id][bcva][week], visited, indices, remaining)
            week_index = self.random_unvisited_sample(

                week, self.dataset.labels[patient_id][bcva], visited, indices, remaining)
            
            bcva_index = self.random_unvisited_sample(

                bcva, self.dataset.labels[patient_id], visited, indices,  remaining)
            
            patient_id_index = self.random_unvisited_sample(

                patient_id, self.dataset.labels, visited, indices, remaining)

            batch.extend([patient_id_index, bcva_index,
                          week_index, image_index])
            visited.update([patient_id_index, bcva_index,
                            week_index, image_index])
            remaining = list(set(indices).difference(visited))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
            remaining = list(set(indices).difference(visited))

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
