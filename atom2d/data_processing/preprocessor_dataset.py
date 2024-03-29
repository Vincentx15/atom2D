import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from atom3d.datasets import LMDBDataset
from collections import defaultdict
import numpy as np
import pickle
import time
import torch
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.Atom3DDataset import Atom3DDataset
from data_processing.main import process_df  # noqa

os.environ['OMP_NUM_THREADS'] = '4'  # use one thread for numpy and scipy


def dummy_collate(x):
    return x


class DryRunDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self._lmdb_dataset = LMDBDataset(lmdb_path)

    def __len__(self) -> int:
        return len(self._lmdb_dataset)

    def get_mapping(self, dumpfile=None):
        """
        First let us iterate through the database and collect all systems to preprocess per LMDB 'item'
        Then return a dict {unique_subunit : items}
        We will then be able to process each of those.
        :return:
        """
        dumpfile = os.path.join(self.lmdb_path, 'systems_mapping.p') if dumpfile is None else dumpfile
        if os.path.exists(dumpfile):
            print('Skipping dry run, found a mapping already.')
            return pickle.load(open(dumpfile, 'rb'))
        t0 = time.time()
        loader = torch.utils.data.DataLoader(self,
                                             # num_workers=0,
                                             num_workers=os.cpu_count(),
                                             batch_size=1,
                                             collate_fn=dummy_collate)
        subunits_mapping = defaultdict(list)
        for i, batch_subunits in enumerate(loader):
            for subunit in batch_subunits[0]:
                subunits_mapping[subunit].append(i)
            if not i % 100:
                print(f"Done {i}/{len(self)} in {time.time() - t0}")
        subunits_mapping = dict(subunits_mapping)
        pickle.dump(subunits_mapping, open(dumpfile, 'wb'))
        return subunits_mapping

    def __getitem__(self, index):
        """
        Return a list of subunit for this item.
        :param index:
        :return:
        """
        raise NotImplementedError


class ProcessorDataset(Atom3DDataset):
    def __init__(self, lmdb_path, geometry_path, operator_path, subunits_mapping, recompute=False, verbose=False):
        """

        :param lmdb_path:
        :param geometry_path:
        :param operator_path:
        :param subunits_mapping: the output from a dry run, a dict unique_system : lmdb_id
        """
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)
        self.failed_set = set()
        self.systems_to_compute = [(unique_name, lmdb_ids[0]) for unique_name, lmdb_ids in subunits_mapping.items()]
        self.recompute = recompute
        self.verbose = verbose

    def __len__(self) -> int:
        return len(self.systems_to_compute)

    @staticmethod
    def print_error(name, index, error):
        print("--" * 20)
        print(f'Failed precomputing for {name}, index: {index}')
        print(error)
        print("--" * 20)

    def run_preprocess(self):
        """
        Default class to go through a dataset without collating and batching, for preprocessing
        :param dataset:
        :return:
        """

        # Finally, we need to iterate to precompute all relevant surfaces and operators

        def load_batch(dataloader_iterator):
            try:
                data = next(dataloader_iterator)
                return data
            except StopIteration:
                return None

        print("Running preprocessing")
        n_jobs = max(2 * os.cpu_count() // 3, 1)
        timeout = 20
        executor = ThreadPoolExecutor(max_workers=1)

        loader = torch.utils.data.DataLoader(self, num_workers=n_jobs, batch_size=1, collate_fn=lambda x: x)
        success_codes, failed_list = [], []

        dataloader_iterator = iter(loader)
        for i in tqdm(range(len(loader))):
            try:
                future = executor.submit(load_batch, dataloader_iterator)
                data = future.result(timeout)
                if data is None:
                    break

                success_code, failed = data[0]
                success_codes.append(success_code)
                if failed is not None:
                    failed_list.append(failed)

            except TimeoutError:
                print(f"Skipping batch {i} due to timeout")
                continue

        print(f'{sum(success_codes)}/{len(success_codes)} processed')
        print(list(failed_list))

        # Save the failed set
        with open(os.path.join(self.lmdb_path, 'failed_set.txt'), 'w') as f:
            for name in failed_list:
                f.write(f'{name[0]}, {name[1]}' + '\n')

    def process_lists(self, names, dfs, index):
        """
        Sometimes one element yields several systems
        :param names:
        :param dfs:
        :param index:
        :return:
        """
        for name, dataframe in zip(names, dfs):
            success, outputs = self.process_one(name, dataframe, index)
            if not success:
                return 0, outputs
        return 1, None

    def process_one(self, name, df, index):
        if name in self.failed_set:
            return 0, (name, index)
        try:
            dump_surf_dir = self.get_geometry_dir(name)
            dump_operator_dir = self.get_operator_dir(name)
            is_valid_mesh = process_df(df=df,
                                       name=name,
                                       dump_surf_dir=dump_surf_dir,
                                       dump_operator_dir=dump_operator_dir,
                                       recompute=self.recompute,
                                       verbose=self.verbose,
                                       clean_temp=False  # several chains are always computed,
                                       # making the computation buggy with cleaning
                                       )
            if not is_valid_mesh:
                self.failed_set.add(name)
                self.print_error(name, index, "Invalid mesh")
                return 0, (name, index)
        except Exception as e:
            self.failed_set.add(name)
            self.print_error(name, index, e)
            return 0, (name, index)
        return 1, None

    def __getitem__(self, index):
        # unique_name, lmdb_id = self.systems_to_compute[index]
        # lmdb_item = self._lmdb_dataset[lmdb_id] ...
        raise NotImplementedError


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    dataset = ProcessorDataset(lmdb_path='example',
                               geometry_path='example',
                               operator_path='example_geo')
    dataset.run_preprocess()
