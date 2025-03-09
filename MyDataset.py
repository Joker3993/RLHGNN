import os.path

from dgl.data import DGLDataset
import dgl
from tqdm import tqdm


class MyDataset(DGLDataset):

    def __init__(self,
                 name=None,
                 url=None,
                 raw_dir="./raw_dir",
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 type=None,
                 choice=None):
        self.type = type
        self.choice = choice
        super(MyDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        pass

    def process(self):
        raw_dir_new = self.raw_dir + '/' + self.name

        self.Positive_graph, self.labels = self.load_graph(raw_dir_new)

    def load_graph(self, path):
        Positive_graph_path = os.path.join(path, "part1/" + f"Positive_graph_{self.type}_choice_{self.choice}")
        Positive_graph, laebls_dict_1 = dgl.load_graphs(Positive_graph_path)

        labels = laebls_dict_1['labels']
        print("加载一半数据：-----------")
        print(f"样本数 ：{len(labels)}")

        return Positive_graph, labels

    def __getitem__(self, idx):
        return self.Positive_graph[idx], self.labels[idx]

    def __len__(self):
        return len(self.Positive_graph)

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass
