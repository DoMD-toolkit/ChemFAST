"""In-memory storage for the independent 1-2/1-3/1-4 charge pipeline."""

from collections.abc import Callable, Iterable

from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from chemfast.ff.opls.opls_ml.get_data import get_data_edge_mode

ChargeSample = tuple[Chem.Mol, dict[int, float]]


class ChargeInMemoryDataset(InMemoryDataset):
    """Collate path-flow molecular graphs into tensors plus slice offsets.

    ``samples`` is needed only while creating the processed file.  The cache
    name includes every fragment parameter, so the 1-3/1-4 data cannot collide
    with the legacy bond-only dataset or with another ``max_r_cut`` setting.
    """

    def __init__(
            self,
            root: str,
            samples: Iterable[ChargeSample] | None = None,
            r_cut: int = 12,
            r_buf: int = 3,
            max_r_cut: int = 20,
            transform: Callable[[Data], Data] | None = None,
            force_reload: bool = False,
    ) -> None:
        self.samples = samples
        self.r_cut = r_cut
        self.r_buf = r_buf
        self.max_r_cut = max_r_cut
        super().__init__(
            root=root,
            transform=transform,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])
        self.samples = None

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> str:
        return f"charge_graphs_1314_r{self.r_cut}_b{self.r_buf}" f"_max{self.max_r_cut}.pt"

    def process(self) -> None:
        if self.samples is None:
            raise ValueError("samples are required to build the dataset")
        data_list = []
        for mol, partial_charges in tqdm(self.samples):
            try:
                data = get_data_edge_mode(mol, partial_charges, r_cut=self.r_cut, r_buf=self.r_buf,
                                          max_r_cut=self.max_r_cut)
                data_list.append(data)
            except:
                pass
        print(f"Clean data/All data: {len(data_list)}/{len(self.samples)}")
        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    import pickle

    with open("alldatanew.pkl", "rb") as f:
        itp_files = pickle.load(f)

    samples = []
    for i, itp_file in enumerate(tqdm(itp_files)):
        mol = itp_file[0]
        if mol.GetNumAtoms() > 120:
            continue
        fc = Chem.GetFormalCharge(mol)
        labels = {atom.GetIdx(): itp_file[1][atom.GetIdx()][1] for atom in mol.GetAtoms()}
        if abs(fc - sum(labels.values())) > 0.1:
            continue
        if mol.GetNumAtoms() != len(labels):
            continue
        samples.append((mol, labels))

    root = "charge_dataset_13_14"
    ChargeInMemoryDataset(
        root,
        samples,
        r_cut=12,
        r_buf=3,
        max_r_cut=20,
        force_reload=True,
    )
