from mini_pyabsa.utils.data_utils.dataset_item import DatasetItem

class ASTEDatasetList(list):
    """
    The following datasets are for aspect polarity classification task.
    The datasets are collected from different sources, you can use the id to locate the dataset.
    """

    Laptop14 = DatasetItem("Laptop14", "401.Laptop14")
    Restaurant14 = DatasetItem("Restaurant14", "402.Restaurant14")

    Restaurant15 = DatasetItem("Restaurant15", "403.Restaurant15")
    Restaurant16 = DatasetItem("Restaurant16", "404.Restaurant16")

    SemEval = DatasetItem("SemEval", "400.SemEval")

    Chinese_Zhang = DatasetItem("Chinese_Zhang", ["405.Chinese_Zhang"])

    Multilingual = DatasetItem("Multilingual", ["ASTE"])

    def __init__(self):
        super(ASTEDatasetList, self).__init__(
            [
                self.Laptop14,
                self.Restaurant14,
                self.Restaurant15,
                self.Restaurant16,
                self.SemEval,
                self.Chinese_Zhang,
                self.Multilingual,
            ]
        )
