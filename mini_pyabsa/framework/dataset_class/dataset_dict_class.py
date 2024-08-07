class DatasetDict(dict):
    def __init__(self, *args, **kwargs):
        """
        A dict-like object for storing datasets

        :param args: args
        :param kwargs: kwargs

        dataset_dict = {
            'train': [
                {'data': 'This is a text for training', 'label': 'Positive'},
                {'data': 'This is a text for training', 'label': 'Negative'},
            ],
            'test': [
                {'data': 'This is a text for testing', 'label': 'Positive'},
                {'data': 'This is a text for testing', 'label': 'Negative'},
            ],
            'valid': [
                {'data': 'This is a text for validation', 'label': 'Positive'},
                {'data': 'This is a text for validation', 'label': 'Negative'},
            ],
            'dataset_name': str(),
            'column_names': list(),
            'label_names': list(),
        }

        """
        super().__init__(
            train=[],
            test=[],
            valid=[],
            dataset_name="custom_dataset",
            column_names=["text"],
            label_name=["label"],
            *args,
            **kwargs
        )
