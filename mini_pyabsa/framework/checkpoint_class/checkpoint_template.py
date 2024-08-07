import os
from pathlib import Path
from typing import Union

from findfile import find_file
from termcolor import colored

from mini_pyabsa.framework.flag_class.flag_template import TaskCodeOption
from mini_pyabsa.framework.checkpoint_class.checkpoint_utils import (
    available_checkpoints,
    download_checkpoint,
)
from mini_pyabsa.utils.file_utils.file_utils import unzip_checkpoint
from mini_pyabsa.utils.pyabsa_utils import fprint

from mini_pyabsa.AspectSentimentTripletExtraction import (
    AspectSentimentTripletExtractor,
)


class CheckpointManager:
    def parse_checkpoint(
        self,
        checkpoint: Union[str, Path] = None,
        task_code: str = TaskCodeOption.Aspect_Polarity_Classification,
    ) -> Union[str, Path]:
        """
        Parse a given checkpoint file path or name and returns the path of the checkpoint directory.

        Args:
            checkpoint (Union[str, Path], optional): Zipped checkpoint name, checkpoint path, or checkpoint name queried from Google Drive. Defaults to None.
            task_code (str, optional): Task code, e.g. apc, atepc, tad, rnac_datasets, rnar, tc, etc. Defaults to TaskCodeOption.Aspect_Polarity_Classification.

        Returns:
            Path: The path of the checkpoint directory.

        Example:
            ```
            manager = CheckpointManager()
            checkpoint_path = manager.parse_checkpoint("checkpoint.zip", "apc")
            ```
        """
        if isinstance(checkpoint, str) or isinstance(checkpoint, Path):
            # directly load checkpoint from local path
            if os.path.exists(checkpoint):
                return checkpoint

            try:
                self._get_remote_checkpoint(checkpoint, task_code)
            except Exception as e:
                fprint(
                    "No checkpoint found in Model Hub for task: {}".format(checkpoint)
                )

            if find_file(os.getcwd(), [checkpoint, task_code, ".config"]):
                # load checkpoint from current working directory with task specified
                checkpoint_config = find_file(
                    os.getcwd(), [checkpoint, task_code, ".config"]
                )
            else:
                # load checkpoint from current working directory without task specified
                checkpoint_config = find_file(os.getcwd(), [checkpoint, ".config"])

            if checkpoint_config:
                # locate the checkpoint directory
                checkpoint = os.path.dirname(checkpoint_config)
            elif isinstance(checkpoint, str) and checkpoint.endswith(".zip"):
                checkpoint = unzip_checkpoint(
                    checkpoint
                    if os.path.exists(checkpoint)
                    else find_file(os.getcwd(), checkpoint)
                )

        return checkpoint

    def _get_remote_checkpoint(
        self, checkpoint: str = "multilingual", task_code: str = None
    ) -> str:
        """
        Downloads a checkpoint file and returns the path of the downloaded checkpoint.

        Args:
            checkpoint (str, optional): Zipped checkpoint name, checkpoint path, or checkpoint name queried from Google Drive. Defaults to "multilingual".
            task_code (str, optional): Task code, e.g. apc, atepc, tad, rnac_datasets, rnar, tc, etc. Defaults to None.

        Returns:
            Path: The path of the downloaded checkpoint.

        Raises:
            SystemExit: If the given checkpoint file is not found.

        Example:
            ```
            manager = CheckpointManager()
            checkpoint_path = manager._get_remote_checkpoint("multilingual", "apc")
            ```
        """
        available_checkpoint_by_task = available_checkpoints(task_code)
        if checkpoint.lower() in [
            k.lower() for k in available_checkpoint_by_task.keys()
        ]:
            fprint(colored("Downloading checkpoint:{} ".format(checkpoint), "green"))
        else:
            fprint(
                colored(
                    "Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints".format(
                        checkpoint
                    ),
                    "red",
                )
            )
        return download_checkpoint(
            task=task_code,
            language=checkpoint.lower(),
            checkpoint=available_checkpoint_by_task[checkpoint.lower()],
        )


class ASTECheckpointManager(CheckpointManager):
    """
    This class manages the checkpoints for Aspect Sentiment Term Extraction.
    """

    def __init__(self):
        """
        Initializes an instance of the ASTECheckpointManager class.
        """
        super(ASTECheckpointManager, self).__init__()

    @staticmethod
    def get_aspect_sentiment_triplet_extractor(
        checkpoint: Union[str, Path] = None, **kwargs
    ) -> "AspectSentimentTripletExtractor":
        """
        Get an AspectExtractor object initialized with the given checkpoint for Aspect Sentiment Term Extraction.

        :param checkpoint: A string or Path object indicating the path to the checkpoint or a zip file containing the checkpoint.
            If the checkpoint is not registered in PyABSA, it should be the name of the checkpoint queried from Google Drive.
        :param kwargs: Additional keyword arguments to be passed to the AspectExtractor constructor.
        :return: An AspectExtractor object initialized with the given checkpoint.
        """
        return AspectSentimentTripletExtractor(
            CheckpointManager().parse_checkpoint(
                checkpoint, TaskCodeOption.Aspect_Sentiment_Triplet_Extraction
            )
        )