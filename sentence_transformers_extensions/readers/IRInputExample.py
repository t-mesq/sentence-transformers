from typing import Union, List
from sentence_transformers.readers import InputExample


class IRInputExample(InputExample):
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = '', texts: List[str] = None, label: Union[int, float] = 0, query_first: bool = True):
        """
        Creates one IRInputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        :param query_first: object
            specify if IR examples anchor documents or queries
        """
        super(IRInputExample, self).__init__(guid=guid, texts=texts, label=label)
        self.query_first = query_first

    def __str__(self):
        format_str = "<InputExample> label: {}, query: {}, documents: {}" if self.query_first else "<InputExample> label: {}, document: {}, query: {}"
        return format_str.format(str(self.label), self.texts[0], "; ".join(self.texts[1:]))
