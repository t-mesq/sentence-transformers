from typing import Union, List
from sentence_transformers.readers import InputExample


class IRInputExample(InputExample):
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = '', queries: List[str] = None, documents: List[str] = None, label: Union[int, float] = 0, query_first: bool = True):
        """
        Creates one IRInputExample with the given queries, documents, guid and label


        :param guid
            id for the example
        :param queries
            the queries for the example. Note, str.strip() is called on the texts
        :param documents
            the documents for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        :param query_first: bool
            specify if IR examples anchor documents or queries
        """

        queries_mask = len(queries) * ['query']
        documents_mask = len(documents) * ['document']
        if query_first:
            texts = queries + documents
            self.encoder_mask = queries_mask + documents_mask
        else:
            texts = documents + queries
            self.encoder_mask = documents_mask + queries_mask

        super(IRInputExample, self).__init__(guid=guid, texts=texts, label=label)
        self.query_first = query_first
        self.queries = queries
        self.documents = documents

    def __str__(self):
        format_str = "<InputExample> label: {}, queries: {}, documents: {}"
        return format_str.format(str(self.label), ";\n\t".join(self.queries), ";\n\t".join(self.documents))
