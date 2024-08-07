class ASTEModelList(list):
    from .model import EMCGCN

    EMCGCN = EMCGCN

    def __init__(self):
        super(ASTEModelList, self).__init__([self.EMCGCN])
