#An object to run a WF calculation using molpro.


class MolproExt:

    def __init__(self, x):
        molpro_info = None

    def update_emb_pot(self, x):
        return False

    def run(self, x):
        return False

    def generate_input_file(self, x):
        return False
