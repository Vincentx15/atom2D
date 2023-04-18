import Bio.PDB as bio

import pandas as pd

from atom3d.util.formats import bp_to_df, df_to_bp

"""
In this file we make the following transformations easy
df <->pdb
"""


def pdb_to_df(pdb_file_name):
    parser = bio.MMCIFParser() if pdb_file_name.endswith('.cif') else bio.PDBParser
    structure = parser.get_structure("", pdb_file_name)
    return bp_to_df(bp=structure)


def df_to_pdb(df, out_file_name, discard_hetatm=True):
    """
    Utility function to go from a df object to a PDB file
    :param df:
    :param out_file_name:
    :return:
    """

    def filter_notaa(struct):
        """
        Discard Hetatm, copied from biopython as as_protein() method is not in biopython 2.78
        :param struct:
        :return:
        """
        remove_list = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != ' ' or not bio.Polypeptide.is_aa(residue):
                        remove_list.append(residue)

        for residue in remove_list:
            residue.parent.detach_child(residue.id)

        for chain in struct.get_chains():  # Remove empty chains
            if not len(chain.child_list):
                chain.parent.detach_child(chain.id)
        return struct

    structure = df_to_bp(df)
    structure = filter_notaa(structure) if discard_hetatm else structure
    io = bio.PDBIO()
    io.set_structure(structure)
    io.save(out_file_name)


if __name__ == '__main__':
    # Get structure and convert it to PDB and a df similar to ones in ATOM3D
    csv_name = '../data/example_files/4kt3.csv'
    pdb_path = "../data/example_files/4kt3.pdb"

    # PDB => csv
    pdb_parser = bio.PDBParser()
    structure = pdb_parser.get_structure("", pdb_path)
    df = bp_to_df(bp=structure)
    df.to_csv(csv_name)

    # csv => df => PDB
    df = pd.read_csv(csv_name, keep_default_na=False)  # keep default is key otherwise bugged
    name_test = '../data/example_files/from_db.pdb'
    df_to_pdb(df, name_test)

    # Check that you can open this pdb
    structure = pdb_parser.get_structure("", name_test)
