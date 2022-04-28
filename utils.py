import os


def name_to_dir(pdb_path):
    """
    toto_test.pdb -> t/o/t/o/

    :param pdb_path: a string that starts with a PDB code
    :return:
    """

    return os.path.join(pdb_path[0], pdb_path[1], pdb_path[2], pdb_path[3])


def name_to_path(pdb_path):
    """
    toto_test.pdb -> t/o/t/o/toto_test.pdb

    :param pdb_path: a string that starts with a PDB code
    :return:
    """

    return os.path.join(name_to_dir(pdb_path), pdb_path)
