import os

def atom3dname_to_name(name):
    """
    117e.pdb1.gz_1_A => 117e_1_A.pdb1.gz
    """
    filename = "".join(name.split(".pdb1.gz"))+'.pdb1.gz'
    return filename

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

# name = atom3dname_to_name("117e.pdb1.gz_1_A")
# print(name)