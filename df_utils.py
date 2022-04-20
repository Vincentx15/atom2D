import Bio.PDB as bio
import numpy as np
import pandas as pd

"""
In this file we make the following transformations easy

"""


def structure_to_df(structure, discard_het=True):
    """
    Function to go from a structure object as defined by biopython to a pandas DF following the notations of ATOM3D
    :param structure:
    :param discard_het:
    :return:
    """
    chain = list()
    resname = list()
    residue_list = list()
    name = list()
    serial_number = list()
    coords = list()
    for i, atom in enumerate(structure.get_atoms()):
        residue = atom.get_parent()
        if not discard_het or residue.id[0] == " ":
            chain.append(residue.get_parent().id)
            resname.append(residue.get_resname())
            residue_list.append(residue.id[1])
            name.append(atom.get_name())
            coord = atom.get_coord()
            serial_number.append(i + 1)
            coords.append(coord)
    chain = np.asarray(chain)
    resname = np.asarray(resname)
    residue = np.asarray(residue_list)
    name = np.asarray(name)
    serial_number = np.asarray(serial_number)
    coords = np.asarray(coords)
    df = pd.DataFrame({'chain': chain,
                       'resname': resname,
                       'residue': residue,
                       'name': name,
                       'serial_number': serial_number,
                       'x': coords[:, 0],
                       'y': coords[:, 1],
                       'z': coords[:, 2]})
    return df


def pdb_to_df(pdb_file_name, discard_het=True):
    parser = bio.MMCIFParser() if pdb_file_name.endswith('.cif') else bio.PDBParser
    structure = parser.get_structure("", pdb_file_name)
    return structure_to_df(structure=structure, discard_het=discard_het)


def df_to_pdb(df, out_file_name):
    """
    Utility function to go from a df object to a PDB file
    :param df:
    :param out_file_name:
    :return:
    """

    def lists_to_pdb(lists, out_file_name, to_print=False):
        def format_name(name: str):
            """
            The right formatting for resname :
            1234
             N
             CA
            HD11
            :param name:
            :return:
            """
            if len(name) <= 3:
                name = name.ljust(3)
            return name.rjust(4)

        def format_float(value: float):
            """
            The right formatting is '3.3'
            """
            integer_part, decimal_part = str(value).split('.')
            return f'{str(integer_part).rjust(3)}.{str(decimal_part[:3]).ljust(3)}'

        with open(out_file_name, 'w') as f:
            for splitted_line in lists:
                serial_number, name, resname, chain, residue, x, y, z, x = splitted_line
                # if float(serial_number) > 90:
                #     iguh
                line = f"ATOM   {serial_number:>4} {format_name(name)} {resname:>3} {chain} {residue:>3}     " \
                       f"{format_float(x)} {format_float(y)} {format_float(z)}  1.00 61.66           {name[0]}"
                if to_print:
                    print(line)
                f.write(line + "\n")

    chain = df['chain']
    resname = df['resname']
    residue = df['residue']
    name = df['name']
    serial_number = df['serial_number']
    x = df['x']
    y = df['y']
    z = df['z']
    lists = zip(serial_number, name, resname, chain, residue, x, y, z, x)
    lists_to_pdb(lists, out_file_name)


if __name__ == '__main__':
    # Get structure and convert it to PDB and a df similar to ones in ATOM3D
    mmcif_parser = bio.MMCIFParser()
    structure = mmcif_parser.get_structure("", "/home/vmallet/.pdbs/4kt3.cif")
    io = bio.PDBIO()
    io.set_structure(structure)
    io.save("data/example_files/from_biopython.pdb")
    df = structure_to_df(structure=structure)
    csv_name = 'data/example_files/4kt3.csv'
    df.to_csv(csv_name)

    # From a df, build a PDB with text formatting
    df = pd.read_csv(csv_name)
    name_test = 'data/example_files/from_db.pdb'
    df_to_pdb(df, name_test)

    # Check that you can open this pdb
    pdb_parser = bio.PDBParser()
    structure = pdb_parser.get_structure("", name_test)
