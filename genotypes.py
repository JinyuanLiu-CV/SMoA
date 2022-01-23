from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'none',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'Spatialattention',
    'Denseblocks',
    'Residualblocks'
]

# epoch 20
genotype_en1 = Genotype(cell=[('sep_conv_3x3', 0), ('Spatialattention', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('Denseblocks', 4), ('Denseblocks', 3)], cell_concat=range(2, 6))
genotype_en2 = Genotype(cell=[('Spatialattention', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('Spatialattention', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], cell_concat=range(2, 6))
genotype_de = Genotype(cell=[('sep_conv_3x3', 0), ('Denseblocks', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('Denseblocks', 4), ('sep_conv_3x3', 0)], cell_concat=range(2, 6))
