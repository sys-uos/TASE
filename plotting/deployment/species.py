import math

from TASE.src.core.species import Species


class Phoenicurs_phoenicurus(Species):

    def __init__(self):
        super().__init__()

        self.lat_name = 'Phoenicurus phoenicurus'
        self.eng_name = 'Common Redstart'

        self.mean_territory_size = 34482.7586206897 # in meter²

        self.ground_truth = [[52.00894502,  8.05572935, 200, 'red'],  # unten rechts
                             [52.00981674,  8.05338621, 200, 'red'],  # unten links
                             [52.01072807,  8.05489898, 200 , 'red'],  # mitte
                             [52.01161956 , 8.05601478, 200 , 'red'],  # oben rechts
                             [52.01171201 , 8.05390423, 200, 'orange']]  # oben links
        self.bw = 0.15
        self.heatmap_vmax = 0.00002

class Sylvia_borin(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Sylvia borin'
        self.eng_name = 'Garden Warbler'

        self.mean_territory_size = 25000 # in meter²

        self.ground_truth = [[52.00958088, 8.05551784, 200, 'red'],  # mitte rechts
                             [52.00958088, 8.05351784, 200, 'orange'],  # mitte links
                             [52.00965825, 8.05753827, 200, 'gray'], # outside rechts
                             [52.01126297, 8.05837512, 200, 'gray']] # outside rechts oben (außerhalb map)
        self.bw = 0.15
        self.heatmap_vmax = 0.00002


class Sylvia_atricapilla(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Sylvia atricapilla'
        self.eng_name = 'Eurasian Blackcap'

        self.mean_territory_size = 15151.5151515152 # in meter²

        self.ground_truth = [[52.011567, 8.054888, 175, 'red'], # oben mitte
                             [52.010761, 8.055038, 175, 'red'],  # mitte am teich
                             [52.010352, 8.053915, 175, 'red'], # links
                             [52.009387, 8.054695, 175, 'red'], # mitte an der hecke
                             [52.010068, 8.05801, 175, 'gray'], # outside, an der straße
                             [52.008767, 8.057111, 175, 'gray'] ]  # unten rechts

        self.heatmap_vmax = 0.00002  # used in the evaluation
        self.bw = 0.15

class Troglodytes_troglodytes(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Troglodytes troglodytes'
        self.eng_name = 'Eurasian Wren'

        self.mean_territory_size = 21276.5957446808 # in meter²

        self.ground_truth = [ [52.01111721, 8.05425022,200, 'red'],  # oben links
                              [52.01114410, 8.05565000,200, 'red'],  # oben rechts
                              # [52.01132287, 8.05475594, 150, 'red'],  # oben mitte
                              [52.00900776, 8.05684889,200, 'gray'],  # unten rechts
                              [52.00939079, 8.05308580,200, 'gray']]  # outside unten links
        self.heatmap_vmax = 0.00005
        self.bw = 0.15


class Phylloscopus_collybita(Species):
    def __init__(self):
        super().__init__()
        self.lat_name = 'Phylloscopus collybita'
        self.eng_name = 'Common Chiffchaff'

        self.mean_territory_size = 15625 # in meter²

        self.ground_truth = [[52.011295700, 8.05428906, 200, 'red'], # oben links
                             [52.010969570, 8.05588906, 200, 'red'],  # oben rechts
                             [52.00999618 , 8.05379391, 200, 'red'],  # unten links
                             [52.00885917 , 8.05566154, 200, 'red']] # unten rechts
        self.bw = 0.15
        self.heatmap_vmax = 0.00002


class Muscicapa_striata(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Muscicapa striata'
        self.eng_name = 'Spotted Flycatcher'

        self.mean_territory_size = 42553.1914893617 # in meter²

        self.ground_truth = [[52.011083, 8.054212, 200, 'red'],  # oben links
                             [52.011098, 8.055382, 200, 'red'],  #  mitte rechts
                             [52.008833, 8.056052, 200, 'orange']]  # unten links
        self.bw = 0.15
        self.heatmap_vmax = 0.00002


class Erithacus_rubecula(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Erithacus rubecula'
        self.eng_name = 'European Robin'

        self.mean_territory_size = 16393.4426229508  # in meter²

        self.ground_truth = [ [52.00894502 , 8.05614352,175, 'red'],  # unten rechts
                              [52.01056014, 8.05398703,175, 'red'],  # unten links
                              [52.01093939, 8.05422306,175, 'red'],  # oben links
                              [52.01090637, 8.05570364,175, 'orange'],  # links am teich
                              [52.01157994, 8.05517874,175, 'red'],  # oben rechts
                              # [52.011633, 8.056197, 150, 'orange'],  # not safe, oben rechts
                              [52.010959, 8.058397, 175, 'orange'],  # not safe, mitte rechts
                              [52.008833, 8.053097, 175, 'gray'],   # outside, links unten
                              [52.007862, 8.055661, 175, 'gray']] # outside, rechts unten
        self.bw = 0.15
        self.heatmap_vmax = 0.00002


class Certhia_brachydactyla(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Certhia brachydactyla'
        self.eng_name = 'Short-toed Treecreeper'

        self.mean_territory_size = 55555.5555555556  # in meter²

        self.ground_truth = [[52.00884596, 8.05617034, 200, 'red'],  # unten rechts
                             [52.01011391, 8.05377245, 200, 'red'],  # links unten
                             [52.01105495, 8.05412114, 200, 'red'],  # oben links
                             [52.01097712, 8.05549979, 200, 'red']]  # oben rechts
        self.bw = 0.15
        self.heatmap_vmax = 0.00002


class Fringilla_coelebs(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Fringilla coelebs'
        self.eng_name = 'Eurasian chaffinch'

        self.mean_territory_size = 9478.67298578199  # in meter²

        self.ground_truth = [[52.010986, 8.053837, 200.0, 'red'],  # oben links
                             [52.011286, 8.055837, 200.0, 'red'],  # oben rechts
                             [52.00887238, 8.05587700, 200.0, 'red'],  # unten rechts
                             # [52.00992184, 8.05401250 , 150.0, 'red'],  # unten links
                             [52.01049882, 8.05480242, 200.0, 'red'],  # mitte
                             [52.01128278, 8.05815728, 200.0, 'gray'],  # oben rechts
                             [52.008747, 8.053118, 200, 'gray'],  # outside, left
                             [52.007717, 8.056036, 200, 'gray']]  # outside, right
        self.bw = 0.15
        self.heatmap_vmax = 0.00002

class Turdus_philomelos(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Turdus philomelos'
        self.eng_name = 'Song thrush'

        self.mean_territory_size = 20202.0202020202  # in meter²

        self.ground_truth = [[52.01031486, 8.05474877, 500, 'red'] ]
        self.bw = 0.15
        self.heatmap_vmax = 0.00002


class Anthus_trivialis(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Anthus trivialis'
        self.eng_name = 'Song thrush'

        self.mean_territory_size = 25316.4556962025  # in meter²

        self.ground_truth = [[52.00920258, 8.05560708, 225, 'red']]
        self.bw = 0.15
        self.heatmap_vmax = 0.00002



# ------------------------ Without groundtruth


class Parus_major(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Parus major'
        self.eng_name = 'Great tit'

        self.mean_territory_size = 12269.9386503067  # in meter²


class Cyanistes_caeruleus(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Cyanistes caeruleus'
        self.eng_name = 'Blue tit'

        self.mean_territory_size = 12500  # in meter²


class Columba_palumbus(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Columba palumbus'
        self.eng_name = 'Common wood pigeon' # Common_wood_pigeon


class Phylloscopus_trochilus(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Phylloscopus trochilus'
        self.eng_name = 'Willow warbler'

        self.mean_territory_size = 17391.3043478261  # in meter²



class Dryocopus_martius(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Dryocopus martius'
        self.eng_name = 'Black woodpecker'

        self.mean_territory_size = 20000  # in meter²


class Dendrocopos_major(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Dendrocopos major'
        self.eng_name = 'Great Spotted Woodpecker'

        self.mean_territory_size = 20000  # in meter²


class Regulus_regulus(Species):
    def __init__(self):
        super().__init__()

        self.lat_name = 'Regulus regulus'
        self.eng_name = 'Goldcrest'

        self.mean_territory_size = 40816.3265306122 # in meter²

def evaluation_specs():
    return [Anthus_trivialis(), Certhia_brachydactyla(), Erithacus_rubecula(), Fringilla_coelebs(),
            Muscicapa_striata(), Phoenicurs_phoenicurus(), Phylloscopus_collybita(), Sylvia_atricapilla(),
            Sylvia_borin(), Troglodytes_troglodytes(), Turdus_philomelos()]