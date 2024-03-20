
import re
import numpy as np

# create mapping to fix multiple names / wrong spellings / etc. landmark names
# some are left untouched
CAESAR_LANDMARK_MAPPING =  {
     '10th Rib Midspine': '10th Rib Midspine',
     'AUX LAND': 'AUX LAND',
     'Butt Block': 'Butt Block',
     'Cervical': 'Cervicale', # FIXED
     'Cervicale': 'Cervicale', 
     'Crotch': 'Crotch',
     'Lt. 10th Rib': 'Lt. 10th Rib',
     'Lt. ASIS': 'Lt. ASIS',
     'Lt. Acromio': 'Lt. Acromion', # FIXED
     'Lt. Acromion': 'Lt. Acromion',
     'Lt. Axilla, An': 'Lt. Axilla, Ant.', # FIXED
     'Lt. Axilla, Ant': 'Lt. Axilla, Ant.', # FIXED
     'Lt. Axilla, Post': 'Lt. Axilla, Post.', # FIXED
     'Lt. Axilla, Post.': 'Lt. Axilla, Post.',
     'Lt. Calcaneous, Post.': 'Lt. Calcaneous, Post.', 
     'Lt. Clavicale': 'Lt. Clavicale',
     'Lt. Dactylion': 'Lt. Dactylion',
     'Lt. Digit II': 'Lt. Digit II',
     'Lt. Femoral Lateral Epicn': 'Lt. Femoral Lateral Epicn',
     'Lt. Femoral Lateral Epicn ': 'Lt. Femoral Lateral Epicn', # FIXED
     'Lt. Femoral Medial Epicn': 'Lt. Femoral Medial Epicn',
     'Lt. Gonion': 'Lt. Gonion',
     'Lt. Humeral Lateral Epicn': 'Lt. Humeral Lateral Epicn',
     'Lt. Humeral Medial Epicn': 'Lt. Humeral Medial Epicn',
     'Lt. Iliocristale': 'Lt. Iliocristale',
     'Lt. Infraorbitale': 'Lt. Infraorbitale',
     'Lt. Knee Crease': 'Lt. Knee Crease',
     'Lt. Lateral Malleolus': 'Lt. Lateral Malleolus',
     'Lt. Medial Malleolu': 'Lt. Medial Malleolus', # FIXED
     'Lt. Medial Malleolus': 'Lt. Medial Malleolus',
     'Lt. Metacarpal-Phal. II': 'Lt. Metacarpal Phal. II', # FIXED
     'Lt. Metacarpal-Phal. V': 'Lt. Metacarpal Phal. V', # FIXED
     'Lt. Metatarsal-Phal. I': 'Lt. Metatarsal Phal. I', # FIXED
     'Lt. Metatarsal-Phal. V': 'Lt. Metatarsal Phal. V', # FIXED
     'Lt. Olecranon': 'Lt. Olecranon',
     'Lt. PSIS': 'Lt. PSIS',
     'Lt. Radial Styloid': 'Lt. Radial Styloid',
     'Lt. Radiale': 'Lt. Radiale',
     'Lt. Sphyrio': 'Lt. Sphyrion', # FIXED
     'Lt. Sphyrion': 'Lt. Sphyrion',
     'Lt. Thelion/Bustpoin': 'Lt. Thelion/Bustpoint', # FIXED
     'Lt. Thelion/Bustpoint': 'Lt. Thelion/Bustpoint',
     'Lt. Tragion': 'Lt. Tragion',
     'Lt. Trochanterion': 'Lt. Trochanterion',
     'Lt. Ulnar Styloid': 'Lt. Ulnar Styloid',
     'Nuchale': 'Nuchale',
     'Rt. 10th Rib': 'Rt. 10th Rib',
     'Rt. ASIS': 'Rt. ASIS',
     'Rt. Acromio': 'Rt. Acromion', # FIXED
     'Rt. Acromion': 'Rt. Acromion',
     'Rt. Axilla, An': 'Rt. Axilla, Ant.', # FIXED
     'Rt. Axilla, Ant': 'Rt. Axilla, Ant.', # FIXED
     'Rt. Axilla, Post': 'Rt. Axilla, Post.', # FIXED
     'Rt. Axilla, Post.': 'Rt. Axilla, Post.',
     'Rt. Calcaneous, Post.': 'Rt. Calcaneous, Post.',
     'Rt. Clavicale': 'Rt. Clavicale',
     'Rt. Dactylion': 'Rt. Dactylion',
     'Rt. Digit II': 'Rt. Digit II',
     'Rt. Femoral Lateral Epicn': 'Rt. Femoral Lateral Epicn',
     'Rt. Femoral Lateral Epicn ': 'Rt. Femoral Lateral Epicn', # FIXED
     'Rt. Femoral Medial Epic': 'Rt. Femoral Medial Epicn', # FIXED
     'Rt. Femoral Medial Epicn': 'Rt. Femoral Medial Epicn',
     'Rt. Gonion': 'Rt. Gonion',
     'Rt. Humeral Lateral Epicn': 'Rt. Humeral Lateral Epicn',
     'Rt. Humeral Medial Epicn': 'Rt. Humeral Medial Epicn',
     'Rt. Iliocristale': 'Rt. Iliocristale',
     'Rt. Infraorbitale': 'Rt. Infraorbitale',
     'Rt. Knee Creas': 'Rt. Knee Crease', # FIXED
     'Rt. Knee Crease': 'Rt. Knee Crease',
     'Rt. Lateral Malleolus': 'Rt. Lateral Malleolus',
     'Rt. Medial Malleolu': 'Rt. Medial Malleolus', # FIXED
     'Rt. Medial Malleolus': 'Rt. Medial Malleolus',
     'Rt. Metacarpal Phal. II': 'Rt. Metacarpal Phal. II',
     'Rt. Metacarpal-Phal. V': 'Rt. Metacarpal Phal. V', # FIXED
     'Rt. Metatarsal-Phal. I': 'Rt. Metatarsal Phal. I', # FIXED
     'Rt. Metatarsal-Phal. V': 'Rt. Metatarsal Phal. V', # FIXED
     'Rt. Olecranon': 'Rt. Olecranon',
     'Rt. PSIS': 'Rt. PSIS',
     'Rt. Radial Styloid': 'Rt. Radial Styloid',
     'Rt. Radiale': 'Rt. Radiale',
     'Rt. Sphyrio': 'Rt. Sphyrion', # FIXED
     'Rt. Sphyrion': 'Rt. Sphyrion',
     'Rt. Thelion/Bustpoin': 'Rt. Thelion/Bustpoint', # FIXED
     'Rt. Thelion/Bustpoint': 'Rt. Thelion/Bustpoint',
     'Rt. Tragion': 'Rt. Tragion',
     'Rt. Trochanterion': 'Rt. Trochanterion',
     'Rt. Ulnar Styloid': 'Rt. Ulnar Styloid',
     'Sellion': 'Sellion',
     'Substernale': 'Substernale',
     'Supramenton': 'Supramenton',
     'Suprasternale': 'Suprasternale',
     'Waist, Preferred, Post.': 'Waist, Preferred, Post.'
    }

SMPL_INDEX_LANDMARKS = {'10th Rib Midspine': 3024,
                        'Cervicale': 829, 
                        'Crotch': 1353, 
                        'Lt. 10th Rib': 1481, 
                        'Lt. ASIS': 3157, 
                        'Lt. Acromion': 1862, 
                        'Lt. Axilla, Ant.': 1871, 
                        'Lt. Axilla, Post.': 2991, 
                        'Lt. Calcaneous, Post.': 3387,
                        'Lt. Clavicale': 1300,
                        'Lt. Dactylion': 2446,
                        'Lt. Digit II': 3222,
                        'Lt. Femoral Lateral Epicn': 1008,
                        'Lt. Femoral Medial Epicn': 1016,
                        'Lt. Gonion': 148,
                        'Lt. Humeral Lateral Epicn': 1621,
                        'Lt. Humeral Medial Epicn': 1661,
                        'Lt. Iliocristale': 677,
                        'Lt. Infraorbitale': 341,
                        'Lt. Knee Crease': 1050,
                        'Lt. Lateral Malleolus': 3327,
                        'Lt. Medial Malleolus': 3432,
                        'Lt. Metacarpal Phal. II': 2258,
                        'Lt. Metacarpal Phal. V': 2082,
                        'Lt. Metatarsal Phal. I': 3294,
                        'Lt. Metatarsal Phal. V': 3348,
                        'Lt. Olecranon': 1736,
                        'Lt. PSIS': 3097,
                        'Lt. Radial Styloid': 2112,
                        'Lt. Radiale': 1700,
                        'Lt. Sphyrion': 3417,
                        'Lt. Thelion/Bustpoint': 598,
                        'Lt. Tragion': 448,
                        'Lt. Trochanterion': 808,
                        'Lt. Ulnar Styloid': 2108,
                        'Nuchale': 445,
                        'Rt. 10th Rib': 4953,
                        'Rt. ASIS': 6573,
                        'Rt. Acromion': 5342,
                        'Rt. Axilla, Ant.': 5332,
                        'Rt. Axilla, Post.': 6450,
                        'Rt. Calcaneous, Post.': 6786,
                        'Rt. Clavicale': 4782,
                        'Rt. Dactylion': 5907,
                        'Rt. Digit II': 6620,
                        'Rt. Femoral Lateral Epicn': 4493,
                        'Rt. Femoral Medial Epicn': 4500,
                        'Rt. Gonion': 3661,
                        'Rt. Humeral Lateral Epicn': 5090,
                        'Rt. Humeral Medial Epicn': 5131,
                        'Rt. Iliocristale': 4165,
                        'Rt. Infraorbitale': 3847,
                        'Rt. Knee Crease': 4535,
                        'Rt. Lateral Malleolus': 6728,
                        'Rt. Medial Malleolus': 6832,
                        'Rt. Metacarpal Phal. II': 5578,
                        'Rt. Metacarpal Phal. V': 5545,
                        'Rt. Metatarsal Phal. I': 6694,
                        'Rt. Metatarsal Phal. V': 6715,
                        'Rt. Olecranon': 5205,
                        'Rt. PSIS': 6521,
                        'Rt. Radial Styloid': 5534,
                        'Rt. Radiale': 5170,
                        'Rt. Sphyrion': 6817,
                        'Rt. Thelion/Bustpoint': 4086,
                        'Rt. Tragion': 3941,
                        'Rt. Trochanterion': 4310,
                        'Rt. Ulnar Styloid': 5520,
                        'Sellion': 410,
                        'Substernale': 3079,
                        'Supramenton': 3051,
                        'Suprasternale': 3171,
                        'Waist, Preferred, Post.': 3021}

def process_caesar_landmarks(landmark_path: str, scale: float = 1000.0):
    """
    Process CAESAR dataset landmarks from .lnd file. 
    Reading file from AUX to END flags.

    :param landmark_path (str): path to landmark .lnd file
    :param scale (float): scale of landmark coordinates

    Return: list of landmark names and coordinates
    :return landmark_dict (dict): dictionary with landmark names as keys and
                                    landmark coordinates as values
                                    landmark_coords are (np.array): (1,3) array 
                                    of landmark coordinates
    """

    landmark_coords = []
    landmark_names = []

    with open(landmark_path, 'r') as file:
        do_read = False
        for line in file:

            # start reading file when encounter AUX flag
            if line == "AUX =\n":
                do_read = True
                # skip to the next line
                continue
                
            # stop reading file when encounter END flag
            if line == "END =\n":
                do_read = False
            

            if do_read:
                # EXAMPLE OF LINE IN LANDMARKS
                # 1   0   1   43.22   19.77  -38.43  522.00 Sellion
                # where the coords should be
                # 0.01977, -0.03843, 0.522
                # this means that the last three floats before 
                # the name of the landmark are the coords
                
                # find landmark coordinates
                landmark_coordinate = re.findall(r"[-+]?\d+\.*\d*", line)
                x = float(landmark_coordinate[-3]) / scale
                y = float(landmark_coordinate[-2]) / scale
                z = float(landmark_coordinate[-1]) / scale
                
                # find landmark name
                # (?: ......)+ repeats the pattern inside the parenthesis
                # \d* says it can be 0 or more digits in the beginning
                # [a-zA-Z]+ says it needs to be one or more characters
                # [.,/]* says it can be 0 or more symbols
                # \s* says it can be 0 ore more spaces
                # NOTE: this regex misses the case for landmarks with names
                # AUX LAND 79 -- it parses it as AUX LAND -- which is ok for our purposes
                landmark_name = re.findall(r" (?:\d*[a-zA-Z]+[-.,/]*\s*)+", line)
                landmark_name = landmark_name[0][1:-1]
                landmark_name_standardized = CAESAR_LANDMARK_MAPPING[landmark_name]

                # * zero or more of the preceding character. 
                # + one or more of the preceding character.
                # ? zero or one of the preceding character.
                
                landmark_coords.append([x,y,z])
                landmark_names.append(landmark_name_standardized)
                
    landmark_coords = np.array(landmark_coords)

    return dict(zip(landmark_names, landmark_coords))