desc_to_icd = {
    "Lip": "C00",
    "Base of tongue": "C01",
    "Other and unspecified parts of tongue": "C02",
    "Gum": "C03",
    "Floor of mouth": "C04",
    "Palate": "C05",
    "Other and unspecified parts of mouth": "C06",
    "Parotid gland": "C07",
    "Other and unspecified major salivary gland": "C08",
    "Tonsil": "C09",
    "Oropharynx": "C10",
    "Nasopharynx": "C11",
    "Pyriform sinus": "C12",
    "Hypopharynx": "C13",
    "Other and ill-defined sites in lip, oral cavity and pharynx": "C14",
    "Nasal cavity and middle ear": "C30",
    "Accessory sinuses": "C31",
    "Larynx": "C32",
    "Thyroid gland": "C73",
}

icd_to_desc = {v: k for k, v in desc_to_icd.items()}
