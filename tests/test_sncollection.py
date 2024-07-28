from caat import CAAT, SNType


def test_sncollection():
    caat = CAAT().caat

    sntype = caat["Type"].values[0]
    subtypes = list(set(caat[caat["Type"] == sntype].Subtype.values))
    sntype = SNType(type=sntype)

    assert len(sntype.subtypes) == len(subtypes)
    assert len(sntype.sne) > 0
