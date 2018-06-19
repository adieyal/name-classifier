import features

def test_gen_ngrams():
    ngrams = features.gen_ngrams("ABC DEF", 2)
    assert "AB" in ngrams
    assert "BC" in ngrams
    assert "b_AB" in ngrams
    assert "l_BC" in ngrams
    assert "DE" in ngrams
    assert "EF" in ngrams
    assert "b_DE" in ngrams
    assert "l_EF" in ngrams
    assert len(ngrams) == 8

    ngrams = features.gen_ngrams("ABC DEF", 3)
    assert "ABC" in ngrams
    assert "b_ABC" in ngrams
    assert "l_ABC" in ngrams
    assert "DEF" in ngrams
    assert "b_DEF" in ngrams
    assert "l_DEF" in ngrams
    assert len(ngrams) == 6

def test_split_name_parts():
    assert features.split_name_parts("Adi Eyal")["name"] == ["ADI"]
    assert features.split_name_parts("Adi Eyal")["surname"] == ["EYAL"]
    assert features.split_name_parts("Adi Craig Eyal")["name"] == ["ADI", "CRAIG"]
    assert features.split_name_parts("Adi Craig Eyal")["surname"] == ["EYAL"]
    assert features.split_name_parts("Adi-Craig Eyal")["name"] == ["ADI-CRAIG"]
    assert features.split_name_parts("Adi Craig Eyal")["surname"] == ["EYAL"]

    assert features.split_name_parts("Adi van der Stel")["name"] == ["ADI"]
    assert features.split_name_parts("Adi van der Stel")["surname"] == ["VAN", "DER", "STEL"]
    assert features.split_name_parts("Adi von Nel")["surname"] == ["VON", "NEL"]

    assert features.split_name_parts("Adi Janse van Rensburg")["name"] == ["ADI"]
    assert features.split_name_parts("Adi Janse van Rensburg")["surname"] == ["JANSE", "VAN", "RENSBURG"]

    assert features.split_name_parts("Adi du Preez")["name"] == ["ADI"]
    assert features.split_name_parts("Adi du Preez")["surname"] == ["DU", "PREEZ"]

def test_gen_features():
    name = "Adi Eyal"
    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=False,
        include_trigrams=False,
        include_4grams=False,
        include_surname=False,
        include_fullname=False)
    assert len(name_features) == 0

    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=False,
        include_trigrams=False,
        include_4grams=False,
        include_surname=False,
        include_fullname=True)
    assert len(name_features) == 1
    assert "name_ADI_EYAL" in name_features

    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=False,
        include_trigrams=False,
        include_4grams=False,
        include_surname=True,
        include_fullname=False)
    name_features = features.gen_features(name, include_name_part=False, include_bigrams=False, include_trigrams=False, include_4grams=False, include_surname=True, include_fullname=False)
    assert len(name_features) == 1
    assert "surname_EYAL" in name_features

    name_features = features.gen_features(name,
        include_name_part=True,
        include_bigrams=False,
        include_trigrams=False,
        include_4grams=False,
        include_surname=False,
        include_fullname=False)
    assert len(name_features) == 2
    assert "namepart_ADI" in name_features
    assert "namepart_EYAL" in name_features

    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=True,
        include_trigrams=False,
        include_4grams=False,
        include_surname=False,
        include_fullname=False)
    assert len(name_features) == 9
    ngrams = ["AD", "DI", "EY", "YA", "AL", "b_AD", "b_EY","l_DI", "l_AL"]
    for ngram in ngrams:
        assert ngram in name_features
    assert len(name_features) == len(ngrams)

    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=False,
        include_trigrams=True,
        include_4grams=False,
        include_surname=False,
        include_fullname=False)
    assert len(name_features) == 7
    ngrams = ["ADI", "EYA", "YAL", "b_ADI", "l_ADI", "b_EYA", "l_YAL"]
    for ngram in ngrams:
        assert ngram in name_features
    assert len(name_features) == len(ngrams)

    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=False,
        include_trigrams=False,
        include_4grams=True,
        include_surname=False,
        include_fullname=False)
    assert len(name_features) == 3
    ngrams = ["EYAL", "b_EYAL", "l_EYAL"]
    for ngram in ngrams:
        assert ngram in name_features
    assert len(name_features) == len(ngrams)

def test_compound_surnames():
    name = "Adi du Preez"
    name_features = features.gen_features(name,
        include_name_part=False,
        include_bigrams=False,
        include_trigrams=False,
        include_4grams=False,
        include_surname=True,
        include_fullname=False)
    
    ngrams = ["surname_DU_PREEZ"]
    for ngram in ngrams:
        assert ngram in name_features
    assert len(name_features) == len(ngrams)
