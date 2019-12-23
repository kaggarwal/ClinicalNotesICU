# mimic3-text

Clone https://github.com/YerevaNN/mimic3-benchmarks and run all data generation steps to generate training data without text features.

## Text Scripts
1. Run extract_notes.py file under scripts folder.
2. Run extract_T0.py file under scripts folder.

## Configuration
1. Update all paths and configuration in config.py file.

## Models.
1. For IHM run ihm_model.py file under tf_trad.

    Number of train_raw_names:  14681 <br>
    Succeed Merging:  11579 - Model will train on this many episodes as it contains text. <br>
    Missing Merging:  3102 - These texts don't have any text for first 48 hours. 

2. For Decompensation, run decom_los_model.py file under tf_trad.

    Text Not found for patients:  6897 <br>
    Successful for patients:  22353

3. Lenght of Stay, run decom_los_model.py file under tf_trad.

    Successful for episodes for training:  22353
