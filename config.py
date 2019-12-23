import os
class Config():
    def __init__(self):
        self.basepath = '/home/expumn_gmail_com/mimic_mnt_backup/mimic3/data/'
        self.data = '../mimic3-benchmarks/data/in-hospital-mortality/'
        self.timestep = 1.0
        self.normalizer_state = '/home/sonu/mimic3-text/mimic3-benchmarks/mimic3models/in_hospital_mortality/ihm_ts1.0.input_str:previous.start_time:zero.normalizer'
        self.imputation = 'previous'
        self.small_part = False
        self.textdata = self.basepath + 'text/'
        self.embeddingspath = '~/embeds/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
        self.buffer_size = 100
        self.model_path = os.path.join(self.basepath, 'wv.pkl')
        self.learning_rate = 1e-4
        self.max_len = 1000
        self.break_text_at = 300
        self.padding_type = 'Zero'
        self.los_path = '/home/sonu/data/length-of-stay/'
        self.decompensation_path = '/home/sonu/data/decompensation/'
        self.ihm_path = '/home/expumn_gmail_com/mimic3-text/mimic3-benchmarks/data/in-hospital-mortality/'
        self.textdata_fixed = '/home/expumn_gmail_com/text_fixed/'
        self.multitask_path = '/home/sonu/data/multitask/'
        self.starttime_path = '/home/expumn_gmail_com/starttime.pkl'
        self.rnn_hidden_units = 256
        self.maximum_number_events = 150
        self.conv1d_channel_size = 256
        self.test_textdata_fixed = '/home/expumn_gmail_com/mimic3-text/mimic3-benchmarks/data/root/test_text_fixed/'
        self.test_starttime_path = '/home/expumn_gmail_com/mimic3-text/mimic3-benchmarks/data/test_starttime.pkl'
        self.dropout = 0.9 #keep_prob

        # Not used in final mode, just kept for reference.
	    self.trainpicklepath = self.basepath + 'train.pkl'
        self.evalpicklepath = self.basepath + 'val.pkl'
        self.patient2hadmid_picklepath = os.path.join(self.basepath,'patient2hadmid.pkl')
        self.trainpicklepath_new = os.path.join(self.basepath , 'train_text_ts.pkl')
        self.evalpicklepath_new = os.path.join(self.basepath , 'val_text_ts.pkl')
        self.num_blocks = 3
        self.mortality_class_ce_weigth = 10