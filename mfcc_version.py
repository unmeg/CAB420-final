def wav_to_mfcc(file):
    fs, signal = wav.read(fileurl)
    mfcc_feat = mfcc(signal,fs)
#d_mfcc_feat = delta(mfcc_feat, 2) # change
#fbank_feat = logfbank(sig,rate) # change

    return mfcc_feat