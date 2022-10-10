from datasets import load_dataset

print('begin to handle train.clean.100')
train100 = load_dataset('librispeech_asr', 'all', split='train.clean.100')
train100.save_to_disk('./train100')
print('handle train.clean.100 done')

print('begin to handle train.clean.360')
train360 = load_dataset('librispeech_asr', 'all', split='train.clean.360')
train360.save_to_disk('./train360')
print('handle train.clean.360 done')

print('begin to handle train.other.500')
train500 = load_dataset('librispeech_asr', 'all', split='train.other.500')
train500.save_to_disk('./train100')
print('handle train.other.500 done')

print('begin to handle validation.clean')
vc = load_dataset('librispeech_asr', 'all', split='validation.clean')
vc.save_to_disk('./validation_clean')
print('handle validation.clean done')

print('begin to handle test.clean')
tc = load_dataset('librispeech_asr', 'all', split='test.clean')
tc.save_to_disk('./test_clean')
print('handle test.clean done')

print('begin to handle validation.other')
vo = load_dataset('librispeech_asr', 'all', split='validation.other')
vo.save_to_disk('./validation_other')
print('handle validation.other done')

print('begin to handle test.other')
to = load_dataset('librispeech_asr', 'all', split='test.other')
to.save_to_disk('./test_other')
print('handle test.other done')

