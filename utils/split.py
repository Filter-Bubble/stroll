import torch
from torch.utils.data import random_split

from stroll.conllu import ConlluDataset

torch.manual_seed(43)

sonar = ConlluDataset('sonar1_fixed_19class.conllu')

train_length = int(0.90 * len(sonar))
test_length = len(sonar) - train_length


train_set, test_set = random_split(sonar, [train_length, test_length])

with open('test.conllu', 'w') as f:
    for s in test_set:
        f.write(s.__repr__() + '\n\n')

with open('train.conllu', 'w') as f:
    for s in train_set:
        f.write(s.__repr__() + '\n\n')

print('Total length', len(sonar))
print(sonar.statistics())
print('Test length', test_length)
print('Train length', train_length)

test = ConlluDataset('test.conllu')
quick_length = int(0.10 * len(test))
remainder = len(test) - quick_length

print('Test length', len(test))
print('Quick length', quick_length)
print('Remainder', remainder)

quick_set, _ = random_split(test, [quick_length, remainder])

with open('quick.conllu', 'w') as f:
    for s in quick_set:
        f.write(s.__repr__() + '\n\n')
