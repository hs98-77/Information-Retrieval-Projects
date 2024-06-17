import re
from collections import defaultdict
from itertools import combinations

from mrjob.job import MRJob
from mrjob.step import MRStep

WORD_RE = re.compile(r"[\w']+")

def insert_value(k_list, v_list, key, value):
    for i in range(len(v_list)):
        if value > v_list[i]:
            v_list[i+1:len(v_list)-1] = v_list[i:len(v_list)-2]
            k_list[i+1:len(k_list)-1] = k_list[i:len(k_list)-2]
            v_list[i]=value
            k_list[i]=key
            return

top100 = [0 for i in range(100)]
pairs = ["" for i in range(100)]


class MRRelativeFreq(MRJob):

    def mapper(self, _, line):
        vocab = WORD_RE.findall(line)
        for w1 in vocab:
            for w2 in vocab:
                if not w1.lower()==w2.lower():
                    p = [w1.lower(),w2.lower()]
                    p.sort()
                    yield p, 1

    def reducer(self, key, value):
        #sum should be divided in two. beacause value include [a,b] and [b,a] counts
        s_value = int(sum(value)/2)
        insert_value(pairs,top100,key,s_value)
        yield key , s_value


if __name__ == '__main__':
    MRRelativeFreq.run()

print("-----------------------------Here comes the 100 top pairs--------------------------")
for i in range(100):
    print(pairs[i],"\t",top100[i])