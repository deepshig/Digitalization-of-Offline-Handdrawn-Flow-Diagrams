
class CorrectWord:
    def __init__(self, ):
        def train(lines):
            model = {}
            for line in lines: model[line.split('\t')[0]] = line.split('\t')[1]
            return model

        self.NWORDS = train(file('myDic').read().split('\n'))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def correct(self, word):
        def edits1(word):
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [a + b[1:] for a, b in splits if b]
            transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
            replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
            inserts = [a + c + b for a, b in splits for c in self.alphabet]
            return set(deletes + transposes + replaces + inserts)

        def known_edits2(word):
            return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in self.NWORDS)

        def known(words): return set(w for w in words if w in self.NWORDS)

        candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
        return min(candidates, key=self.NWORDS.get)
