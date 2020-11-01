import numpy as np
class Transformer(object):
    '''
    Encoder and Decoder with bi-directional CNN multi-head attention is the main innovation of the Transformer model.

    A secondary important feature is the use of positional encoding to mark the position of each word in a sentence.

    Encoder:
        Two sub-layers 1) multi-head attention and 2) fully connected layer. 
        A residual connection is applied to each of the two sub-layers, followed by layer normalization.
    Decoder:
        The same two sub-layers "with masking" and an addition layer 3) multi-head attention over the output of the encoder stack.

      
    The NN() class can actually initiate Numpy neural network training, just haven't got time to merge it with the Transformer().
    
    The helper() class is to compute NLP metrics and implement beam search.
    '''
    def __init__(self, param):
        # initialise parameters here (so far this is a numpy pseudo code)
        self.d_emb = 512 # the embedding dimension
        self.d_words = 20 # arbitrary for now
        self.d_k = 10 # dimension of query and key vectors, arbitrary for now
        self.d_v = self.d_k # dimension of value vector, arbitrary for now
    def _position_function(self, d_words, d_emb):
        P = np.empty((self.d_word, self.d_emb))
        for pos in range(self.d_words):
            for i in range(0, self.d_emb, 2):
                P[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_emb)))
                P[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_emb)))

    def _position_encoder(self, X):
        '''
        word embedding vector + positional encoding vector = word with positional encoding vector
        X[:d_words][:d_emb]: input matrix
        P[:d_words][:d_emb]

        Dimensions:
        d_words: number of words in a sentence
        d_emb: embedding vector dimensions
        '''
        # this just assigns each word with a wave function, with words at 
        # later position with higher frequency and a smaller wavelength
        P = _position_function(self.d_words, self.d_emb)  # each word has position vector computed by two functions, a sine and cosine
        X += P
        return X

    def _attention(self, X_in):
        '''
           The multihead_attention() is for the encoder block
           Embedding matrix:
           X[:d_words][:d_emb]: embedding matrix, a list of word embedding vectors each of dimension 512. The longest sentence 
                    in the training dataset will be the list size. 
                    Notice, the encoder input and output (query, key, value) has dimension much less than 512. 
        
           Training weight matrices:
           W_Q[:d_emb][:d_k]: weight matrix
           W_K[:d_emb][:d_k]: weight matrix
           W_V[:d_emb][:d_k]: weight matrix
           Q[:d_words][:d_k]: output token, e.g., "la", 
           K[:d_words][:d_k]: input token, e.g., "the", "girl"
           V[:d_words][:d_v]: input token, e.g., 
           
           Dimension parameters:
           d_k: dimension of "query" and "key" vector
           d_v: dimension of "value" vector

           Notes:
           For every output (input) position, generate a query (key) vector, each containes dimension d_k.
           The relevance score is computed by the dot product of the query and key vector.
           Use the softmax to normalize the relevance score (i.e., probability of the value), 
           and do a weighted sum (average) of the values to get the final output

        I am applying matrix form in the real code, but
        here are the codes to visualize how each word gets 
        its attention in the input sequence. Notice for multihead, there will be 8 such attention
        computed for each word and weighted averaged to get the final attention for each word. 
        This multihead attention is to avoid attention focusing too much on the word itself, 
        and give relationship to other word. The attention vector is then fed to the feed-forward
        net one word at a time (this can be parallelized):
        

        for i in range(self.sequence_length):
            Q[i] = X[i] * W_Q # each word embedding vector flows in the self attention layer, W_Q is the weight matrix for training
            K[i] = X[i] * W_K 
            V[i] = X[i] * W_V 
            for j in range(self.sequence_length):
                # inner product all key vectors are applied on all queries, this is the weight assigned to the "values"
                # the sqrt(d_k) counteracts the large values given by dot product
                relevance[j] = np.dot(Q[i], K[j])/sqrt(d_k) 
                # the attention of ith word comes from the weighted sum of relevance of all other words
                # the irrelevant word values will get very small relevance score and gets drowned out
                attention[i] += softmax(relevance[j]) * value[j] 

        '''               
        Q = X * W_Q # (d_words x d_k) matrix, this part can be parallelized with word fed simultaneously
        K = X * W_K
        V = X * W_V # (d_words x d_v) matrix
        Z = self.softmax((Q * K.T)/np.sqrt(d_k)) * V # (d_words x d_v) matrix, this is attention matrix
        return Z

    def _masked_multihead_attention(self, currindex):
        # code for masked multihead attention here
        '''
        The _masked_multihead_attention() is for the decoder block
        '''
        mask =np.empty((self.d_words, self.d_emb))
        mask[currindex:][:] = float('-inf')) # mask the words following the current index
        Zi = []
        for _ in range(h):
            Zi = np.concatenate((Zi,self._attention(X)+mask), axis=2) # each Z has dim (d_words x d_v) so (d_words, h*d_v), probably apply linked list here to accelerate
        Zo = Zi * Wo # (d_words, d_emb) = (d_words,h*d_v) * (h*d_v, d_emb)
        return Zo

    def _multihead_attention(self, X, h=8):
        # code for multihead attention here
        '''
        The multihead_attention() is for the decoder block
        X[d_words][:d_emb]: the sequence word embedding matrix
        Wo[:h*d_v][:d_emb]
        Zi[:d_words][:h*d_v]
        Zo[:d_words][:d_emb]

        Dimensions:
        d_v: 
        d_emb: the embedding vector size
        h: number of heads
        '''
        Zi = []
        for _ in range(h):
            Zi = np.concatenate((Zi,self._attention(X)), axis=2) # each Z has dim (d_words x d_v) so (d_words, h*d_v), probably apply linked list here to accelerate
        Zo = Zi * Wo # (d_words, d_emb) = (d_words,h*d_v) * (h*d_v, d_emb)
        return Zo

    def _layer_norm(self,M):
        '''
        args: 
           M[:d_words][:d_emb]: matrix to apply layer normalization for "stabilization"
        '''
        M_mean = np.mean(M, axis = 1)
        M_std  = np.std(M, axis = 1)
        M_normalized = (M-M_mean)/M_std # standard normal for now
        return M_normalized
         
    def _position_wise_feed_forward(self, Z, W1, W2, b1, b2):
        Y = max(Z*W1 + b1, 0)*W2 + b2 # Z[:d_words, :d_emb], W1[:d_emb,:nodes], W2[:nodes,:emb]
        return Y

    def _encoder(self, X, layers=6 ):
        '''
        Encoder consists of 6 layers, each layer has these sublayers:
         position encoder -> multihead attention -> residual -> layernorm -> feed forward 
        '''
        for i in range(layers)
            X = self._position_encoder(X) # (d_words, d_emb)
            Z = self._multihead_attention(X) # (d_words, d_emb)
            Z = self._layer_norm(X + Z) # X+Z is the residual connection
            Y = self._position_wise_feed_forward(Z, W1, W2, b1, b2) # (d_words, d_emb)
            Y = self._layernorm(Z + Y) # residual connection and layer normalization
            X = Y 
        return Y  # this will be fed to decoder

    def _decoder(self, X, Ye):
        '''
        args:
          Ye: encoder output
        return:
          Y: the probability vector of the dictionary
        '''
        for i in range(layers)
            X = self._position_encoder(X) # (d_words, d_emb)
            Z1 = self._masked_multihead_attention(X) # (d_words, d_emb)
            Z2 = self._multihead_attention(Ye + Z1) # (d_words, d_emb), an additional layer for encoder attention
            Z = self._layer_norm(Z1 + Z2)
            Y = self._position_wise_feed_forward(Z, W1, W2, b1, b2) # (d_words, d_emb)
            X = Y 
         Y = self._linear(Y)
         Y = self.softmax(Y)
         return Y
    def make_transformer(self, Xi, Xo):
        '''
        Xi: input embedding
        Xo: output embedding
        '''
        # take the output of encoder and merge with decoder inputs
        Y1 = self._encoder(X1)
        Y = self._decoder(Y1,X2) # decoder return 
    def make_loss(self):
        '''
         reuse the codes from NN() class methods 
        '''
    def initialize_network(self):
        '''
         initialize the session and variables reusing the codes from NN() class methods
        '''
    def run(self, args):
        # code to run a single input sequence and generate outputs
        # can use this function to both train and infer only
    def save_model(self):
        # save the model according to requirements



class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.is_word = False
        
class Trie(object):
    '''
    This is the data structure to implement the bleu score in helper class methods
    '''
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        root = self.root
        for w in word:
            root = root.childeren.setdefault(w, TrieNode())
        root.is_word = True   

    def search(self, word):
        root = self.root
        for w in word:
            if w in root.children:
                root = root.children[w] 
            else:
                return False
        return root.is_word

    def prefix(self, word): 
        root = self.root
        for w in word:
            if w in root.children:
                root = root.children[w] 
            else:
                return False
        return True

class helper(object):
    '''
    methods:
       beam search (algos: Quickselect, Max heap) 
       combined_bleu_score (algos: Trie search). Note: bleu is not the best metric for NLP
    '''
    def combined_bleu_score(self, N):
        '''
        average the modified precision scores (bleu score) of the n-grams (n=1,...,N) 
        Utilize the Trie data structure to accelerate the search for n-grams O(M) where M is the maximum n-grams characters
        
        Example:
        true_seq = ["The", "cat", "is", "on", "the", "mat"].
        pred_seq = ["The", "cat", "the", "cat", "on", "the", "mat"].
        
        pred 2-grams   pred-count    clipped-count
        -------------  ----------    -------------
        the cat          2              1
        cat the          1              0
        cat on           1              1
        on the           1              1
        the mat          1              1 
        
        precision = sum(clipped-count)/sum(pred-count) = 4/6

        Bleu score weakness:
             It doesn’t consider meaning:
                 BLEU does not measure meaning. It only rewards systems for n-grams
                 that have exact matches in the reference system. That means that a difference
                 in a function word (like “an” or “on”) is penalized as heavily as a difference 
                 in a more important content word. It also means that a translation that had a
                 perfectly valid synonym that just didn’t happen to show up in the reference
                 translation will be penalized.
                 human: I ate the apple
                 Based on BLEU, these are all “equally bad” output sentences.
                 I consumed the apple.
                 I ate an apple.
                 I ate the potato.
             It doesn’t directly consider sentence structure
             It doesn’t handle morphologically rich languages well
             It doesn’t map well to human judgements

        Bleu alternatives (some are better!):
             NIST: weights n-grams based on rareness, ie., correctly matching rare n-grams improves the scroe more than matching common words.
             Perplexity: measures how well the learned probability distribution of words matches that of the input text.
             WER (Word error rate): measures the number of substitutions (“an” for “the”), deletions and insertions in the output sequence given a reference input.
             ROUGE: focus on recall TP/(TP+FN) rather than precision TP/(TP+FP), ie., how many n-grams in the reference translation show up in the output, rather than the reverse.
                    probably use f1 = average of Bleu and Rouge.
             f1 score: mean of precision (how many predictions were correct) and recall (how many of the possible correct predictions were made)

        Bleu alternatives for seq2seq tasks:
             STM: subtree metric, compares the parses for the reference and output translations and penalizes outputs with different syntactic structures.
             METEOR: similar to BLEU but includes additional steps, like considering synonyms and comparing the stems of words
                     (so that “running” and “runs” would be counted as matches). Also unlike BLEU, it is explicitly designed to use to compare sentences rather than corpora.
             TER: or translation error rate, measures the number of edits needed to chance the original output translation into an acceptable human-level translation.
             TERp: or TER-plus, is an extension of TER that also considers paraphrases, stemming, and synonyms.
             hLEPOR: a metric designed to be better for morphologically complex languages like Turkish or Czech.
                     Among other factors it considers things like part-of-speech (noun, verb, etc.) that can help capture syntactic information.
             RIBES: like hLEPOR, doesn’t rely on languages having the same qualities as English.
                    It was designed to be more informative for Asian languages―like Japanese and Chinese―and doesn’t rely on word boundaries.
             MEWR: probably the newest metric on the list, is one that I find particularly exciting: it doesn’t require reference translations!
                   This is great for low resource languages that may not have a large parallel corpus available.
                   It uses a combination of word and sentence embeddings (which capture some aspects of meaning) and perplexity to score translations.
        '''
        def get_word_grams_freq(seq, trie, n):
            word_freq = defaultdict(lambda:0)
            for i in range(len(seq)-n): # given a 3 word sentence, a bi-gram will go through the index set {[0,1], [1,2]}, which is range(1)
                word_grams = str(seq[i:i+n])
                if trie.prefix(word_grams):
                    word_freq[word_grams]+=1
            return word_freq

        def brevity_penalty(true_seq_len, pred_seq_len):
            '''
            Penalize for predicted short sentences, which complements for precision metric which penalizes for longer than reference predicted sentences
            '''
            B = 1
            if true_seq_len > pred_seq_len:
                B = math.exp(1 - true_seq_len / pred_seq_len)
            return B

        def bleu_score(trie, n, true_seq, pred_seq):
            '''
            Compute the single n-grams bleu_score 
            Suppose only one true_seq, there could be more
            '''
            word_freq_true = get_word_grams_freq(true_seq, trie, n)
            word_freq_pred = get_word_grams_freq(pred_seq, trie, n)
            pred_count = len(pred_seq)-n
            for word_grams in word_freq_pred:
                if word_grams in word_freq_true: 
                   clipped_count += min(word_freq_true[word_grams], word_freq_pred[word_grams])
            precision = clipped_count/pred_count # TP/(TP+FP) 
            B = self.brevity_penalty(len(true_seq), len(pred_seq))
            Bleu = B*precision
            return Bleu
        trie = Trie() # reduce word sequence search time
        # build the trie for true_seq on maximum n-grams
        word_freq_true = defaultdict(lambda:0)
        for i in range(len(true_seq)-N): # given a 3 word sentence, a bi-gram will go through the index set {[0,1], [1,2]}, which is range(1)
            word_grams = str(true_seq[i:i+N])
            trie.insert(word_grams)  # not considering space in word and case insensitive 
        Bleu = 0
        for n in range(1,N+1):
            Bleu += bleu_score(trie, n, true_seq, pred_seq)
        return Bleu/N

    def beam_search(self, model, x, k):
        '''
        Define the beam size k and 
        use "Teacher forcing", previous output y_prev as model input 
        for the currect output y_curr, which gives the p(y_curr| vec(y_prev), x).
        '''
        def _partition(nums, wordInd, l, r):
            for i in range(l, r):
                if nums[i]<=nums[r]:
                    nums[i], nums[l] = nums[l], nums[r]
                    wordInd[i], wordInd[l] = wordInd[l], wordInd[i] 
                    l+=1
            nums[l], nums[r] = nums[r], nums[l]
            wordInd[l], wordInd[r] = wordInd[r], wordInd[l] 
            return l

        def quick_select(nums, wordInd, k): # average O(n), worse O(n2)
            n = len(nums)-k # the index of the top kth sorted number
            while l<=r:
                mid = _partition(nums, wordInd, l, r)
                if mid ==n:
                    return mid
                elif mid < n:
                    r = mid - 1
                else:
                    l = mid + 1
        def heapify(nums, n=None):
            '''
            Time complexity: O(n)
            max heapify according to the 0th column (the value) in nums
            args:
              nums: candidates 
              n: the internal node to start heapify from top to bottom
            [1,3,7,3,2,1]
            [7,3,3,2,1,1]
            '''
            N = len(nums)-1
            if not n:
                n = (N-1)//2 # the first non-leaf internal node is the default node to start heapifying
            while n>=0:
                largest = n
                while True:
                    p = largest # parent node
                    l = 2*p+1 # left child node
                    r = 2*p+2 # right child node
                    largest = p
                    if l<=N and nums[p][0] < nums[l][0]:
                        largest = l
                    if r<=N and nums[p][0] < nums[r][0]:
                        largest = r
                    if largest!=p:
                        nums[p], nums[largest] = nums[largest], nums[p]
                    else:
                        break
                n-=1
            return nums 
        def heappop(nums): 
            '''
            Time complexity: O(logn)
            return the root and heapify 
            ''' 
            res = nums[0] # this is the max root to return
            nums[0] = nums.pop() # move the last leaf to the root and heapify
            heapify(nums, n=0)
            return res    
        nwords = 300000 # the dictionary size
        y = [word2vec(x)]*k # transform into word embedding vecotrs, and make k copies
        p = [1]*k # set the input probability to 1 and make k copies
        all_candidtates = []
        while y:
            candidates = [] # this is the candidate joint probability vector (beams) 
            for y_prev, p_prev in y, p: 
                wordInd = list(range(len(nwords))) # the word index
                p_curr = model(y_prev) # y output node has all 300,000 word probability in dictionary
                quick_select(p_prev*p_curr, wordInd, k) # get top k prediction scores using quick select 
                y_curr = word2vec(wordInd[::-1][:k]) # pick top k probability word index from the beams of all words 
                candidates.append((p_prev*p_curr[::-1][:k], y_prev, y_curr)) 
            #quick_select( [c[0] for c in candidates], candidates, k ) # average O(n), get the top k candidates, either use this or heap select
            heapify(candidates) # O(n)
            cand=[]
            for _ in range(k):
                cand.append(heappop(candidates)) # O(klogn)
            y = [c[2] for c in cand]
            p = [c[0] for c in cand]
            all_candidates.append(cand)
        return all_candidates
                     

