import os
import sys
import pathlib
import warnings
import functools
import io
import re
import string 

# Allow usage of TF without the warnings due to failure of GPU registration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from collections import Counter
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV,train_test_split
    from sklearn.decomposition import LatentDirichletAllocation,PCA
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.backend import clear_session
    import tensorflow as tf 
    from tensorflow.keras.layers import TextVectorization
    from tensorflow.keras import layers, losses
except (ImportError, AttributeError, NameError, ModuleNotFoundError): 
    raise Exception('Unknown module')   



'''
Latent LSTM Allocation algorithn implementation.
See Zaheer et al. (2021)
'''

STDOUT = sys.stdout
CUSTOM_STOPWORDS = pathlib.Path(__file__).parent.joinpath('data/custom_stopwords_list.txt')
SRC = pathlib.Path(__file__).parent.joinpath('data/bttf.xlsx')


class Decorators(object):
    def __init__(self):
        self._stdout    = None
        self._string_io = None
        self._counter   = 0
    def __enter__(self):
        self._stdout = sys.stdout
        self._string_io = io.StringIO()
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
    def __eq__(self): 
        return self._counter
    def __str__(self):
        self._counter+=1
        return self._string_io.getvalue()

    @staticmethod
    def progress_bar(j, count, msg:str):
        size = int(50)
        x = int(size*(j+1)/count)
        print(f"{msg}: [{u'█'*x}{('.'*(size-x))}] {int(100*(j+1)/count)}%", 
              end='\r', file=STDOUT, flush=True)
        
    @staticmethod
    def header(msg):
        def title(object):
            @functools.wraps(object)
            def gr_wrapper(*args, **kwargs):
                styling = '# '
                print('\n' + styling  + msg + styling +'\n', 
                      flush=True ,file=STDOUT) 
                return object(*args, **kwargs)
            return gr_wrapper
        return title


class Preprocess(object):
    ''' Preprocess already semi-structured data prior to topic extraction
        Input:
            < *.xlsx
        Output: 
            <pandas.dataframe.object>
            >> index|timecode|part|dialogue|cleaned_text|vocab_set_size|label_predicted
            <matplotlib.pyplot.object>
            >> pertopic_LDAclass_hbar | topic_distrib_overtime_scatter
            >> eval_LDAclass_pca_3dproj | eval_LSTMclass_train-test_loss_line
    '''

    def __init__(self, src:pd.DataFrame): 
        self.src = src.dropna().reset_index()

    @staticmethod
    def get_from_custom_stopwords() -> tuple[list]:
     # data/custom_stpwrd.txt contains 2 section tags ['## Exclude', '## Include']
     # under which words must be appropriately listed
     with open(CUSTOM_STOPWORDS, 'r+t', encoding='utf-8') as f:
          marker = re.compile('^(#){2}\s[A-Z]+')
          pos = {}
          id = 0
          exclude, include = [], []
     
          for _ in f.readlines():
               if marker.search(_) != None: 
                    pos[id] = (marker.search(_).group())
               id+=1
          pos[id] = 'EOF'
          f.seek(0)
          cursor = 0
          while cursor < list(pos.keys())[1]-1: 
               exclude.append(re.sub('\s', '', f.readline()))
               cursor+=1
          f.seek(f.tell()+2) # Skip newline
          while cursor < list(pos.keys())[2]-1: 
               include.append(re.sub('\s', '', f.readline()))
               cursor+=1  
     return exclude[1:], include[1:]

    def get_textdata(self) -> pd.DataFrame:
        print(f'\nPREPROCESSING DATA...\n(head shown)\n', file=STDOUT)
        # Compile regex patterns used for targeted cleaning
        r = re.compile(r'\[[A-Za-z0-9\.]+\]') 
        s = re.compile(r'([^\s\w]|_)+')
        t = re.compile(r'[.]{1,3}|[,]+|[\!]+|[\?]+')

        # List custom include/exclude word list 
        _custom_stopwords, _include_words = Preprocess.get_from_custom_stopwords()
        _stop_words = _custom_stopwords + list(string.printable) + stopwords.words('english')
        
        # Load data into df + tokenize/lemmatize/reduce text + get vocab size
        df = pd.DataFrame()
        df['timecode'] = self.src['timecode']
        df['speaker']=[_ for _ in self.src['part']]
        df['raw_text']=[_ for _ in self.src['dialogue']]
        df['cleaned_text'] = self.src['dialogue'] \
                                .apply(lambda x : ' '.join([WordNetLemmatizer().lemmatize(wd) 
                                       for wd in word_tokenize(re.sub(t,'',x),preserve_line=False)
                                       if wd not in _stop_words or wd in _include_words])) \
                                .apply(lambda x : ' '.join(Counter(word_tokenize(x)))) 
        df['vocabulary_size'] = df['cleaned_text'].apply(lambda x : len(word_tokenize(x)))
        print(df.loc[:,['timecode','speaker','raw_text', 'cleaned_text', 'vocabulary_size']].head())
        return df


class TopicExtraction():
    ''' params estimation: gridsearch w/ 5-fold crossvalidation
        embeddings generation: pretrained LSTM(GloVe)
        topic inference: LDA(n_topics,max_iter)
    '''
    
    def __init__(self, data:pd.DataFrame):
        #Parameters
        self.df             = data
        self.data           = data['cleaned_text']            
        self.raw_data       = data['raw_text']
        self.n_samples      = len(data['cleaned_text'])           # dataset size
        self.n_features     = len(data['cleaned_text'])           # max|features, matrix coef.| < dataset size
        self.n_top_words    = int(len(data['cleaned_text'])*0.05) # quantile of topic elements  
        self.labels:int
    
    def estimate_best_params(self) -> dict:        
        #Gridsearch for best params
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('lda', LatentDirichletAllocation()),
                ('mean', KMeans(n_init='auto'))
            ]
        )

        parameters = {
            'vect__max_features': (int(self.n_features*0.25),int(self.n_features*0.5)), # reduces analysis time... For demo purposes only!   , int(self.n_features*0.75), self.n_features),
            'vect__ngram_range': ((1,1), (1,2)),  
            'lda__n_components': ([6, 8, 10]), 
            'lda__max_iter': (int(self.n_features*0.25), int(self.n_features*0.5)) # reduces analysis time... For demo purposes only!        , int(self.n_features*0.75), self.n_features)
        }   
        
        # NB: set cv=5 for 5-folds cross-validation... but analysis time will be a lot longer...
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=2, verbose=1)

        print('\nPERFORMING GRIDSEARCH:\n', 
              f'\tpipeline: {[name for name, _ in pipeline.steps]}\n',
              f'\tevaluated params: {parameters}\n',
              file=STDOUT)
        
        print('It will take a moment...', file=STDOUT)
        grid_search.fit(self.data)
        

        # Show estimated params
        print('\nBest parameters set:', file=STDOUT)
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f'\t{param_name}: {best_parameters[param_name]}', file=STDOUT)  
        
        #return best_parameters
        return best_parameters

    def embedding(self, best_params:dict) -> list:
        # Free cached layers
        clear_session()

        # Simulate manually labeled data
        # NOTE  LSTM accuracy won't be evaluated in this demo. 
        #       Originally, this algorithm was used to complete/revise 
        #       manually labeled dataset by social sciences researchers.
        #       To avoid a lot of code editing, this line does the trick... 
        y = np.linspace(0,best_params['lda__n_components'], self.data.shape[0])

        # Preprocess and cast real data into an array 
        sentences = self.data.values
        
        # TTS 
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.3, random_state=1000)
        
        # Tokenize and Vectorize (return integer, count, multi-hot, or TF-IDF). 
        max_features = best_params['vect__max_features'] #self.n_features
        embedding_dim = 5
        sequence_length = 100
        ngram_range = best_params['vect__ngram_range'][1], 
        epochs=10
        steps_epochs=5
        maxlen = 100

        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(sentences_train)
        
        vectorizer = TextVectorization(
            max_tokens=max_features, 
            standardize='lower_and_strip_punctuation',
            split='whitespace', 
            ngrams=ngram_range,
            output_mode='int',
            output_sequence_length=sequence_length)


        # adapt() method on an array to generatea vocabulary index for the data
        vectorizer.adapt(sentences_train)
        X_train = vectorizer(sentences_train)
        X_test = vectorizer(sentences_test)
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        input_dim = X_train.shape[1]  

        #Embeddings
        def create_embedding_matrix(filepath, word_index, embedding_dim):
            print('\nCOMPUTING WORD TO TOPIC PRIOR DISTRIBUTION THROUGH EMBEDDINGS...\n ')
            embedding_matrix = np.zeros((max_features, embedding_dim))
            with open(filepath,encoding='utf8') as f:
                for line in f:
                    word, *vector = line.split()
                    if word in Counter(word_index).most_common(100000):
                        idx = word_index[word] 
                        embedding_matrix[idx] = np.array(
                            vector, dtype=np.float32)[:embedding_dim]
            return embedding_matrix

        #glove = pathlib.Path(__file__).parent.joinpath('data/glove.6B.50d.txt')
        #embedding_matrix = create_embedding_matrix(glove, tokenizer.word_index, embedding_dim)
        

        #Build model layers 
        model = Sequential([
            layers.Embedding(max_features, embedding_dim, name='embeddings'),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(1)])

        #Specify objective functions
        model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
        model.summary()

        #Fit model
        history = model.fit(X_train, y_train,
                            steps_per_epoch=steps_epochs,
                            epochs=epochs,
                            verbose=True,
                            validation_data=(X_test, y_test),
                            batch_size=5)
        
        #Evaluate model
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print('\nTraining Accuracy: {:.4f}'.format(accuracy*3))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print('Testing Accuracy:  {:.4f}'.format(accuracy*3))
        


        plot().lstm_history(history)

        weights = model.get_layer('embeddings').get_weights()[0]
        return weights

    def LDA(self, best_params):
        n_topics = best_params['lda__n_components'] #+1
        max_iter = best_params['lda__max_iter']
        ngram_range = best_params['vect__ngram_range']
        embeddings = None
        # Priors definition (word/doc index, [topic values])
        D2T_Prior=pathlib.Path(__file__).parent.joinpath('data/D2T_Prior.txt')
        prior_topic_words = []
        prior_doc_topics = []
        with open(D2T_Prior,'r+') as readfile:
            for line in readfile.readlines():
                l=re.sub(' ','',line)
                l=re.sub('\n','',line)
                x=l.split(' ')
                for elmt in x[:n_topics]:
                    i=int(x[0])
                    v=x[1]
                    v=list(v.split(','))
                    z=[float(n) for n in v[:n_topics]]
                prior_doc_topics.append([i,z])    

        if embeddings is not None:    
            for x,y in enumerate(embeddings): 
                n=[x,y] #[(0,[0.,0.,0.,0.,0.,0.])]
                prior_doc_topics.append(n)

        
        # Prepare tf bow features for LDA
        self.CountVec=CountVectorizer(analyzer='word', ngram_range=ngram_range)
        self.tfbow=self.CountVec.fit_transform(self.data)
        
        # Fit the LDA model
        print(f'\nFITTING THE LDA MODEL ON {self.n_samples} SAMPLES...\n')
        lda = LatentDirichletAllocation(n_components=n_topics,
                                        max_iter=max_iter,
                                        batch_size=150,
                                        learning_method='online',
                                        learning_offset=15,
                                        random_state=0,
                                        evaluate_every=1,
                                        n_jobs=-1
                                        )
        lda.fit(self.tfbow)
        feature_names = self.CountVec.get_feature_names_out()
        t=lda.transform(self.tfbow)
        df_t=pd.DataFrame(t)

        #eta (word_topic_prior)
        if prior_topic_words is not None:
            for ptw in prior_topic_words:
                word_index=ptw[0]
                word_topic_values=ptw[1]
                self.components_[:, word_index] *= word_topic_values
        
        #theta (doc_topic_prior)
        if prior_doc_topics is not None:
            for pdt in prior_doc_topics:
                doc_index=pdt[0]
                doc_topic_values=pdt[1]
                t[doc_index, :] *= doc_topic_values
              
        # Re-arrange distrib per topic before principal component analysis
        topic_idx = np.arange(1, n_topics+1)
        distrib_per_topic = df_t.T.to_numpy()
        dmap = dict(zip(topic_idx, distrib_per_topic))

        y_pred = []
        for c in range(distrib_per_topic.shape[1]):
            x = np.apply_over_axes(np.max, distrib_per_topic[:, c], 0),
            if x[0]>0:
                y_pred.append(
                    list(
                        (k for k, v in dmap.items() \
                         if (v[c] == x))
                    )[0]
                )
            else: y_pred.append(0) # when proba(topic) = 0, i.e. 'hidden topic' ignored 
        y_pred = np.array(y_pred)
        
        # Load data + results into a new xls table
        df_results = pd.concat([self.df, pd.DataFrame(y_pred, columns=['label_predicted'])], 
                               axis=1)
        OutputResults(
            df= df_results.loc[:, 
                ['timecode', 'speaker','raw_text', 'label_predicted']
            ]
        ).to_excel()

        print('DONE\n')

        # Evaluate model's performance

        categories, data = evaluate(n_topics=n_topics,
                                    model=lda).pca(df_t, y_pred)
      
        plot(k=n_topics,
             model=lda,
             feature_names=feature_names, 
             n_top_words=self.n_top_words, 
             title='LDA model - Topics composition').topics()
        
        plot(k=n_topics,
             model=lda,
             feature_names=feature_names,
             n_top_words=self.n_top_words, 
             title='LDA (embeddings, Log loss)').pca_3d(categories=categories, data=data)
        
        plot(k=n_topics,
             model=lda,
             feature_names=feature_names, 
             n_top_words=self.n_top_words, 
             title='Topics distributions over time').distrib_time()

        # Save figures in a png file
        OutputResults().to_png()


class evaluate():
    def __init__(self,n_topics=None,model=None,transform=None,Y_True=None,vocabulary=None):
        self.n_topics=n_topics
        self.model=model
        self.transform=transform
        self.Y_labels=Y_True
        self.vocab=vocabulary

    def pca(self, df_t, y_pred):
        embedding_matrix=df_t.to_numpy()
        pca = PCA(n_components=3,svd_solver='full',whiten=True,random_state=2345)
        vis_dims = pca.fit_transform(embedding_matrix)
        coef_1=np.array(pca.explained_variance_ratio_)
        coef_2= np.array(pca.singular_values_)
        data=pd.DataFrame()

        data[0]=[x for x in vis_dims]
        data[1]=[x for x in y_pred]

        categories = sorted(data[1].unique())

        return categories, data


class plot():
    fig = plt.figure(figsize=(20,15), dpi=300)
    subfig = fig.subfigures(1, 2, width_ratios=[1.5, 1.])
    plt.style.use('ggplot')
    plt.tight_layout()

    def __init__(self,k=None,model=None,feature_names=None,n_top_words=None,title=None):
        self.k=k
        self.model=model
        self.feature_names=feature_names
        self.n_top_words=n_top_words
        self.title=title

    def get_y_pred(self):
        topic_idx = np.arange(1, self.k+1)
        dmap = dict(zip(topic_idx, self.model.components_))
        y_pred = []

        for c in range((self.model.components_.shape[1]+1)):
            x = np.apply_over_axes(np.max, self.model.components_[:, c-1], 0),
            y_pred.append(*[k for k, v in dmap.items() if v[c-1] == x])
        y_pred = np.array(y_pred)
        return y_pred

    def topics(self):
        if self.k<3:
            x=1
        else:
            x=int(self.k/3)+1
        for topic_idx, topic in enumerate(self.model.components_):
            axes = plot.subfig[0].add_subplot(x, 3, topic_idx+1)  # index must be >0 -> topic_idx+1
            top_features_ind = topic.argsort()[:-self.n_top_words - 1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            axes.barh(top_features, weights, height=0.4,color='grey')
            axes.set_title(f'Topic {topic_idx +1}',fontsize=10)
            axes.invert_yaxis()
            axes.tick_params(axis='both', which='major',labelsize=9)
            for i in 'top right left'.split():
                axes.spines[i].set_visible(True)
        plot.subfig[0].suptitle(self.title, fontsize=14)
        plt.subplots_adjust(top=0.93, wspace=0.5, hspace=0.5)

    def distrib_time(self):
        y_pred = self.get_y_pred()
        timecode=pd.read_excel(SRC) 
        an=pd.DataFrame([timecode['timecode']]).transpose()
        an['timecode']=pd.to_datetime(an['timecode'])
        x = pd.date_range(min(an['timecode']), max(an['timecode']), len(y_pred)).values
        axes = plot.subfig[0].add_subplot(3,1,3)
        axes.scatter(x,y_pred,color='grey')
        axes.set_xlabel('Time',fontsize=10)  
        axes.set_ylabel('Topics',fontsize=10)
        axes.set_title(f'{self.title}\n',fontsize=14) 
        axes.tick_params(axis='both', which='major',labelsize=10)
        axes.set_yticks(np.arange(1, len(set(y_pred))+1, 1))
        plt.gcf().autofmt_xdate()  
        myFmt = mdates.DateFormatter('%M:%S')
        plt.gca().xaxis.set_major_formatter(myFmt)
  
    def pca_3d(self,categories=None, data=None):
        ax = plot.subfig[1].add_subplot(2,1,1, projection='3d')
        cmap = plt.get_cmap('tab10')
        for i, cat in enumerate(categories[1:]):
            sub_matrix = np.array(data[data[1] == cat][0].to_list())
            x = sub_matrix[:, 0]
            y = sub_matrix[:, 1]
            z = sub_matrix[:, 2]
            colors = [cmap(i/len(categories))] * len(sub_matrix)
            ax.scatter(xs=x, ys=y, zs=z,  c=colors, label=cat, s=200)
        ax.set_xlabel('PC_1', labelpad=5, fontsize=10)
        ax.set_ylabel('PC_2', labelpad=5, fontsize=10)
        ax.set_zlabel('PC_3', labelpad=5, fontsize=10)
        ax.tick_params(labelsize=10)
        plt.legend(bbox_to_anchor=(-0.1, 0.7, 0.1, 0.1), fontsize=10)
        ax.set_title(f'Segments to topic classification PCA(n=3)\n{self.title}\n', fontsize=14)

    def lstm_history(self, history):
        acc = [x*3 for x in history.history['binary_accuracy']]
        val_acc = [x*3 for x in history.history['val_binary_accuracy']]
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)
        
        # Accuracy not evaluated in demo mode
        ax1 = self.subfig[1].add_subplot(2,2,3)
        ax1.plot(x, np.zeros(len(acc)), 'b', label='Training acc')
        ax1.plot(x, val_acc, 'r', label='Validation acc')
        ax1.set_title('LSTM accuracy', fontsize=14)
        ax1.text(0,1, s='(not evaluated in demo mode)')
        ax1.tick_params(axis='both', labelsize=10)
        ax1.legend(fontsize=10)
        
        ax2 = self.subfig[1].add_subplot(2,2,4)
        ax2.plot(x, loss, 'b', label='Training loss')
        ax2.plot(x, val_loss, 'r', label='Validation loss')
        ax2.set_title('LSTM loss', fontsize=14)
        ax2.tick_params(axis='both', labelsize=10)
        ax2.legend(fontsize=10)


class OutputResults(object):
    def __init__(self, df:pd.DataFrame=None):
        self.df     = df 
        self.path   = pathlib.Path(__file__).parent.joinpath('data')
    
    def to_excel(self):
        with pd.ExcelWriter(self.path.joinpath('bttf_results_df.xlsx'), 
                            engine='openpyxl') as w:
            self.df.to_excel(w, sheet_name='bttf_topics')
    
    def to_png(self):
        plt.savefig(self.path.joinpath('bttf_results_figs.png'),
                    dpi='figure', format='png', metadata=None,
                    bbox_inches=None, pad_inches=0.1
        )
        

def main():  
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np.random.seed(19680801)
        
        print('\nDATASET ', SRC)
        transcript_data = pd.read_excel(SRC) 
        
        data_samples = Preprocess(transcript_data).get_textdata()

        best_params = TopicExtraction(data_samples).estimate_best_params()
        TopicExtraction(data_samples).embedding(best_params)
        TopicExtraction(data_samples).LDA(best_params)    
        
        #plt.show()


if __name__=='__main__':
    main()