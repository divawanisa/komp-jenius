import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings("ignore")
import math
from itertools import groupby
import datetime as dt
from plotly.graph_objs import *
from dash import Dash, html, dcc, dash_table, Input, Output, ctx
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
###########
jago_pros = pd.read_csv('data/jago.csv', index_col = 'Unnamed: 0')
jenius_pros = pd.read_csv('data/jenius.csv', index_col = 'Unnamed: 0')
neo_pros = pd.read_csv('data/neo.csv', index_col = 'Unnamed: 0')
tmrw_pros =pd.read_csv('data/tmrw.csv', index_col = 'Unnamed: 0')
sea_pros = pd.read_csv('data/sea.csv', index_col = 'Unnamed: 0')
blu_pros = pd.read_csv('data/blu.csv', index_col = 'Unnamed: 0')
digi_pros=pd.read_csv('data/digi.csv', index_col = 'Unnamed: 0')
        
variabl = ['jago', 'jenius', 'neo', 'tmrw', 'sea', 'blu', 'digi']
pros = []
for i in variabl:
    pros.append(globals()[f'{i}_pros'])

z_tab = pd.DataFrame(None, columns = variabl)
for i,j in zip(variabl,pros):
    z_tab[i] = j['z_score']
z_tab.index = z_tab.index.str.title()
z_tab.columns = z_tab.columns.str.title()  

def modelling_pca(variabel, nilai):
    global yahi, yaho, pca_awal, buat_pca, for_pca
    if nilai == 'jaccard':
        # Membuat tabel nilai
        buat_pca = pd.DataFrame(None, columns = variabl)    
        for i,j in zip(variabl,pros):
            buat_pca[i] = j[nilai]
        buat_pca = buat_pca[variabel]

        #Membuat ranking
        for_pca = buat_pca.copy()
        for i in variabel:
            for_pca[i] = buat_pca[i].rank(ascending=False)
        yaho = for_pca.T

        #Scaling Data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(yaho)
        yahi = scaler.transform(yaho)

        from sklearn.decomposition import PCA
        pca_awal = PCA(n_components=len(variabel))
        pca_samples = pca_awal.fit_transform(yahi)

        PC_values = np.arange(pca_awal.n_components_) + 1
        plt.figure(figsize=(5,3))
        plot = plt.plot(PC_values, pca_awal.explained_variance_, 'o-', linewidth=2, color='green')
        plt.title('SCREE PLOT')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalues')

    elif nilai == 'z_score':
        # Membuat tabel nilai
        buat_pca = pd.DataFrame(None, columns = variabl) 
        for i,j in zip(variabl,pros):
            buat_pca[i] = j[nilai]
        buat_pca = buat_pca[variabel]

        yaho = buat_pca.T

        #Scaling Data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(yaho)
        yahi = scaler.transform(yaho)

        from sklearn.decomposition import PCA
        pca_awal = PCA(n_components=len(variabel))
        pca_samples = pca_awal.fit_transform(yahi)

        PC_values = np.arange(pca_awal.n_components_) + 1
        plt.figure(figsize=(5,3))
        plot = plt.plot(PC_values, pca_awal.explained_variance_, 'o-', linewidth=2, color='green')
        plt.title('SCREE PLOT')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalues')

def hasil_pca(n):
    global hasil, group, pca, loading_matrix
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(yahi)
    pca_samples = pca.transform(yahi)
    
    #Factor loading
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    lizt = []
    for i in range(1,n+1):
        namez = 'PC ' + str(i)
        lizt.append(namez)
    loading_matrix = pd.DataFrame(loadings, columns=lizt, index=buat_pca.index)

    #Element loading
    hasil = pd.DataFrame(pca_samples, columns=lizt, index=buat_pca.columns)

    grouping = loading_matrix.T[abs(loading_matrix.T) == abs(loading_matrix.T).max()[0:30]]
    listy = []
    for i in grouping.columns:
        cihuy = grouping.loc[~grouping[i].isna()].index
        listy.append(cihuy)
    group = pd.DataFrame(listy, index = grouping.columns, columns = ['pc']).sort_values(by='pc')    

def define_var_pca(namvar, kurang = 1):
    globals()[f'hasil_pca_{namvar}'] = hasil.copy()
    globals()[f'group_{namvar}'] = group.copy()
    globals()[f'loading_matrix_{namvar}'] = (loading_matrix*kurang).copy()
    globals()[f'asli_{namvar}'] = yaho.T.copy()
    globals()[f'scaled_{namvar}'] = pd.DataFrame(yahi.T, index = yaho.columns, columns = yaho.index)
    globals()[f'pca_awal_{namvar}'] = pca_awal
    globals()[f'pca_{namvar}'] = pca
    globals()[f'buat_pca_{namvar}'] = buat_pca.copy()

def plotting(fig):
    fig.update_traces(textposition="bottom right")
    fig.update_layout(paper_bgcolor="white",plot_bgcolor='white')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='Gray',gridwidth=0.01, gridcolor='LightGray', griddash = 'dash')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='Gray',gridwidth=0.01, gridcolor='LightGray', griddash = 'dash')
    fig.add_hline(y=0, line_width = 0.75)
    fig.add_vline(x=0, line_width = 0.75)

def atribut_dimensi(huruf,thedata, dimensi,namvar):
    for i in range(1,dimensi + 1):
        globals()[f'{huruf}_pc{i}'] = thedata[thedata.pc == 'PC {}'.format(str(i))].\
        apply(lambda x: globals()[f'loading_matrix_{namvar}'].loc[x.index]['PC {}'.format(str(i))]).\
        rename(columns = {'pc':'PC {}'.format(str(i))}).\
        sort_values(by='PC {}'.format(str(i)), key= abs, ascending=False)
        globals()[f'{huruf}_pc{i}'].index = globals()[f'{huruf}_pc{i}'].index.str.title()
        globals()[f'{huruf}_pc{i}_pos'] = round(globals()[f'{huruf}_pc{i}'][globals()[f'{huruf}_pc{i}']['PC {}'.format(str(i))]>=0].rename_axis('Atribut PC {}'.format(str(i))).reset_index().rename(columns = {'PC {}'.format(str(i)): 'Korelasi'}),3)
        globals()[f'{huruf}_pc{i}_neg'] = round(globals()[f'{huruf}_pc{i}'][globals()[f'{huruf}_pc{i}']['PC {}'.format(str(i))]<0].rename_axis('Atribut PC {}'.format(str(i))).reset_index().rename(columns = {'PC {}'.format(str(i)): 'Korelasi'}),3)

def competition(data, brand, dimensi):
    global perkompetisian, slat
    comp = pd.read_csv('data/untuk kompetitor 2.csv', delimiter = ';')
    comp = comp[~comp['var_1'].isin(['line','motion'])]
    comp = comp[~comp['var_2'].isin(['line','motion'])].reset_index(drop = True)
    n= brand           # Number of brands
    k= dimensi         # Number of dimension
    comp['slater'] = [None]*len(comp)

    for var1, var2, i in zip(comp['var_1'],comp['var_2'], range(0,len(comp))):
        penjumlah_akhir = 0
        for isi in range(1,k+1):
            var_penjumlah = ((data.loc[var1]['PC '+str(isi)]-data.loc[var2]['PC '+str(isi)])**2)
            penjumlah_akhir = penjumlah_akhir+var_penjumlah
        comp['slater'][i] = (np.sqrt(penjumlah_akhir))/np.sqrt((2*n*k)/(n-1))

    slat = pd.DataFrame(comp.pivot_table(comp, index = 'var_1', columns ='var_2'))
    slat.columns = slat.columns.get_level_values(1)
    slat['mean'] = [None]*len(slat)
    for i in slat.index:
        slat['mean'][i] = np.sum(slat[i])/(len(slat.columns) - 2)
    
    kompetitor = slat.copy()
    for i in kompetitor.drop(['mean'], axis = 1).columns:
        kompetitor[i] = np.where(kompetitor[i] < kompetitor['mean'], 'Ya', '-')
    
    yoha = kompetitor.drop(['mean'], axis = 1).copy()
    for i in kompetitor.drop(['mean'], axis = 1).columns:
        yoha[i] = np.where(slat[i].isin(slat[i].sort_values(ascending = False).tail(3)), 'yas', '-')
    for i in yoha.index:
        yoha.loc[i][i] = '-'
    perkompetisian = yoha.T

modelling_pca(variabl, 'jaccard')
hasil_pca(3)
define_var_pca('j')
atribut_dimensi('group_j',group_j,3,'j')

fig = px.scatter(hasil_pca_j, x="PC 1", y="PC 2", text=hasil_pca_j.index.str.title(), title = 'PETA PC 1 VS PC 2',hover_name = hasil_pca_j.index.str.title())
plotting(fig)

#kompetisi
global kompetisi
competition(hasil_pca_j, len(variabl), 3)
kompetisi = pd.DataFrame(None, columns = ['merek_1','merek_2','merek_3'])
kompetisi['merek_1'] = perkompetisian.columns
for i in range(0,len(kompetisi)):
    kompetisi['merek_2'][i] = perkompetisian.loc[kompetisi.merek_1.loc[i]][perkompetisian.loc[kompetisi.merek_1.loc[i]] == 'yas'].index[0]
    kompetisi['merek_3'][i] = perkompetisian.loc[kompetisi.merek_1.loc[i]][perkompetisian.loc[kompetisi.merek_1.loc[i]] == 'yas'].index[1]
kompetisi = kompetisi.T
kompetisi.columns = range(1,len(variabl)+1)

for i in range(1,len(kompetisi.columns)+1):
    modelling_pca(kompetisi[i],'z_score')
    hasil_pca(2)
    define_var_pca(str(i),0.8)
    atribut_dimensi('group_{}'.format(str(i)),globals()[f'group_{i}'],2,str(i))

j = kompetisi.loc['merek_1'][kompetisi.loc['merek_1'] == 'Jenius'.lower()].index[0]
fig2 = px.scatter(globals()[f'hasil_pca_{j}'], x="PC 1", y="PC 2", text=globals()[f'hasil_pca_{j}'].index.str.title(), title = 'PETA',hover_name = globals()[f'hasil_pca_{j}'].index.str.title())  
plotting(fig2)

app = Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

#TABBBB 2
tab2 =  html.Div([  
    html.Div([ #tabel atas
            dash_table.DataTable(data=group_4_pc2_pos.to_dict('records'),id = 'tabel2-atas', 
            fixed_rows={'headers': True}, style_table={'height': 150}, 
            style_cell={'textAlign': 'center','font-family':'Helvetica','font-size':'75%'}, 
            style_header = {'background-color':'#cfcfcf', 'font-weight':'bold'})], 
            style = {'height':'50%', 'width': '20%', 'margin':'0px auto'}),

    html.Div([ #tabel kiri
        dash_table.DataTable(data=group_4_pc1_neg.to_dict('records'),id = 'tabel2-kiri', 
        fixed_rows={'headers': True}, style_table={'height': 150}, 
        style_cell={'textAlign': 'center','font-family':'Helvetica','font-size':'75%'}, 
        style_header = {'background-color':'#cfcfcf', 'font-weight':'bold'})], 
        style = {'height':'50%', 'width':'20%', 'display':'inline-block', 'float':'left', 'margin-top':'150px'}),

    html.Div([dcc.Graph(id ='second-page-graph',figure = fig2, 
    clickData = {'points':[{'text':'Sea'}]})],style = {'width':'50%', 'float':'left', 'display':'inline-block','margin-left':'100px', 'margin-top':'25px'}), 
    
    html.Div([ #tabel kanan
            dash_table.DataTable(data=group_4_pc1_pos.to_dict('records'),id = 'tabel2-kanan', 
            fixed_rows={'headers': True}, style_table={'height': 150}, 
            style_cell={'textAlign': 'center','font-family':'Helvetica','font-size':'75%'}, 
            style_header = {'background-color':'#cfcfcf', 'font-weight':'bold'})], 
            style = {'height':'50%', 'width':'20%', 'float':'right', 'display':'inline-block', 'margin-top':'150px'}),

    html.Div([ #tabel bawah
            dash_table.DataTable(data=group_4_pc2_neg.to_dict('records'),id = 'tabel2-bawah', 
            fixed_rows={'headers': True}, style_table={'height': 150}, 
            style_cell={'textAlign': 'center','font-family':'Helvetica','font-size':'75%'}, 
            style_header = {'background-color':'#cfcfcf', 'font-weight':'bold'})], 
            style = {'height':'50%', 'margin':'0px auto', 'width': '20%', 'margin':'0px auto'})

        ], style = {'margin-top': '5px','background-color':'white', 'float':'left', 'width':'100%'})

# LAYOUT
app.layout = html.Div(style = {'background-color':'white'},children = [
html.Div([    
    html.H3(['DASHBOARD PETA PERSEPTUAL'], style ={'font-family':'Helvetica', 'text-align':'left','height':'50%', 'text-align':'center'}), # Main page
]), style= {'display':'inline-block', 'width':'100%'}),
html.Div(id ='tab-pertama', children = tab1, style= {'padding':'0 0 0 0'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
