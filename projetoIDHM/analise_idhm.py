import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import re
from matplotlib.patches import Patch

COLOR_PALETTE = ['#6AF307', '#38A6B7', '#15DDCE', '#0AE885', '#15D84A', '#0AEAFC']
sns.set_palette(COLOR_PALETTE)

FIG_SIZE = (10, 6)

def carregar_dados():
    arquivo = "./dados/idhm_bd.xlsx"
    abas = pd.ExcelFile(arquivo).sheet_names
    
    COLUNAS = ["ANO", "AGREGACAO", "CODIGO", "NOME", "IDHM", "IDHM_L", "IDHM_E", "IDHM_R", "IDHMAD", "IDHMAD_L", "ESPVIDA"]
    df_idhm = pd.read_excel(arquivo, sheet_name='Base de Dados', usecols=COLUNAS)
    df_cor = pd.read_excel(arquivo, sheet_name='COR')
    df_sexo = pd.read_excel(arquivo, sheet_name='SEXO')
    
    return df_idhm, df_cor, df_sexo

def plot_evolucao_idhm_negra(df_cor):
    df_negra = df_cor[(df_cor['AGREGACAO'] == 'BRASIL') & (df_cor['COR'] == 'NEGRO')]
    
    plt.figure(figsize=FIG_SIZE)
    sns.lineplot(data=df_negra, x='ANO', y='IDHM', marker='o', color=COLOR_PALETTE[1])
    plt.title('Evolução do IDHM da População Negra no Brasil (2012–2021)')
    plt.ylabel('IDHM')
    plt.xlabel('Ano')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_renda_raca_2021(df_cor):
    df_2021 = df_cor[(df_cor['AGREGACAO'] == 'BRASIL') & (df_cor['ANO'] == 2021)]
    df_branca_negra = df_2021[df_2021['COR'].isin(['BRANCO', 'NEGRO'])]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_branca_negra, x='COR', y='IDHM_R', 
                palette=[COLOR_PALETTE[2], COLOR_PALETTE[0]], hue='COR')
    plt.title('IDHM de Renda por Raça/Cor - Brasil (2021)')
    plt.ylabel('IDHM de Renda')
    plt.xlabel('Raça/Cor')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_longevidade_raca(df_cor):
    df_long = df_cor[(df_cor['AGREGACAO'] == 'BRASIL') &
                    (df_cor['ANO'].isin([2012, 2021])) &
                    (df_cor['COR'].isin(['BRANCO', 'NEGRO']))]
    
    plt.figure(figsize=FIG_SIZE)
    sns.barplot(data=df_long, x='ANO', y='IDHM_L', hue='COR', 
                palette=[COLOR_PALETTE[2], COLOR_PALETTE[0]])
    plt.title('IDHM de Longevidade por Raça/Cor - Brasil (2012 vs 2021)')
    plt.ylabel('IDHM de Longevidade')
    plt.xlabel('Ano')
    plt.legend(title='Raça/Cor')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_renda_pandemia(df_cor):
    df_pandemia = df_cor[(df_cor['AGREGACAO'] == 'BRASIL') &
                        (df_cor['COR'] == 'NEGRO') &
                        (df_cor['ANO'].isin([2018, 2019, 2020, 2021]))]
    
    df_pandemia.loc[:, 'ANO'] = df_pandemia['ANO'].astype(int)
    df_pandemia = df_pandemia.sort_values('ANO')
    
    plt.figure(figsize=FIG_SIZE)
    sns.lineplot(data=df_pandemia, x='ANO', y='IDHM_R', marker='o', color=COLOR_PALETTE[3])
    
    for i, row in df_pandemia.iterrows():
        plt.text(row['ANO'], row['IDHM_R'] + 0.001, f"{row['IDHM_R']:.3f}", ha='center')
    
    plt.ylim(0.650, 0.700)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.title('IDHM de Renda da População Negra (Pré e Pós Pandemia)')
    plt.ylabel('IDHM de Renda')
    plt.xlabel('Ano')
    plt.xticks(df_pandemia['ANO'].unique())
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_renda_sexo_2021(df_sexo):
    df_2021 = df_sexo[(df_sexo['ANO'] == 2021) & (df_sexo['AGREGACAO'] == 'BRASIL')]
    
    plt.figure(figsize=FIG_SIZE)
    sns.barplot(data=df_2021, x='SEXO', y='IDHM_R', hue='SEXO', 
                palette=COLOR_PALETTE[0:2], dodge=False, legend=False)
    
    plt.ylim(0.000, 0.800)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.title('IDHM de Renda por Sexo - Brasil (2021)')
    plt.xlabel('Sexo')
    plt.ylabel('IDHM de Renda')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_longevidade_sexo(df_sexo):
    df_longev = df_sexo[(df_sexo['AGREGACAO'] == 'BRASIL') & (df_sexo['SEXO'].isin(['HOMEM', 'MULHER']))]
    
    plt.figure(figsize=FIG_SIZE)
    sns.lineplot(data=df_longev, x='ANO', y='IDHM_L', hue='SEXO', 
                 marker='o', palette=COLOR_PALETTE[0:2])
    
    plt.title('IDHM de Longevidade por Sexo (2012–2021)')
    plt.ylabel('IDHM de Longevidade')
    plt.xlabel('Ano')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_diferenca_longevidade_sexo(df_sexo):
    df_longev = df_sexo[(df_sexo['AGREGACAO'] == 'BRASIL') & (df_sexo['SEXO'].isin(['HOMEM', 'MULHER']))]
    df_diff_longevidade = df_longev.pivot(index='ANO', columns='SEXO', values='IDHM_L')
    df_diff_longevidade['diferença'] = df_diff_longevidade['MULHER'] - df_diff_longevidade['HOMEM']
    
    plt.figure(figsize=FIG_SIZE)
    sns.lineplot(data=df_diff_longevidade, x=df_diff_longevidade.index, 
                 y='diferença', marker='o', color=COLOR_PALETTE[4])
    
    plt.title('Diferença no IDHM de Longevidade (Mulheres - Homens)')
    plt.ylabel('Diferença')
    plt.xlabel('Ano')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_disparidade_educacao_sexo(df_sexo):
    df_edu = df_sexo[(df_sexo['AGREGACAO'] == 'BRASIL') & (df_sexo['SEXO'].isin(['HOMEM', 'MULHER']))]
    df_diff_edu = df_edu.pivot(index='ANO', columns='SEXO', values='IDHM_E')
    df_diff_edu['diferença (%)'] = (df_diff_edu['MULHER'] - df_diff_edu['HOMEM'])
    
    plt.figure(figsize=FIG_SIZE)
    sns.lineplot(data=df_diff_edu, x=df_diff_edu.index, 
                 y='diferença (%)', marker='o', color=COLOR_PALETTE[5])
    
    plt.ylim(0.040, 0.060)
    plt.yticks([round(x, 3) for x in np.arange(0.040, 0.061, 0.005)])
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    for ano, valor in df_diff_edu['diferença (%)'].items():
        plt.text(ano, valor + 0.0005, f'{valor:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.title('Disparidade (%) no IDHM de Educação (Mulheres - Homens)')
    plt.ylabel('Diferença')
    plt.xlabel('Ano')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_mulheres_regioes(df_sexo, df_idhm):
    # Define regions
    regioes = {
        'Norte': ['Acre', 'Amapá', 'Amazonas', 'Pará', 'Rondônia', 'Roraima', 'Tocantins'],
        'Sul': ['Paraná', 'Rio Grande do Sul', 'Santa Catarina']
    }
    
    df_mulheres_2021 = df_sexo[
        (df_sexo['SEXO'] == 'MULHER') &
        (df_sexo['ANO'] == 2021) &
        (df_sexo['AGREGACAO'] == 'UF')
    ]
    
    idhm_sul = df_mulheres_2021[df_mulheres_2021['NOME'].isin(regioes['Sul'])]['IDHM'].mean()
    idhm_norte = df_mulheres_2021[df_mulheres_2021['NOME'].isin(regioes['Norte'])]['IDHM'].mean()
    
    df_regioes = pd.DataFrame({
        'Região': ['Sul', 'Norte'],
        'IDHM Médio (Mulheres)': [idhm_sul, idhm_norte]
    })
    
    plt.figure(figsize=FIG_SIZE)
    sns.barplot(
        data=df_regioes,
        x='Região',
        y='IDHM Médio (Mulheres)',
        hue='Região',
        palette=COLOR_PALETTE[0:2],
        legend=False
    )
    plt.title('IDHM Médio de Mulheres - Sul vs Norte (2021)')
    plt.ylabel('IDHM')
    plt.xlabel('Região')
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_idhm_regioes_agrupadas(df_idhm):
    regioes = {
        'Norte': ['Acre', 'Amazonas', 'Amapá', 'Pará', 'Rondônia', 'Roraima', 'Tocantins'],
        'Nordeste': ['Alagoas', 'Bahia', 'Ceará', 'Maranhão', 'Paraíba', 'Pernambuco', 'Piauí', 'Rio Grande do Norte', 'Sergipe'],
        'Centro-Oeste': ['Distrito Federal', 'Goiás', 'Mato Grosso', 'Mato Grosso do Sul'],
        'Sudeste': ['Espírito Santo', 'Minas Gerais', 'Rio de Janeiro', 'São Paulo'],
        'Sul': ['Paraná', 'Rio Grande do Sul', 'Santa Catarina']
    }
    
    df_idhm['REGIAO'] = df_idhm['NOME'].map({uf: reg for reg, ufs in regioes.items() for uf in ufs})
    df_regioes_2021 = df_idhm[(df_idhm['AGREGACAO'] == 'UF') & (df_idhm['ANO'] == 2021)]
    
    media_regioes = df_regioes_2021.groupby('REGIAO')['IDHM'].mean().reset_index()
    media_regioes['Agrupamento'] = media_regioes['REGIAO'].apply(
        lambda x: 'Norte/Nordeste' if x in ['Norte', 'Nordeste'] else 'Centro-Sul')
    
    media_agrupada = media_regioes.groupby('Agrupamento')['IDHM'].mean().reset_index()
    
    contagem_ufs = {
        'Norte/Nordeste': len(regioes['Norte']) + len(regioes['Nordeste']),
        'Centro-Sul': len(regioes['Centro-Oeste']) + len(regioes['Sudeste']) + len(regioes['Sul'])
    }
    
    plt.figure(figsize=FIG_SIZE)
    ax = sns.barplot(data=media_agrupada, x='Agrupamento', y='IDHM', 
                     palette=COLOR_PALETTE[3:5], hue='Agrupamento')
    
    for i, row in media_agrupada.iterrows():
        ax.text(i, row['IDHM'] + 0.009, f"{row['IDHM']:.3f}",
                ha='center', va='top', color='black', fontweight='bold')
    
    info_text = "\n".join([
        f"Centro-Sul: {contagem_ufs['Centro-Sul']} UFs",
        f"Norte/Nordeste: {contagem_ufs['Norte/Nordeste']} UFs"
    ])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='gray')
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props)
    
    plt.title('IDHM Médio por Agrupamento Regional (2021)')
    plt.ylabel('IDHM')
    plt.ylim(0.65, 0.80)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_comparacao_regioes(df_idhm):
    regioes = {
        'Nordeste': ['Alagoas', 'Bahia', 'Ceará', 'Maranhão', 'Paraíba', 'Pernambuco', 'Piauí', 'Rio Grande do Norte', 'Sergipe'],
        'Centro-Oeste': ['Distrito Federal', 'Goiás', 'Mato Grosso', 'Mato Grosso do Sul']
    }
    
    df_idhm['REGIAO'] = df_idhm['NOME'].map({uf: reg for reg, ufs in regioes.items() for uf in ufs})
    df_regioes_2021 = df_idhm[(df_idhm['AGREGACAO'] == 'UF') & (df_idhm['ANO'] == 2021)]
    
    media_regioes = df_regioes_2021[df_regioes_2021['REGIAO'].isin(['Nordeste', 'Centro-Oeste'])]
    media_regioes = media_regioes.groupby('REGIAO')['IDHM'].mean().reset_index()
    
    contagem_ufs = {
        'Nordeste': len(regioes['Nordeste']),
        'Centro-Oeste': len(regioes['Centro-Oeste'])
    }
    
    plt.figure(figsize=FIG_SIZE)
    ax = sns.barplot(data=media_regioes, x='REGIAO', y='IDHM', 
                     palette=COLOR_PALETTE[4:6], hue='REGIAO',
                     order=['Nordeste', 'Centro-Oeste'],
                     estimator=np.mean, errorbar=None)
    
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2.,
                height + 0.005,
                f"{height:.3f}",
                ha='center', va='bottom',
                color='black',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_ylim(0.65, 0.80)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.title('Comparação Precisa do IDHM Médio (2021)')
    plt.xlabel('Região')
    plt.ylabel('IDHM')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_comparacao_pe_to(df_idhm):
    df_regioes_2021 = df_idhm[(df_idhm['AGREGACAO'] == 'UF') & (df_idhm['ANO'] == 2021)]
    df_pe_to = df_regioes_2021[df_regioes_2021['NOME'].isin(['Pernambuco', 'Tocantins'])][['NOME', 'IDHM', 'IDHM_E', 'IDHM_R', 'IDHM_L']].copy()
    
    df_pe_to_melt = df_pe_to.melt(id_vars='NOME',
                                var_name='Indicador',
                                value_name='Valor')
    
    ordem_indicadores = ['IDHM', 'IDHM_E', 'IDHM_L', 'IDHM_R']
    df_pe_to_melt['Indicador'] = pd.Categorical(df_pe_to_melt['Indicador'],
                                              categories=ordem_indicadores,
                                              ordered=True)
    
    plt.figure(figsize=FIG_SIZE)
    ax = sns.barplot(data=df_pe_to_melt,
                    x='Indicador',
                    y='Valor',
                    hue='NOME',
                    palette=COLOR_PALETTE[0:2],
                    saturation=0.8)
    
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width()/2,
                height + 0.003,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                color='black',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
            )
    
    leg = ax.legend(title='Estado',
                   loc='upper right',
                   bbox_to_anchor=(0.98, 0.98),
                   frameon=True,
                   framealpha=0.9,
                   edgecolor='gray')
    leg.get_frame().set_linewidth(0.5)
    
    ax.set_xticklabels(['IDHM Geral', 'Educação', 'Longevidade', 'Renda'])
    plt.title('Comparação dos Subíndices do IDHM: PE vs TO (2021)', pad=15, fontsize=12)
    plt.ylabel('Valor do Índice', labelpad=8, fontsize=10)
    plt.xlabel('')
    plt.ylim(0.60, 0.85)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_comparacao_sp_to(df_idhm):
    df_cidades = df_idhm[
        (df_idhm['ANO'] == 2021) &
        (df_idhm['AGREGACAO'] == 'RM_RIDE') &
        (df_idhm['NOME'].isin(['Região Metropolitana de São Paulo (SP)', 'Região Metropolitana de Tocantins (TO)']))
    ][['NOME', 'IDHM', 'IDHM_E', 'IDHM_R', 'IDHM_L']].copy()
    
    df_cidades_melt = df_cidades.melt(
        id_vars='NOME',
        var_name='Indicador',
        value_name='Valor'
    )
    
    ordem_indicadores = ['IDHM', 'IDHM_E', 'IDHM_L', 'IDHM_R']
    df_cidades_melt['Indicador'] = pd.Categorical(
        df_cidades_melt['Indicador'],
        categories=ordem_indicadores,
        ordered=True
    )
    
    plt.figure(figsize=FIG_SIZE)
    ax = sns.barplot(
        data=df_cidades_melt,
        x='Indicador',
        y='Valor',
        hue='NOME',
        palette=COLOR_PALETTE[1:3],
        saturation=0.8
    )
    
    for patch in ax.patches:
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.003,
            f"{patch.get_height():.3f}",
            ha='center',
            va='bottom',
            color='black',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
        )
    
    leg = ax.legend(
        title='Cidade',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )
    leg.get_frame().set_linewidth(0.5)
    
    ax.set_xticklabels(['IDHM Geral', 'Educação', 'Longevidade', 'Renda'])
    plt.title('Comparação dos Subíndices do IDHM: São Paulo vs Palmas (2021)', pad=15, fontsize=12)
    plt.ylabel('Valor do Índice', labelpad=8, fontsize=10)
    plt.xlabel('')
    plt.ylim(0.60, 0.90)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_capitais_vs_estados(df_idhm):
    sigla_para_estado = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
        'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal',
        'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão',
        'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
        'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco',
        'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
        'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima',
        'SC': 'Santa Catarina', 'SP': 'São Paulo', 'SE': 'Sergipe',
        'TO': 'Tocantins'
    }
    
    df_capitais = df_idhm[
        (df_idhm['AGREGACAO'] == 'RM_RIDE') &
        (df_idhm['ANO'] == 2021) &
        (df_idhm['NOME'].str.contains('Região Metropolitana'))
    ].copy()
    
    df_capitais['UF_SIGLA'] = df_capitais['NOME'].str.extract(r'\((\w{2})\)$')
    df_capitais['UF_NOME'] = df_capitais['UF_SIGLA'].map(sigla_para_estado)
    
    df_estados = df_idhm[
        (df_idhm['AGREGACAO'] == 'UF') &
        (df_idhm['ANO'] == 2021) &
        (df_idhm['NOME'].isin(df_capitais['UF_NOME'].unique()))
    ][['NOME', 'IDHM']].rename(columns={'NOME': 'UF_NOME', 'IDHM': 'IDHM_ESTADO'})
    
    df_comparacao = df_capitais.merge(df_estados, on='UF_NOME')
    df_comparacao['DIFERENCA'] = df_comparacao['IDHM'] - df_comparacao['IDHM_ESTADO']
    
    top_capitais = df_comparacao.sort_values(by='DIFERENCA', key=abs, ascending=False).head(7)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    paleta = {'IDHM': COLOR_PALETTE[0], 'IDHM_ESTADO': COLOR_PALETTE[3]}
    
    ax = sns.barplot(
        data=df_comparacao.melt(
            id_vars=['NOME', 'UF_NOME', 'DIFERENCA'],
            value_vars=['IDHM', 'IDHM_ESTADO'],
            var_name='TIPO',
            value_name='VALOR_IDHM'
        ),
        x='NOME',
        y='VALOR_IDHM',
        hue='TIPO',
        palette=paleta,
        order=top_capitais.sort_values('DIFERENCA', ascending=False)['NOME']
    )
    
    for i, (_, row) in enumerate(top_capitais.iterrows()):
        ax.text(i-0.2, row['IDHM']+0.005, f"{row['IDHM']:.3f}",
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)
        ax.text(i+0.2, row['IDHM_ESTADO']+0.005, f"{row['IDHM_ESTADO']:.3f}",
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)
    
    ax.set_xticklabels([
        f"{extrair_nome_capital(n)}\n({uf})" for n, uf in zip(top_capitais['NOME'], top_capitais['UF_NOME'])
    ], fontsize=10, rotation=0, ha='center')
    
    plt.ylabel('IDHM', fontsize=11, labelpad=10)
    plt.xlabel('Capital', fontsize=11, labelpad=10)
    plt.ylim(0.58, 0.87)
    
    plt.legend(
        handles=[
            Patch(facecolor=COLOR_PALETTE[0], label='Capital'),
            Patch(facecolor=COLOR_PALETTE[3], label='Estado Médio')
        ],
        title='Comparação',
        bbox_to_anchor=(1.15, 1),
        loc='upper right',
        frameon=True,
        framealpha=0.9
    )
    
    plt.title('Top 7 Capitais com Maior Diferença entre IDHM da Capital e do Estado (2021)',
             pad=20, fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def extrair_nome_capital(nome_completo):
    match = re.search(r'Região Metropolitana de (.+?) \(', nome_completo)
    if match:
        return match.group(1)
    return nome_completo

def plot_evolucao_sul_nordeste(df_idhm):
    regioes = {
        'Sul': ['Paraná', 'Rio Grande do Sul', 'Santa Catarina'],
        'Nordeste': ['Alagoas', 'Bahia', 'Ceará', 'Maranhão', 'Paraíba', 'Pernambuco', 'Piauí', 'Rio Grande do Norte', 'Sergipe']
    }
    
    estado_para_regiao = {
        estado: regiao for regiao, estados in regioes.items() for estado in estados
    }
    
    estados_selecionados = regioes['Sul'] + regioes['Nordeste']
    df_evol = df_idhm[
        (df_idhm['AGREGACAO'] == 'UF') &
        (df_idhm['NOME'].isin(estados_selecionados))
    ].copy()
    
    df_evol['REGIAO'] = df_evol['NOME'].map(estado_para_regiao)
    df_evol['IDHM'] = pd.to_numeric(df_evol['IDHM'], errors='coerce')
    media_ano_regiao = df_evol.groupby(['ANO', 'REGIAO'])['IDHM'].mean().unstack()
    
    plt.figure(figsize=FIG_SIZE)
    sns.set_style("white")
    
    plt.fill_between(
        media_ano_regiao.index,
        media_ano_regiao['Sul'],
        media_ano_regiao['Nordeste'],
        where=(media_ano_regiao['Sul'] > media_ano_regiao['Nordeste']),
        interpolate=True,
        color=COLOR_PALETTE[4],
        alpha=0.3,
        label='Disparidade (Sul - Nordeste)'
    )
    
    sns.lineplot(x=media_ano_regiao.index, y=media_ano_regiao['Sul'], 
                 label='Sul', color=COLOR_PALETTE[0], linewidth=2)
    sns.lineplot(x=media_ano_regiao.index, y=media_ano_regiao['Nordeste'], 
                 label='Nordeste', color=COLOR_PALETTE[1], linewidth=2)
    
    for ano in media_ano_regiao.index:
        y_sul = media_ano_regiao.loc[ano, 'Sul']
        y_nordeste = media_ano_regiao.loc[ano, 'Nordeste']
        dif = y_sul - y_nordeste
        y_meio = (y_sul + y_nordeste) / 2
        plt.text(ano, y_meio, f"{dif:.3f}", ha='center', va='center', 
                fontsize=9, color='darkgreen')
    
    plt.title('Evolução do IDHM: Sul x Nordeste com Disparidade Destacada (2012–2021)', fontsize=14)
    plt.xlabel('Ano', fontsize=11)
    plt.ylabel('IDHM', fontsize=11)
    plt.xticks(media_ano_regiao.index)
    plt.ylim(media_ano_regiao['Nordeste'].min() - 0.01, media_ano_regiao['Sul'].max() + 0.01)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def main():
    st.set_page_config(layout="wide", page_title="Análise IDHM")
    
    st.title("Análise do Índice de Desenvolvimento Humano Municipal (IDHM)")
    st.markdown("""
    A base de dados escolhida foi IDHM (Índice de Desenvolvimento Humano Municipal). O IDHM do Programa das Nações Unidas para o Desenvolvimento (UNDP) é uma adaptação do Índice de Desenvolvimento Humano (IDH) para medir o nível de desenvolvimento das cidades brasileiras. 
                Esta aplicação apresenta uma análise detalhada do IDHM (Índice de Desenvolvimento Humano Municipal) 
    com foco em disparidades regionais, de gênero e raça/cor.
    """)
    
    # Load data
    df_idhm, df_cor, df_sexo = carregar_dados()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise por Raça/Cor", "Análise por Gênero", "Análise Regional"])
    
    with tab1:
        st.header("Visão Geral dos Dados")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Dados IDHM")
            st.dataframe(df_idhm.head())
        
        with col2:
            st.subheader("Dados por Raça/Cor")
            st.dataframe(df_cor.head())
        
        with col3:
            st.subheader("Dados por Sexo")
            st.dataframe(df_sexo.head())
        
        st.subheader("Evolução do IDHM por Região")
        plot_evolucao_sul_nordeste(df_idhm)
        
        st.subheader("Comparação entre Capitais e Estados")
        plot_capitais_vs_estados(df_idhm)
    
    with tab2:
        st.header("Análise por Raça/Cor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Evolução IDHM da população negra do Brasil durante o intervalo dos anos de 2012 até 2021")
            plot_evolucao_idhm_negra(df_cor)
            
            st.subheader("Comparação do IDHM de Longevidade entre a população negra e a população branca do Brasil")
            plot_idhm_longevidade_raca(df_cor)
        
        with col2:
            st.subheader("Comparação do IDHM de renda entre a população negra e a população branca do Brasil (2021)")
            plot_idhm_renda_raca_2021(df_cor)
            
            st.subheader("Observação do IDHM de Renda da População Negra (Pré e Pós Pandemia)")
            plot_idhm_renda_pandemia(df_cor)
    
    with tab3:
        st.header("Análise por Gênero")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("IDHM de Renda por Sexo (2021)")
            plot_idhm_renda_sexo_2021(df_sexo)
            
            st.subheader("IDHM de Longevidade por Sexo")
            plot_idhm_longevidade_sexo(df_sexo)
        
        with col2:
            st.subheader("Diferença no IDHM de Longevidade")
            plot_diferenca_longevidade_sexo(df_sexo)
            
            st.subheader("Disparidade no IDHM de Educação")
            plot_disparidade_educacao_sexo(df_sexo)
        
        st.subheader("IDHM Médio de Mulheres - Sul vs Norte (2021)")
        plot_idhm_mulheres_regioes(df_sexo, df_idhm)
    
    with tab4:
        st.header("Análise Regional")
        
        st.subheader("IDHM Médio por Agrupamento Regional (2021)")
        plot_idhm_regioes_agrupadas(df_idhm)
        
        st.subheader("Comparação Precisa do IDHM Médio (2021)")
        plot_comparacao_regioes(df_idhm)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Comparação PE vs TO (2021)")
            plot_comparacao_pe_to(df_idhm)
        
        with col2:
            st.subheader("Comparação SP vs TO (2021)")
            plot_comparacao_sp_to(df_idhm)

if __name__ == "__main__":
    main()