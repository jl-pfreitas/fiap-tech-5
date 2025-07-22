import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# Carregando os dados 
df_base = pd.read_csv('base_tratada.csv', sep=',', parse_dates=[0], decimal=',', thousands='.')

# Configuração inicial do Streamlit
st.set_page_config(
    page_title = 'Uso de modelos de machine learning para seleção de candidatos',
    layout='wide'
)

# Título Geral
st.markdown("<h1 style='text-align: center; '> Uso de modelos de machine learning para seleção de candidatos</h1>", unsafe_allow_html = True)

# Definindo as páginas
st.sidebar.image('Logo.jpeg', width=200)
paginas = ["Análise Exploratória", "Relatório ML", 'Modelo de previsão']
pagina_selecionada = st.sidebar.selectbox("Escolha uma página:", paginas)

# Conteúdo da página Análise Exploratória
if pagina_selecionada == "Análise Exploratória":

    st.markdown('# Relatório análise exploratória Decision')

    st.markdown("---")

    st.markdown("### 1. OBJETIVO")
    st.markdown('Realizar uma análise exploratória da base de candidatos da empresa Decision, especializada em recrutamento para o setor de TI. A análise visa gerar insights sobre o processo seletivo e comportamento dos candidatos, com base em dados estruturados e visualizações.')

    st.markdown("---")

    st.markdown("### 2. ANÁLISE EXPLORATÓRIA (EDA)")

    # 2.1 STATUS DOS CANDIDATOS
    status_counts = df_base["situacao_candidado"].value_counts(ascending=True).reset_index()
    status_counts.columns = ['Status', 'Quantidade']
    fig = px.bar(status_counts,
                x='Quantidade',
                y='Status',
                orientation='h',
                title='Distribuição dos Status dos Candidatos',
                labels={'Status': 'Status', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside') 
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Status", title_x=0.5, height=600)
    st.plotly_chart(fig)

    st.markdown("#### 2.1 STATUS DOS CANDIDATOS")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar em que etapa do processo os candidatos estão ou foram encerrados.')
    st.markdown('##### **Insights**')
    st.markdown('* O **volume de “Prospect” e “Encaminhado ao Requisitante”** é bem grande (acima de 50mil), mas poucos seguem até o final;')
    st.markdown('* Aproximadamente **6% dos candidatos** chegaram de fato à contratação (Contratado pela Decision, Contratado como Hunting, Proposta Aceita, Documentação CLT, Documentação PJ, Documentação Cooperado, Aprovado);')
    st.markdown('* Mais de **12% foram reprovados** em alguma etapa (cliente, RH ou requisitante).')
    st.markdown('* O status “Desistiu" é alto, indicando possíveis falhas de comunicação ou proposta desalinhada.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* **Quase 4 mil candidatos “Inscritos”* - Candidatos que demonstraram interesse, mas não foram sequer triados ou avaliados. Pode sugerir uma falta de agilidade operacional da equipe de recrutamento.')
    st.markdown('* **Altos índices de desistência ou desinteresse** - Com mais de 2.300 candidatos desistindo ou recusando participar, há sinais de:')
    st.markdown('* * Baixa atratividade das vagas.')
    st.markdown('* * Comunicação ineficaz com os candidatos.')
    st.markdown('* * Processos muito longos ou pouco transparentes.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Automatizar triagens iniciais e comunicações** - Reduz o volume parado no status “Inscrito” ou “Prospect”. Garante que todos os candidatos tenham uma resposta mais rápida.')
    st.markdown('* **Monitorar tempo médio em cada etapa** - Evita que perfis fiquem por longos períodos sem retorno. Estabelecer SLAs com gestores/requisitantes pode melhorar o fluxo.')
    st.markdown('* **Revisar o processo de qualificação de perfis** - Melhorar o match entre perfil e vaga, reduzindo envios irrelevantes. Tornar mais ágil e preciso o encaminhamento de candidatos ao requisitante.')
    st.markdown('* **Acompanhar as causas de desistência** - Criar campos ou análises para identificar por que os candidatos desistem.')

    st.markdown("---")

    # 2.2 CONTRATAÇÕES POR RECRUTADOR
    contratados = df_base[df_base["foi_contratado"] == True]
    contratacoes_recrutador = contratados["recrutador"].value_counts(ascending=True).reset_index()
    contratacoes_recrutador.columns = ['Recrutador', 'Contratações']
    fig = px.bar(contratacoes_recrutador,
                x='Contratações',
                y='Recrutador',
                orientation='h',
                title='Contratações por Recrutador',
                labels={'Recrutador': 'Recrutador', 'Contratações': 'Quantidade de Candidatos Contratados'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade de Candidatos Contratados", yaxis_title="Recrutador", title_x=0.5, height=1200, bargap=0.1)
    st.plotly_chart(fig)

    st.markdown("#### 2.2 CONTRATAÇÕES POR RECRUTADOR")
    st.markdown('##### **Objetivo**')
    st.markdown('Avaliar a **performance individual** dos recrutadores com base no número de contratações realizadas.')
    st.markdown('##### **Insights**')
    st.markdown('* **Forte concentração:** poucos recrutadores são responsáveis pela maior parte das contratações.')
    st.markdown('* Muitos recrutadores com 10 ou menos contratações.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('Recrutadores com baixa entrega constante pode indicar:')
    st.markdown('* Desalinhamento com a estratégia de sourcing;')
    st.markdown('* Menor volume de vagas sob gestão;')
    st.markdown('* Baixa efetividade nas etapas de triagem ou relacionamento com o cliente;')
    st.markdown('* Rotação alta de funcionários no cargo.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Realizar benchmarking interno para entender o que os top performers fazem de diferente.')
    st.markdown('* Redistribuir vagas ou oferecer suporte específico aos recrutadores com menor volume.')
    st.markdown('* Implementar treinamentos baseados nos métodos dos mais eficazes.')

    st.markdown("---")

    # 2.3 TAXA DE CONVERSÃO POR RECRUTADOR
    total_candidatos = df_base['recrutador'].value_counts()
    contratados = df_base[df_base['situacao_candidado'] == 'Contratado pela Decision']['recrutador'].value_counts()
    taxa_conversao = (contratados / total_candidatos * 100).dropna().sort_values(ascending=True).reset_index()
    taxa_conversao.columns = ['Recrutador', 'Taxa de Conversão (%)']
    altura_dinamica = max(300, len(taxa_conversao) * 30)
    fig = px.bar(taxa_conversao,
                x='Taxa de Conversão (%)',
                y='Recrutador',
                orientation='h',
                title='Taxa de Conversão por Recrutador',
                labels={'Recrutador': 'Recrutador', 'Taxa de Conversão (%)': 'Taxa de Conversão (%)'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:.2f}%', textposition='outside')
    fig.update_layout(xaxis_title="Taxa de Conversão (%)", yaxis_title="Recrutador", title_x=0.5, height=altura_dinamica, xaxis_range=[0, 105], bargap=0.1, hovermode="y unified")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    st.plotly_chart(fig)

    st.markdown("#### 2.3 TAXA DE CONVERSÃO POR RECRUTADOR")
    st.markdown('##### **Objetivo**')
    st.markdown('Avaliar a **eficiência de cada recrutador** na conversão de candidatos para contratação (contratações/número de indicados).')
    st.markdown('##### **Insights**')
    st.markdown('* Alguns recrutadores apresentam taxa de conversão superior a 50%, o que pode indicar uma excelente qualificação de candidatos.')
    st.markdown('* Outros têm uma taxa inferior a 10%, o que pode indicar:')
    st.markdown('* * Alto número de encaminhamentos fora do perfil.')
    st.markdown('* * Falta de clareza nos requisitos da vaga.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* Recrutadores com muitas contratações, mas baixa taxa de conversão, podem estar adotando uma abordagem de volume em vez de precisão.')
    st.markdown('* Taxa muito alta (100%) pode ser enganosa em casos com poucos candidatos encaminhados.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Acompanhar taxa de conversão como KPI padrão.')
    st.markdown('* Reforçar o alinhamento de perfil com o cliente.')
    st.markdown('* Evitar envios excessivos e pouco direcionados.')
    st.markdown('* Implementar feedback regular sobre qualidade de candidatos indicados.')

    st.markdown("---")

    # 2.4 TOP 15 CARGOS DESEJADOS PELOS CANDIDATOS
    top_cargos = df_base['objetivo_profissional'].value_counts().head(15).reset_index()
    top_cargos.columns = ['Cargo Desejado', 'Quantidade']
    top_cargos = top_cargos.sort_values(by='Quantidade', ascending=True)
    fig = px.bar(top_cargos,
                x='Quantidade',
                y='Cargo Desejado',
                orientation='h',
                title='Top 15 Cargos Desejados pelos Candidatos',
                labels={'Cargo Desejado': 'Cargo Desejado', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Cargo Desejado", title_x=0.5, height=600, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.4 TOP 15 CARGOS DESEJADOS PELOS CANDIDATOS")
    st.markdown('##### **Objetivo**')
    st.markdown('Identificar quais cargos são mais almejados pelos candidatos, com base nos dados preenchidos em “Objetivo Profissional” ou posição desejada.')
    st.markdown('##### **Insights**')
    st.markdown('* Desenvolvedor é disparado o cargo mais almejado, com **499** candidatos declarando esse objetivo.')
    st.markdown('* Há uma forte presença de **cargos relacionados a SAP**.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* **Sobreposição de cargos**: diferentes candidatos usam nomenclaturas variadas para cargos semelhantes, dificultando análises mais consolidadas.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Padronizar a nomenclatura dos cargos desejados** por meio de um dicionário de equivalência. Exemplo: agrupar “Desenvolvedor Java”, “Java” e “Desenvolvedor” em uma categoria unificada.')

    st.markdown("---")

    # 2.5 DISTRIBUIÇÃO DE CANDIDATOS POR ESTADO
    top_estados = df_base['estado'].value_counts().head(15).reset_index()
    top_estados.columns = ['Estado', 'Quantidade de Candidatos']
    top_estados = top_estados.sort_values(by='Quantidade de Candidatos', ascending=True)
    fig = px.bar(top_estados,
                x='Quantidade de Candidatos',
                y='Estado',
                orientation='h',
                title='Distribuição de Candidatos por Estado',
                labels={'Estado': 'Estado', 'Quantidade de Candidatos': 'Quantidade de Candidatos'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade de Candidatos", yaxis_title="Estado", title_x=0.5, height=600, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.5 DISTRIBUIÇÃO DE CANDIDATOS POR ESTADO")
    st.markdown('##### **Objetivo**')
    st.markdown('Entender a origem geográfica dos candidatos e sua concentração por estado.')
    st.markdown('##### **Insights**')
    st.markdown('* Fortíssima concentração em São Paulo, seguida por Minas Gerais e Rio de Janeiro.')
    st.markdown('* Pode refletir maior volume de vagas nessas regiões ou divulgação mais eficaz.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* Estados com poucos candidatos podem indicar uma divulgação regional ineficiente.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Investir em divulgação regional direcionada, especialmente para posições remotas.')
    st.markdown('* Avaliar onde há potencial não explorado de talentos.')

    st.markdown("---")

    # 2.6 TOP 15 CIDADES COM MAIS CANDIDATOS
    top_cidades = df_base['cidade'].value_counts().head(15).reset_index()
    top_cidades.columns = ['Cidade', 'Quantidade']
    top_cidades = top_cidades.sort_values(by='Quantidade', ascending=True)
    fig = px.bar(top_cidades,
                x='Quantidade',
                y='Cidade',
                orientation='h',
                title='Top 15 Cidades com Mais Candidatos',
                labels={'Cidade': 'Cidade', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Cidade", title_x=0.5, height=600, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.6 TOP 15 CIDADES COM MAIS CANDIDATOS")
    st.markdown('##### **Objetivo**')
    st.markdown('Aprofundar a análise geográfica para o nível de município.')
    st.markdown('##### **Insights**')
    st.markdown('* Concentração muito alta em São Paulo capital.')
    st.markdown('* Cidades do interior e outras capitais como BH e RJ aparecem com menor expressão.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* Risco de dependência de um único polo de talento.')
    st.markdown('* Falta de diversidade regional no pipeline.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Ampliar o escopo geográfico das divulgações, especialmente com vagas remotas.')
    st.markdown('* Criar campanhas de atração regionalizadas.')

    st.markdown("---")

    # 2.7 TOP 15 CIDADES COM MAIS CANDIDATOS
    top_cargos = df_base['titulo'].value_counts().head(15).reset_index()
    top_cargos.columns = ['Cargo', 'Quantidade']
    top_cargos = top_cargos.sort_values(by='Quantidade', ascending=True)
    fig = px.bar(top_cargos,
                x='Quantidade',
                y='Cargo',
                orientation='h',
                title='Top 15 Cargos com Mais Candidatos',
                labels={'Cargo': 'Cargo', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Cargo", title_x=0.5, height=600, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.7 TOP 15 CIDADES COM MAIS CANDIDATOS")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar os cargos que atraíram o maior número de candidatos no processo seletivo, com base nos registros efetivos de candidatura por cargo.')
    st.markdown('##### **Insights**')
    st.markdown('* SAP SD lidera com 272 candidatos, seguido por Backend Dev (169) e Scrum Master (158), demonstrando forte concentração de talentos em áreas técnicas e gestão ágil.')
    st.markdown('* Papéis de gestão como Gerente de Projetos (90) e Project Manager (72) também aparecem com volume expressivo.')
    st.markdown('* A presença de Business Analyst (87) e Scrum Master (158) mostra interesse por profissionais com perfil de ponte entre áreas técnicas e de negócio.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* **Redundância de nomenclatura** novamente pode estar fragmentando análises. Exemplo: “Project Manager” e “Gerente de Projetos” aparecem separadamente.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Unificar nomenclaturas para cargos similares nas análises e filtros do sistema, garantindo maior precisão nos relatórios.')

    st.markdown("---")

    # 2.8 DISTRIBUIÇÃO POR MODALIDADE DE TRABALHO
    modalidade_counts = df_base['modalidade'].value_counts().reset_index()
    modalidade_counts.columns = ['Modalidade', 'Quantidade']
    modalidade_counts = modalidade_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(modalidade_counts,
                x='Quantidade',
                y='Modalidade',
                orientation='h',
                title='Distribuição por Modalidade de Trabalho',
                labels={'Modalidade': 'Modalidade', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Modalidade", title_x=0.5, height=500, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.8 DISTRIBUIÇÃO POR MODALIDADE DE TRABALHO")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar a distribuição dos candidatos contratados por tipo de vínculo de trabalho (modalidade), com o intuito de entender a preferência ou tendência nos formatos de contratação adotados.')
    st.markdown('##### **Insights**')
    st.markdown('* A modalidade **“Cooperado”** foi a mais frequente, com **590 contratações**, indicando que esse modelo tem forte presença na empresa/parceiros.')
    st.markdown('* **PJ (Pessoa Jurídica)** também apresentou alta adesão, com **540 contratações**, revelando uma significativa preferência (ou necessidade) por relações mais flexíveis e com menor vínculo empregatício formal.')
    st.markdown('* O modelo tradicional **CLT** teve **406 contratações**, ocupando a terceira posição.')
    st.markdown('* Modalidades mais específicas, como **Hunting (77) e CLT - Estratégico (70)**, possuem números mais modestos, sugerindo que são utilizados para perfis mais nichados ou com processos diferenciados.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* O volume elevado de PJ e Cooperado (totalizando 1.130 contratações) pode indicar uma dependência de modelos mais flexíveis, o que pode ter impactos jurídicos, tributários e trabalhistas caso não estejam bem estruturados.')
    st.markdown('* A modalidade CLT tradicional tem menor representatividade, o que pode impactar na retenção de talentos, especialmente para perfis que priorizam estabilidade.')
    st.markdown('* A presença de modalidades distintas (CLT, CLT Estratégico, Hunting) pode dificultar a padronização de processos internos e benefícios.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Oferecer mais transparência ao candidato durante o processo seletivo sobre o tipo de contrato. Isso pode reduzir desistências futuras por desalinhamento de expectativas.')

    st.markdown("---")

    # 2.9 DISTRIBUIÇÃO POR NÍVEL PROFISSIONAL
    nivel_prof_counts = df_base['nivel_profissional'].value_counts().reset_index()
    nivel_prof_counts.columns = ['Nível Profissional', 'Quantidade']
    nivel_prof_counts = nivel_prof_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(nivel_prof_counts,
                x='Quantidade',
                y='Nível Profissional',
                orientation='h',
                title='Distribuição por Nível Profissional',
                labels={'Nível Profissional': 'Nível Profissional', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Nível Profissional", title_x=0.5, height=500, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.9 DISTRIBUIÇÃO POR NÍVEL PROFISSIONAL")
    st.markdown('##### **Objetivo**')
    st.markdown('Avaliar a distribuição dos profissionais contratados de acordo com seus níveis de senioridade, com o intuito de entender qual perfil é mais demandado no processo de recrutamento.')
    st.markdown('##### **Insights**')
    st.markdown('* A maior parte das contratações foi para o nível Sênior, com 175 profissionais, representando mais de 50% do total analisado.')
    st.markdown('* Em seguida, os níveis Especialista (46) e Pleno (27) aparecem com valores significativamente menores.')
    st.markdown('* Os níveis Júnior (9) e Estagiário (1) são os menos contratados, o que demonstra uma baixa procura por perfis em início de carreira.')
    st.markdown('* Perfis de liderança ou gestão (como Gerente e Líder) aparecem com números discretos (2 e 15 contratações, respectivamente).')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* **Dependência de perfis seniores**: embora traga maturidade técnica ao time, pode significar **maior custo de folha** e maior dificuldade para encontrar talentos disponíveis.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Reavaliar a estrutura de cargos e planos de carreira: criar trilhas claras de evolução pode incentivar a retenção de talentos e reduzir a necessidade de contratar apenas perfis seniores;')
    st.markdown('* Realizar uma análise mais profunda para entender se as exigências das vagas estão muito elevadas (por exemplo, exigindo senioridade para funções operacionais).')

    st.markdown("---")

    # 2.10 TOP 15 ÁREAS DE ATUAÇÃO DOS CANDIDATOS
    area_counts = df_base['area_atuacao'].value_counts().head(15).reset_index()
    area_counts.columns = ['Área de Atuação', 'Quantidade']
    area_counts = area_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(area_counts,
                x='Quantidade',
                y='Área de Atuação',
                orientation='h',
                title='Top 15 Áreas de Atuação dos Candidatos',
                labels={'Área de Atuação': 'Área de Atuação', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Área de Atuação", title_x=0.5, height=600, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.10 TOP 15 ÁREAS DE ATUAÇÃO DOS CANDIDATOS")
    st.markdown('##### **Objetivo**')
    st.markdown('Identificar as áreas de atuação mais comuns entre os candidatos cadastrados, permitindo alinhar a estratégia de recrutamento com a oferta real de talentos disponíveis no mercado.')
    st.markdown('##### **Insights**')
    st.markdown('* A área **TI - Desenvolvimento/Programação** lidera com **2.251 candidatos**, refletindo a forte demanda e oferta por desenvolvedores no mercado de tecnologia;')
    st.markdown('* Em segundo lugar está **TI - SAP** com **1.826 candidatos**, o que demonstra uma base significativa de profissionais especializados nesse ecossistema — algo relevante por se tratar de uma área mais técnica e específica;')
    st.markdown('* **TI - Projetos (742) e Área Administrativa (581)** também têm volume expressivo, indicando abertura para cargos de gestão, PMO e suporte operacional;')
    st.markdown('* Áreas como **Infraestrutura (460), Controladoria (405) e Processos/Negócios (339)** demonstram que o banco de talentos também contempla funções mais estratégicas e operacionais dentro das empresas;')
    st.markdown('* Outras áreas com participação significativa, embora menor, incluem: **Recursos Humanos, Qualidade/Testes, Banco de Dados, Governança e Arquitetura de TI.**')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* O volume muito elevado em Desenvolvimento/Programação pode gerar excesso de perfis semelhantes, dificultando a triagem e o match ideal com as vagas — especialmente se os perfis não estiverem bem segmentados por linguagem ou senioridade.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Criar filtros automáticos** para áreas saturadas, como Desenvolvimento, para facilitar a priorização de perfis mais aderentes às vagas abertas;')
    st.markdown('* **Aproveitar o volume alto de perfis em SAP e Desenvolvimento** para projetos de hunting estratégico ou ofertas de upskilling, buscando ampliar a atratividade da empresa para esses perfis.')

    st.markdown("---")

    # 2.11 DISTRIBUIÇÃO POR NÍVEL ACADÊMICO
    nivel_acad_counts = df_base['nivel_academico_x'].value_counts().reset_index()
    nivel_acad_counts.columns = ['Nível Acadêmico', 'Quantidade']
    nivel_acad_counts = nivel_acad_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(nivel_acad_counts,
                x='Quantidade',
                y='Nível Acadêmico',
                orientation='h',
                title='Distribuição por Nível Acadêmico',
                labels={'Nível Acadêmico': 'Nível Acadêmico', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Nível Acadêmico", title_x=0.5, height=800, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.11 DISTRIBUIÇÃO POR NÍVEL ACADÊMICO")
    st.markdown('##### **Objetivo**')
    st.markdown('Compreender o grau de escolaridade dos candidatos cadastrados no banco de talentos, permitindo alinhar os requisitos das vagas com o perfil real disponível no mercado.')
    st.markdown('##### **Insights**')
    st.markdown('* A grande maioria dos candidatos possui **Ensino Superior Completo (5.319)**, demonstrando uma base qualificada e compatível com vagas que exigem formação acadêmica mais sólida;')
    st.markdown('* Em seguida, destaca-se o número de **candidatos com Pós-Graduação Completa (3.044)**, o que indica forte presença de profissionais com formação especializada;')
    st.markdown('* A soma de candidatos que **ainda estão cursando Ensino Superior (1.256)** ou possuem **formação incompleta (574)** também é significativa, o que pode ser interessante para posições júniores ou estágio;')
    st.markdown('* Outros níveis com destaque: **Pós-Graduação Cursando (494), Mestrado Completo (284) e Ensino Técnico Completo (186)**;')
    st.markdown('* Há presença discreta de candidatos com níveis mais baixos (Ensino Médio, Fundamental), reforçando o foco da base em perfis técnicos e especializados.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* Candidatos em formação avançada (mestrado/doutorado) são minoria — o que é esperado, mas pode ser um ponto de limitação para vagas mais acadêmicas, estratégicas ou de pesquisa.')
    st.markdown('* A presença de candidatos com formação incompleta (superior, técnico ou pós) exige atenção na triagem para garantir aderência aos requisitos das vagas.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Criar filtros automatizados para escolaridade mínima**, ajudando na qualificação e direcionamento mais rápido dos candidatos;')
    st.markdown('* **Alinhar as exigências de escolaridade nas vagas** com o perfil real disponível: por exemplo, exigir pós-graduação apenas quando realmente necessário, pois o volume de candidatos com superior completo já é alto.')

    st.markdown("---")

    # 2.12 DISTRIBUIÇÃO POR FAIXA ETÁRIA
    df_base['data_nascimento'] = pd.to_datetime(df_base['data_nascimento'], errors='coerce')
    today = datetime(2025, 7, 22)
    df_base['idade'] = df_base['data_nascimento'].apply(lambda x: today.year - x.year if pd.notnull(x) else None)
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['Até 20', '21-30', '31-40', '41-50', '51-60', 'Acima de 60']
    df_base['faixa_etaria'] = pd.cut(df_base['idade'], bins=bins, labels=labels, right=False, ordered=True)
    faixa_counts = df_base['faixa_etaria'].value_counts().sort_index().reset_index()
    faixa_counts.columns = ['Faixa Etária', 'Quantidade']
    fig = px.bar(faixa_counts,
                x='Quantidade',
                y='Faixa Etária',
                orientation='h',
                title='Distribuição por Faixa Etária',
                labels={'Faixa Etária': 'Faixa Etária', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Faixa Etária", title_x=0.5, height=500, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.12 DISTRIBUIÇÃO POR FAIXA ETÁRIA")
    st.markdown('##### **Objetivo**')
    st.markdown('Avaliar a distribuição etária dos candidatos cadastrados no banco de talentos, a fim de compreender o perfil predominante e identificar possíveis lacunas ou oportunidades de inclusão etária nas estratégias de recrutamento.')
    st.markdown('##### **Insights**')
    st.markdown('* A **faixa etária predominante é de 51 a 60 anos, com 8.311 candidatos**, o que representa mais da metade de toda a base analisada.')
    st.markdown('* As faixas **41-50 anos (3.718) e 31-40 anos (3.266)** também se destacam, compondo uma parcela significativa de candidatos com maior bagagem profissional.')
    st.markdown('* Candidatos com *menos de 30 anos** são minoritários:')
    st.markdown('* * Apenas **999 estão entre 21-30 anos**, e')
    st.markdown('* * **132 com até 20 anos**, indicando baixo volume de perfis em início de carreira.')
    st.markdown('* A presença de **1.107 candidatos acima de 60 anos** indica interesse contínuo por recolocação mesmo em idade mais avançada.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* A base atual apresenta **concentração em perfis maduros** e experientes, o que pode ser ótimo para cargos seniores, mas limita a atuação em vagas júnior, estágios e posições de entrada.')
    st.markdown('* A **inclusão de candidatos acima de 60 anos** é relevante, mas exige atenção às práticas de diversidade etária e à adequação das oportunidades oferecidas a esse público.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Adaptar filtros de busca e comunicação das vagas para atingir diferentes faixas etárias com estratégias específicas.')

    st.markdown("---")

    # 2.13 DISTRIBUIÇÃO POR GÊNERO
    sexo_counts = df_base['sexo'].value_counts().reset_index()
    sexo_counts.columns = ['Gênero', 'Quantidade']
    sexo_counts = sexo_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(sexo_counts,
                x='Quantidade',
                y='Gênero',
                orientation='h',
                title='Distribuição por Gênero',
                labels={'Gênero': 'Gênero', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Gênero", title_x=0.5, height=400, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.13 DISTRIBUIÇÃO POR GÊNERO")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar a distribuição de gênero entre os candidatos cadastrados para identificar possíveis assimetrias e apoiar estratégias de diversidade e inclusão no recrutamento.')
    st.markdown('##### **Insights**')
    st.markdown('* A maioria absoluta dos candidatos é do **gênero masculino**, com **10.735 registros**, representando cerca de **83% da base analisada**.')
    st.markdown('* Apenas **2.143 candidatas do gênero feminino** foram identificadas, o que representa **aproximadamente 17% do total**.')
    st.markdown('* O gráfico revela uma **grande disparidade de gênero**, especialmente relevante em setores como tecnologia, onde o desequilíbrio é historicamente recorrente.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* A baixa participação feminina pode indicar uma falta de equidade de gênero no funil de talentos, especialmente em áreas técnicas ou cargos seniores.')
    st.markdown('* Tal desproporção pode limitar o alcance de metas de diversidade organizacional e comprometer a pluralidade de perspectivas nos times.')
    st.markdown('* O viés de gênero pode estar presente já nas etapas de divulgação, linguagem das vagas ou nos filtros de recrutamento utilizados.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Revisar a comunicação das vagas**, utilizando linguagem neutra e inclusiva para ampliar o alcance a talentos femininos.')
    st.markdown('* **Monitorar o funil completo (inscrição, triagem, entrevista e contratação)** por gênero para identificar em qual fase há maior perda de candidatas.')

    st.markdown("---")

    # 2.14 TOP 10 CANAIS DE ORIGEM DOS CANDIDATOS
    origem_counts = df_base['sabendo_de_nos_por'].value_counts().head(10).reset_index()
    origem_counts.columns = ['Origem', 'Quantidade']
    origem_counts = origem_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(origem_counts,
                x='Quantidade',
                y='Origem',
                orientation='h',
                title='Top 10 Canais de Origem dos Candidatos',
                labels={'Origem': 'Origem', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Origem", title_x=0.5, height=550, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.14 TOP 10 CANAIS DE ORIGEM DOS CANDIDATOS")
    st.markdown('##### **Objetivo**')
    st.markdown('Identificar os principais canais de aquisição de candidatos, visando otimizar estratégias de divulgação de vagas e investimento em fontes com maior retorno.')
    st.markdown('##### **Insights**')
    st.markdown('* A categoria **“Outros”** lidera com **7.573 candidatos**, o que representa mais de **50% do total** — isso dificulta a rastreabilidade e análise real da performance dos canais.')
    st.markdown('* O **“Site de Empregos”** é o canal rastreável com maior volume de origem (2.822), seguido de **“Anúncio” (1.598) e “Indeed” (1.493)**, indicando forte dependência de plataformas abertas.')
    st.markdown('* O **site próprio da empresa gerou 914 candidatos**, mostrando bom desempenho como canal orgânico.')
    st.markdown('* As indicações (de amigos, colaboradores e clientes) somadas geram **1.390 candidatos**, mostrando a relevância da estratégia.')
    st.markdown('* Canais como **“Escolas/Faculdades” e “Feiras de RH”** têm impacto praticamente nulo (menos de 20 candidatos no total).')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* A presença elevada de registros na categoria **“Outros”** dificulta ações estratégicas baseadas em dados e pode ocultar o verdadeiro potencial de canais específicos.')
    st.markdown('* **Baixa performance de canais de atração de talentos jovens** (ex: faculdades e feiras), o que pode comprometer a renovação de talentos e ações de employer branding voltadas a estudantes.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Revisar e padronizar os campos de origem de candidatos** para reduzir o uso do campo “Outros” e permitir melhor rastreabilidade.')
    st.markdown('* **Investir mais em canais com alto volume e bom custo-benefício**, como sites de emprego específicos e o site próprio.')
    st.markdown('* **Fortalecer o programa de indicações**, criando campanhas de incentivo para colaboradores e clientes, dado o bom desempenho desses canais.')

    st.markdown("---")

    # 2.15 DISTRIBUIÇÃO DE CANDIDATOS PCD
    pcd_counts = df_base['pcd'].astype(str).value_counts().reset_index()
    pcd_counts.columns = ['PCD', 'Quantidade']
    pcd_counts = pcd_counts.sort_values(by='Quantidade', ascending=False)
    fig = px.bar(pcd_counts,
                x='Quantidade',
                y='PCD',
                orientation='h',
                title='Distribuição de Candidatos PCD',
                labels={'PCD': 'Candidato PCD', 'Quantidade': 'Quantidade'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title="Quantidade", yaxis_title="Candidato PCD", title_x=0.5, height=350, bargap=0.1, hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.15 DISTRIBUIÇÃO DE CANDIDATOS PCD")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar a proporção de candidatos que se autodeclaram como Pessoas com Deficiência (PCD) em relação ao total de candidatos cadastrados, com o intuito de identificar representatividade e possíveis lacunas na inclusão desses profissionais nos processos seletivos conduzidos pela Decision.')
    st.markdown('##### **Insights**')
    st.markdown('* Apenas **186 candidatos** se declararam como PCD, enquanto **10.336 candidatos** não se autodeclararam;')
    st.markdown('* Isso representa aproximadamente **1,8% do total** de candidatos cadastrados.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* A baixa representatividade de candidatos PCD pode indicar:')
    st.markdown('* * Barreiras de acessibilidade no cadastro ou divulgação de vagas.')
    st.markdown('* * Falta de campanhas direcionadas para atração de PCDs.')
    st.markdown('* * Desalinhamento entre os requisitos das vagas dos clientes e os perfis de profissionais PCD disponíveis no mercado.')
    st.markdown('* * Importante considerar que a Decision atua como intermediária para vagas de clientes, o que reforça a necessidade de avaliar se há demanda real de contratação PCD por parte desses clientes.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* Revisar e adaptar os canais de atração para torná-los mais acessíveis e inclusivos, garantindo que candidatos PCD sintam-se acolhidos desde o primeiro contato;')
    st.markdown('* Sugerir aos clientes a abertura de vagas afirmativas ou a flexibilização de requisitos técnicos, quando viável, para ampliar a diversidade dos candidatos.')

    st.markdown("---")

    # 2.16 TAXA DE APROVAÇÃO POR GÊNERO
    status_aprovado = df_base[df_base['situacao_candidado'].str.lower().str.contains('aprovado', na=False)]
    aprov_por_genero = status_aprovado['sexo'].value_counts()
    total_por_genero = df_base['sexo'].value_counts()
    taxa_aprov_genero = (aprov_por_genero / total_por_genero * 100).dropna().reset_index()
    taxa_aprov_genero.columns = ['Gênero', 'Taxa de Aprovação (%)']
    taxa_aprov_genero = taxa_aprov_genero.sort_values(by='Taxa de Aprovação (%)', ascending=False)
    fig = px.bar(taxa_aprov_genero,
                x='Taxa de Aprovação (%)',
                y='Gênero',
                orientation='h',
                title='Taxa de Aprovação por Gênero',
                labels={'Gênero': 'Gênero', 'Taxa de Aprovação (%)': 'Taxa de Aprovação (%)'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Taxa de Aprovação (%)", yaxis_title="Gênero", title_x=0.5, height=400, bargap=0.1, xaxis_range=[0, taxa_aprov_genero['Taxa de Aprovação (%)'].max() * 1.1], hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.16 TAXA DE APROVAÇÃO POR GÊNERO")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar a taxa de aprovação de candidatos nos processos seletivos conduzidos pela Decision, segmentada por gênero, para verificar a existência de disparidades ou possíveis vieses ao longo da jornada de seleção.')
    st.markdown('##### **Insights**')
    st.markdown('* A taxa de aprovação é praticamente igual entre os gêneros:')
    st.markdown('* * **Masculino: 11,8%**')
    st.markdown('* * **Feminino: 11,6%**')
    st.markdown('* A diferença de **0,2 ponto percentual** é estatisticamente irrelevante, o que sugere **equidade de gênero** na fase final do processo seletivo, ou seja, na contratação.')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* Apesar da taxa de aprovação ser equivalente, é necessário considerar que o volume de **candidatos do gênero masculino é significativamente maior** (conforme gráfico acima), o que pode mascarar desigualdades em etapas anteriores (triagem, entrevista, etc.).')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Aprofundar a análise por etapa do processo** (ex: triagem curricular, entrevistas, testes técnicos) para garantir que não haja viés ao longo da jornada;')
    st.markdown('* **Monitorar a representatividade de gênero por área de atuação e nível hierárquico**, garantindo que a equidade se mantenha em diferentes contextos (ex: liderança, tecnologia etc.);')
    st.markdown('* **Incentivar os clientes da Decision** a manter práticas de recrutamento e seleção que favoreçam a equidade de gênero, com foco em critérios técnicos e comportamentais isentos de vieses.')

    st.markdown("---")

    # 2.17 TAXA DE APROVAÇÃO POR NÍVEL ACADÊMICO
    aprovados = df_base[df_base['situacao_candidado'].str.lower().str.contains('aprovado', na=False)]
    total_nivel = df_base['nivel_academico_x'].value_counts()
    aprov_nivel = aprovados['nivel_academico_x'].value_counts()
    taxa_aprov_nivel = (aprov_nivel / total_nivel * 100).dropna().reset_index()
    taxa_aprov_nivel.columns = ['Nível Acadêmico', 'Taxa de Aprovação (%)']
    taxa_aprov_nivel = taxa_aprov_nivel.sort_values(by='Taxa de Aprovação (%)', ascending=False)
    fig = px.bar(taxa_aprov_nivel,
                x='Taxa de Aprovação (%)',
                y='Nível Acadêmico',
                orientation='h',
                title='Taxa de Aprovação por Nível Acadêmico',
                labels={'Nível Acadêmico': 'Nível Acadêmico', 'Taxa de Aprovação (%)': 'Taxa de Aprovação (%)'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Taxa de Aprovação (%)", yaxis_title="Nível Acadêmico", title_x=0.5, height=800, bargap=0.1, xaxis_range=[0, taxa_aprov_nivel['Taxa de Aprovação (%)'].max() * 1.1], hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.17 TAXA DE APROVAÇÃO POR NÍVEL ACADÊMICO")
    st.markdown('##### **Objetivo**')
    st.markdown('Avaliar como o nível acadêmico impacta a taxa de aprovação dos candidatos nos processos seletivos conduzidos pela Decision, a fim de identificar padrões, possíveis vieses ou oportunidades de melhoria na triagem e encaminhamento de perfis aos clientes contratantes.')
    st.markdown('##### **Insights**')
    st.markdown('* As **maiores taxas de aprovação** estão concentradas em candidatos com:')
    st.markdown('* * **Ensino Médio Incompleto e Cursando: 33,3%**')
    st.markdown('* * **Ensino Técnico Incompleto: 25,0%**')
    st.markdown('* Candidatos com **nível superior completo ou pós-graduação** apresentam taxas moderadas de aprovação, entre **8,6% e 13,5%.**')
    st.markdown('* Os **níveis acadêmicos mais altos (mestrado e doutorado completo) têm baixas taxas de aprovação:**')
    st.markdown('* * Mestrado Completo: 9,5%')
    st.markdown('* * Doutorado Completo: 7,1%')
    st.markdown('* **Ensino Fundamental Completo e Incompleto** apresentam as **piores taxas**, sendo **0% e 2,1%**, respectivamente.')
    st.markdown('* **Ensino Médio Completo** também tem taxa baixa: **5,2%.**')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* A alta taxa de aprovação entre candidatos com **ensino médio incompleto/cursando** pode indicar:')
    st.markdown('* * Baixo volume de candidatos nesse grupo (base pequena → distorção da taxa).')
    st.markdown('* * Aprovações em **vagas operacionais ou técnicas** de entrada.')
    st.markdown('* A baixa aprovação de candidatos com **pós-graduação, mestrado ou doutorado completo** pode sugerir:')
    st.markdown('* * **Overqualification (perfil acima do que a vaga exige).**')
    st.markdown('* * Falta de alinhamento entre a formação e as vagas ofertadas pelos clientes da Decision.')
    st.markdown('* É necessário verificar se as vagas dos clientes estão predominantemente em níveis técnicos/operacionais, o que naturalmente reduz a absorção de perfis mais qualificados.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Analisar a compatibilidade entre a complexidade das vagas dos clientes e o perfil acadêmico dos candidatos.**')
    st.markdown('* **Ajustar o direcionamento de candidatos com alta qualificação** para vagas compatíveis, evitando desperdício de talentos.')
    st.markdown('* Considerar ações de **requalificação ou reorientação de carreira** para candidatos com baixa escolaridade, que demonstram interesse, mas têm baixa conversão.')
    st.markdown('* **Segmentar o banco de talentos por escolaridade + área de atuação**, facilitando recomendações mais precisas para os recrutadores.')
    st.markdown('* Incluir alertas no sistema para detectar **descompasso entre qualificação do candidato e nível da vaga**, otimizando o match.')

    st.markdown("---")

# 2.18 TOP 10 CANAIS COM MAIOR TAXA DE APROVAÇÃO
    aprovados = df_base[df_base['situacao_candidado'].str.lower().str.contains('aprovado', na=False)]
    total_origem = df_base['sabendo_de_nos_por'].value_counts()
    aprov_origem = aprovados['sabendo_de_nos_por'].value_counts()
    taxa_aprov_origem = (aprov_origem / total_origem * 100).dropna().sort_values(ascending=False).head(10).reset_index()
    taxa_aprov_origem.columns = ['Origem', 'Taxa de Aprovação (%)']
    taxa_aprov_origem = taxa_aprov_origem.sort_values(by='Taxa de Aprovação (%)', ascending=False)
    fig = px.bar(taxa_aprov_origem,
                x='Taxa de Aprovação (%)',
                y='Origem',
                orientation='h',
                title='Top 10 Canais com Maior Taxa de Aprovação',
                labels={'Origem': 'Origem', 'Taxa de Aprovação (%)': 'Taxa de Aprovação (%)'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Taxa de Aprovação (%)", yaxis_title="Origem", title_x=0.5, height=550, bargap=0.1, xaxis_range=[0, taxa_aprov_origem['Taxa de Aprovação (%)'].max() * 1.1], hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.18 TOP 10 CANAIS COM MAIOR TAXA DE APROVAÇÃO")
    st.markdown('##### **Objetivo**')
    st.markdown('Identificar quais canais de origem de candidatos resultam em maiores taxas de aprovação em processos seletivos conduzidos pela Decision, auxiliando na definição de estratégias mais eficazes de atração de talentos para os clientes.')
    st.markdown('##### **Insights**')
    st.markdown('* **Plaqueiro (50%) e Evento Universitário (44,4%)** são os canais com **melhor taxa de aprovação**, muito acima da média dos demais;')
    st.markdown('* Canais baseados em **relacionamento e engajamento direto** (como **Indicação de Colaborador - 17,3%, Feira de RH - 13,3% e Escolas/Faculdades - 12,5%**) têm performance superior à maioria dos canais digitais;')
    st.markdown('* Canais amplamente utilizados, como **Anúncio (11,7%) e Site de Empregos (13,0%)**, apresentam desempenho inferior em comparação com métodos presenciais ou de indicação;')
    st.markdown('* **Indicação de Cliente**, apesar de remeter a um canal supostamente confiável, está na base do ranking (10,5%).')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* **Canais de maior volume**, como os digitais, **não aparecem entre os mais eficazes em termos de aprovação**;')
    st.markdown('* A alta taxa dos canais como **Plaqueiro** e **Eventos Universitários** pode estar relacionada a:')
    st.markdown('* * Bases pequenas de candidatos, o que eleva artificialmente a taxa.')
    st.markdown('* * Maior **qualidade do engajamento e triagem prévia** dos perfis captados.')
    st.markdown('* Há possibilidade de **viés de qualificação** entre os canais: canais com mais esforço humano envolvido (indicação, eventos, feiras) tendem a atrair perfis mais alinhados.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Investir mais em ações presenciais ou semipresenciais**, como eventos universitários e feiras de RH, especialmente para perfis com maior taxa de conversão.')
    st.markdown('* **Estruturar programas de incentivo à indicação interna (de colaboradores)**, reforçando um canal já comprovadamente eficaz.')
    st.markdown('* Avaliar criticamente o ROI dos canais digitais com baixa conversão. Embora gerem volume, podem exigir **refinamento nas estratégias de atração ou triagem automatizada**.')
    st.markdown('* **Monitorar a representatividade de cada canal**, cruzando taxa de aprovação com volume de candidatos gerados, para decisões equilibradas.')
    st.markdown('* Criar **campanhas específicas de captação por canal de alta conversão**, adaptando a linguagem e abordagem à efetividade do meio.')

    st.markdown("---")

    # 2.19 TAXA DE APROVAÇÃO POR ESTADO
    aprovados = df_base[df_base['situacao_candidado'].str.lower().str.contains('aprovado', na=False)]
    total_estado = df_base['estado'].value_counts()
    aprov_estado = aprovados['estado'].value_counts()
    taxa_aprov_estado = (aprov_estado / total_estado * 100).dropna().sort_values(ascending=False).head(10).reset_index()
    taxa_aprov_estado.columns = ['Estado', 'Taxa de Aprovação (%)']
    taxa_aprov_estado = taxa_aprov_estado.sort_values(by='Taxa de Aprovação (%)', ascending=False)
    fig = px.bar(taxa_aprov_estado,
                x='Taxa de Aprovação (%)',
                y='Estado',
                orientation='h',
                title='Top 10 Estados com Maior Taxa de Aprovação',
                labels={'Estado': 'Estado', 'Taxa de Aprovação (%)': 'Taxa de Aprovação (%)'},
                color_discrete_sequence=['mediumpurple'])
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title="Taxa de Aprovação (%)", yaxis_title="Estado", title_x=0.5, height=550, bargap=0.1, xaxis_range=[0, taxa_aprov_estado['Taxa de Aprovação (%)'].max() * 1.1], hovermode="y unified")
    st.plotly_chart(fig)

    st.markdown("#### 2.19 TAXA DE APROVAÇÃO POR ESTADO")
    st.markdown('##### **Objetivo**')
    st.markdown('Analisar a taxa de aprovação de candidatos por estado de origem, a fim de identificar quais localidades apresentam maior conversão em processos seletivos conduzidos pela Decision para seus clientes.')
    st.markdown('##### **Insights**')
    st.markdown('* **Piauí (33,3%), Bahia (29%) e Paraíba (28,6%)** lideram com as **maiores taxas de aprovação**, bem acima da média geral.')
    st.markdown('* Os quatro primeiros colocados (todos do Nordeste, com exceção de Goiás) se destacam com índices superiores a 25%, evidenciando um **potencial relevante da região Nordeste em termos de qualidade dos candidatos aprovados.**')
    st.markdown('* **Sudeste e Sul**, tradicionalmente mais industrializados e populosos, aparecem apenas nos últimos lugares da lista (São Paulo, Minas Gerais, Rio Grande do Sul, Paraná), com **taxas entre 11,2% e 12,8%.**')
    st.markdown('##### **Pontos de atenção**')
    st.markdown('* Alta taxa de aprovação não necessariamente reflete alto volume de candidatos. Estados como Piauí e Paraíba podem ter bases pequenas, o que distorce a taxa relativa.')
    st.markdown('* Pode haver variação nas vagas ofertadas por região: estados com maior taxa podem estar recebendo demandas mais específicas (ou mais compatíveis com seu perfil populacional).')
    st.markdown('* Estados como São Paulo e Minas Gerais, apesar de possuírem mais candidatos, têm taxa de aprovação menor, o que pode indicar:')
    st.markdown('* * Maior concorrência;')
    st.markdown('* * Desalinhamento entre os perfis aplicantes e as exigências das vagas;')
    st.markdown('* * Processos seletivos mais exigentes para determinadas praças.')
    st.markdown('##### **Oportunidade de Melhoria: Ações práticas**')
    st.markdown('* **Explorar com mais intensidade o recrutamento no Nordeste e Norte**, com campanhas de atração regionalizadas para vagas remotas ou alocação nacional.')
    st.markdown('* Investigar **o perfil e as características dos candidatos aprovados nos estados com melhor desempenho**, para aplicar os aprendizados no sourcing de outras regiões.')
    st.markdown('* Considerar ações de **qualificação ou triagem mais robusta em estados com grande volume e baixa conversão**, como São Paulo e Minas, otimizando tempo e recursos.')

    st.markdown("---")

    st.markdown("### 3. OCONCLUSÃO GERAL")
    st.markdown('A análise dos dados dos candidatos processados pela Decision, considerando aspectos demográficos, educacionais e comportamentais, revela oportunidades estratégicas significativas para otimização do processo seletivo e aumento da assertividade nas contratações para os clientes da empresa.')
    st.markdown('#### Principais Insights Integrados')
    st.markdown('* Perfil Demográfico: A maior parte dos candidatos é composta por homens entre 51 e 60 anos, o que pode indicar uma base madura, porém exige atenção quanto à atualização técnica e aderência a perfis buscados em TI.')
    st.markdown('* Diversidade e Inclusão: A presença de apenas 186 candidatos PCD e uma baixa taxa de aprovação para pessoas com maior nível de escolaridade formal (pós, mestrado e doutorado) indicam desafios de inclusão e compatibilidade entre formação e expectativa de mercado.')
    st.markdown('* Gênero: A taxa de aprovação entre gêneros é praticamente igual (11,8% masculino e 11,6% feminino), sugerindo um processo seletivo relativamente neutro, apesar do volume masculino ser cinco vezes maior.')
    st.markdown('* Educação: Contrariando a lógica tradicional, os candidatos com ensino médio incompleto ou cursando tiveram as maiores taxas de aprovação (33,3%), reforçando que experiência prática e soft skills podem pesar mais que o nível acadêmico formal.')
    st.markdown('* Canais de Origem: Plaqueiros e eventos universitários apresentaram as maiores taxas de aprovação (50% e 44,4%, respectivamente), enquanto canais como anúncios ou sites genéricos, apesar de volume alto, têm taxas menores.')
    st.markdown('* Regiões: Estados do Nordeste e Norte dominaram as primeiras posições em taxa de aprovação, com destaque para Piauí, Bahia e Paraíba, revelando potenciais regiões pouco exploradas no recrutamento técnico.')
    st.markdown('#### Pontos Críticos de Atenção')
    st.markdown('* A baixa inclusão de PCDs e de candidatos com alto nível acadêmico merece investigação e ação específica.')
    st.markdown('* O volume elevado de candidatos vindos de canais genéricos (ex: “Outros”) não se converte em aprovação de forma eficiente, sugerindo desperdício de tempo no funil.')
    st.markdown('* Existe potencial não explorado em estados com alta taxa de aprovação e baixo volume de entrada, o que aponta para uma desigualdade de acesso às vagas e baixo investimento em divulgação regional.')
    st.markdown('#### Encerramento & Recomendação Estratégica')
    st.markdown('A Decision tem uma oportunidade concreta de reposicionar sua abordagem de sourcing, seleção e relacionamento com candidatos, com foco em:')
    st.markdown('* Aumentar a eficiência do funil, concentrando esforços em canais e regiões com melhor taxa de aprovação;')
    st.markdown('* Reduzir vieses geográficos e acadêmicos, compreendendo melhor as competências práticas que geram match com os perfis contratantes;')
    st.markdown('* Expandir ações de diversidade, promovendo inclusão de PCDs e ampliando a participação de mulheres e candidatos de regiões menos representadas;')
    st.markdown('* Apoiar seus clientes com inteligência de dados sobre os candidatos que mais convertem, melhorando a qualidade do alinhamento entre vagas e perfis.')
    st.markdown('Ao transformar esses dados em estratégia, a Decision pode assumir um papel consultivo mais forte junto aos seus clientes, não apenas apresentando candidatos, mas orientando o mercado sobre onde estão os melhores talentos e como atraí-los de forma eficaz.')

# Conteúdo da página Relatório ML
elif pagina_selecionada == "Relatório ML":

    st.markdown('# Relatório Técnico - Análise e Previsão do potencial de candidatos para contratação')

    st.markdown("---")

    st.markdown("### OBJETIVO")
    st.markdown("Detalhar o processo de desenvolvimento do modelo de Machine Learning, abrangendo a extração, pré-processamento, análise exploratória de dados e seleção do modelo ideal.")

    st.markdown("---")

    st.markdown("### EXTRAÇÃO DOS DADOS")
    st.markdown("Para o desenvolvimento do nosso modelo, iniciamos com a extração dos dados brutos a partir de três arquivos .json fornecidos pela Decision. Nesta etapa inicial, identificamos a presença de diversos valores nulos e a possibilidade de unificar as bases utilizando as chaves disponibilizadas no arquivo prospects.")

    st.markdown("---")

    st.markdown("### UNIFICAÇÃO DAS BASES")
    st.markdown("A unificação das bases ocorreu de forma relativamente fluida. A coluna utilizada como chave na base de vagas não apresentava valores nulos ou duplicados, o que simplificou o processo. Contudo, a base applicants exigiu tratamentos mais específicos, como a remoção de linhas em branco e IDs nulos. Após esses ajustes, procedemos com a unificação e aplicamos dois tratamentos adicionais: a padronização de células em branco via expressão regular (regex) e a eliminação de linhas completamente vazias. Exportamos essa base consolidada para um arquivo .csv, visando facilitar a análise exploratória, enquanto direcionávamos nossos esforços para o pré-processamento focado em Machine Learning.")

    st.markdown("---")

    st.markdown("### PRÉ-PROCESSAMENTO PARA MACHINE LEARNING")
    st.markdown("O pré-processamento dos dados para o modelo de Machine Learning seguiu as seguintes etapas:")
    st.markdown("* Remoção de Colunas Univariadas: Eliminamos colunas que apresentavam apenas um valor distinto, pois não contribuem para a variabilidade e poder preditivo do modelo.")
    st.markdown("* Codificação da Variável Alvo (Target): Padronizamos os valores da nossa coluna alvo para 0 (não contratado) e 1 (contratado).")
    st.markdown("* Engenharia de Features - Idade: Calculamos a idade do candidato no momento da aplicação utilizando as colunas data_nascimento e data_requisicao. Essa nova feature foi criada para investigar se a idade poderia influenciar a contratação.")
    st.markdown("* Análise de Associação com Teste Qui-Quadrado: Optamos por utilizar Tabelas de Contingência e o Teste Qui-Quadrado (χ2) para analisar a associação entre variáveis categóricas. Essa abordagem foi escolhida devido à predominância de variáveis textuais (strings) entre as mais de 100 colunas disponíveis, o que tornaria uma matriz de correlação tradicional menos informativa.")
    st.markdown("* * Critérios de Avaliação do Teste Qui-Quadrado:")
    st.markdown("* * * Frequência Esperada: Avaliamos a validade do teste observando que menos de 20% das células da matriz de contingência tivessem uma frequência esperada inferior a 5. Este é um critério comum para garantir a robustez do teste Qui-Quadrado de Pearson.")
    st.markdown("* * * Valor-p (p-value): Consideramos que colunas com um valor-p acima de 0.05 não apresentavam associação estatisticamente significativa com a variável alvo.")
    st.markdown("* * Resultados do Teste Qui-Quadrado: Após a aplicação do teste, verificamos que as colunas sexo, estado_civil e pcd não demonstraram associação estatisticamente significativa com a contratação do candidato. Este resultado é um forte indicativo de imparcialidade no processo de contratação em relação a essas características.")
    st.markdown("* Identificação de Features Relevantes: Quatro colunas passaram nos critérios do teste Qui-Quadrado: modalidade, inserido_por, nivel_ingles_x e vaga_sap. A coluna inserido_por chamou nossa atenção, e uma análise visual (gráfico comparando % de candidatos contratados vs. % de candidatos inseridos) revelou uma diferença considerável nessas porcentagens. Embora seja um ponto relevante para investigação futura com a Decision, optamos por não incluir essa coluna no treinamento do modelo no momento, considerando a possibilidade de uma correlação espúria com base nos dados disponíveis.")
    st.markdown("* Codificação Final dos Dados: Para preparar a base para o treinamento, realizamos a codificação das variáveis categóricas utilizando OneHotEncoder e mapeamento manual, conforme a natureza de cada coluna.")

    porcentagens_totais = df_base['inserido_por'].value_counts(normalize=True) * 100
    df_porcentagens_totais = porcentagens_totais.reset_index()
    df_porcentagens_totais.columns = ['inserido_por', 'Porcentagem Total']
    df_filtrado = df_base[df_base['situacao_candidado_'] == 1]
    porcentagens_filtradas = df_filtrado['inserido_por'].value_counts(normalize=True) * 100
    df_porcentagens_filtradas = porcentagens_filtradas.reset_index()
    df_porcentagens_filtradas.columns = ['inserido_por', 'Porcentagem contratados']
    df_final = pd.merge(df_porcentagens_totais, df_porcentagens_filtradas, on='inserido_por', how='outer').fillna(0)
    df_melted = df_final.melt(id_vars='inserido_por', var_name='Tipo de Porcentagem', value_name='Porcentagem')
    fig = px.bar(df_melted,
                x='inserido_por',
                y='Porcentagem',
                color='Tipo de Porcentagem',
                barmode='group',
                title='Comparação de Porcentagens por "inserido_por"',
                labels={'inserido_por': 'Origem', 'Porcentagem': 'Porcentagem (%)'},
                color_discrete_map={'Porcentagem Total': 'mediumpurple', 'Porcentagem contratados': 'plum'})
    fig.update_layout(xaxis_title='Origem', yaxis_title='Porcentagem (%)', title_x=0.5, height=800, xaxis_tickangle=-45, hovermode="x unified")
    st.plotly_chart(fig)

    st.markdown("---")

    st.markdown("### TREINAMENTO DOS MODELOS DE MACHINE LEARNING")
    st.markdown("O treinamento dos modelos seguiu o seguinte processo:")
    st.markdown("* Balanceamento da Variável Alvo com SMOTE: Devido ao desbalanceamento significativo da variável alvo (95% de classe 0 e 5% de classe 1), aplicamos o SMOTE (Synthetic Minority Over-sampling Technique). Esta técnica de oversampling gerou exemplos sintéticos para a classe minoritária, visando equilibrar a distribuição dos dados e melhorar o desempenho do modelo na previsão de contratações.")
    st.markdown("* Seleção e Avaliação dos Modelos: Treinamos seis modelos distintos: Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting, XGBoost e Support Vector Machine (SVC). A avaliação do desempenho dos modelos foi realizada com base em quatro métricas cruciais: Precision, Recall, F1-Score e AUC-ROC.")
    st.markdown('| Modelo | Precision | Recall | F1-Score | AUC-ROC |\n|-|-|-|-|-|\n| Regressão Logística | 0.5683 | 0.2797 | 0.3749 | 0.6766 |\n| Árvore de Decisão | 0.5673 | 0.2766 | 0.3718 | 0.6733|\n| Random Forest | 0.5669 | 0.2781 | 0.3732 | 0.6736 |\n| Gradient Boosting (GBM) | 0.5669 | 0.2781 | 0.3732| 0.6733 |\n| XGBoost | 0.5683 | 0.2797 | 0.3749 | 0.6748 |\n| Support Vector Machine (SVC) | 0.5683 | 0.2797 | 0.3749 | 0.6535 |')
    st.markdown("Como o modelo de Regressão Linear foi ligeiramente superior aos demais, iremos disponibilizá-lo na página seguinte.")

    st.markdown("---")

    st.markdown("### CONCLUSÃO")
    st.markdown("Apesar do rigoroso processo de pré-processamento e balanceamento dos dados, os modelos tradicionais de Machine Learning apresentaram uma performance extremamente similar, com variações mínimas (a partir das casas milésimais) em todas as métricas avaliadas. Concluímos, portanto, que modelos tradicionais de Machine Learning podem não ser os mais adequados para a complexidade da seleção de candidatos da Decision.")
    st.markdown("Alguns fatores que podem justificar essa performance abaixo do esperado incluem:")
    st.markdown("* Variedade de Vagas: A grande diversidade das vagas, que exigem diferentes conjuntos de competências e experiências, representa um desafio significativo para modelos de Machine Learning tradicionais.")
    st.markdown("* Variedade de Clientes: Mesmo vagas aparentemente similares (ex: Desenvolvedor Júnior) podem apresentar peculiaridades e requisitos específicos de cada cliente, o que dificulta a generalização do modelo.")
    st.markdown("* Variedade de Perfis de Candidatos: Candidatos possuem características únicas e nuances contextuais que são intrinsecamente difíceis de serem traduzidas e representadas em um formato estruturado para um modelo tradicional.")
    st.markdown("Considerando os desafios mencionados e a necessidade crítica de um modelo capaz de interpretar contextos complexos e nuances, uma solução promissora seria a utilização de modelos de Linguagem de Grande Escala (LLMs). Modelos mais avançados, que empregam técnicas como 'Chain of Thought', poderiam oferecer uma capacidade superior de compreensão e raciocínio contextual, tornando-os mais aptos a lidar com a complexidade do processo seletivo da Decision.")

# Conteúdo da página Modelo de previsão
elif pagina_selecionada == "Modelo de previsão":

    # Conteúdo do relatório
    st.markdown('# Modelo de previsão para a escolha de um candidato')
    st.markdown("---")

    # 1. Carregar o modelo, o codificador e a ordem das features
    @st.cache_resource # Usar st.cache_resource para carregar recursos pesados uma vez
    def load_assets():
        model = joblib.load('regressao_logistica_model.joblib')
        ohe_encoder = joblib.load('one_hot_encoder_modalidade.joblib')
        features_order = joblib.load('model_features_order.joblib')
        return model, ohe_encoder, features_order

    model, ohe_modalidade, feature_columns_order = load_assets()

    # 2. Definir os mapeamentos manuais (devem ser os mesmos usados no treinamento)
    ingles_mapping_streamlit = {
        'Nenhum': 1,
        'Básico': 2,
        'Intermediário': 3,
        'Avançado': 4,
        'Fluente': 5,
    }

    vaga_sap_mapping_streamlit = {
        'Sim': 1,
        'Não': 0
    }

    # 3. Criar inputs para o usuário
    st.header('Insira os dados da vaga:')

    # 3.1 Opções para Modalidade de contratação
    modalidade_options = ['CLT', 'PJ', 'Cooperado', 'CLT - Estratégico', 'Hunting']
    modalidade_input = st.selectbox('Modalidade de contratação:', modalidade_options)

    # 3.2 Opções para Nível de Inglês
    nivel_ingles_input = st.selectbox(
        'Nível de Inglês:',
        ['Nenhum', 'Básico', 'Intermediário', 'Avançado', 'Fluente']
    )

    # 3.3 Opções para Vaga SAP
    vaga_sap_input = st.selectbox('Vaga SAP?', ['Não', 'Sim'])

    # 4. Processar os inputs do usuário para o formato que o modelo espera
    if st.button('Fazer Previsão'):
        input_data_dict = {
            'modalidade': [modalidade_input],
            'nivel_ingles_x': [nivel_ingles_input],
            'vaga_sap': [vaga_sap_input]
        }
        input_df = pd.DataFrame(input_data_dict)

        # 4.1 Codificação de 'modalidade' com o OneHotEncoder salvo
        encoded_modalidade_input = ohe_modalidade.transform(input_df[['modalidade']])
        df_encoded_modalidade_input = pd.DataFrame(
            encoded_modalidade_input,
            columns=ohe_modalidade.get_feature_names_out(['modalidade']),
            index=input_df.index
        )

        # 4.2 Codificação de 'nivel_ingles_x'
        input_df['nivel_ingles_encoded'] = input_df['nivel_ingles_x'].map(ingles_mapping_streamlit)
        input_df['nivel_ingles_encoded'].fillna(0, inplace=True)

        # 4.3 Codificação de 'vaga_sap'
        input_df['vaga_sap_encoded'] = input_df['vaga_sap'].map(vaga_sap_mapping_streamlit)

        # 4.4 Montar o DataFrame final na ordem correta das colunas
        processed_input = pd.DataFrame(columns=feature_columns_order, index=input_df.index)

        # 4.5 Preencher as colunas One-Hot
        for col in ohe_modalidade.get_feature_names_out(['modalidade']):
            if col in df_encoded_modalidade_input.columns:
                processed_input[col] = df_encoded_modalidade_input[col]
            else:
                processed_input[col] = 0

        # 4.6 Preencher as colunas mapeadas manualmente
        processed_input['nivel_ingles_encoded'] = input_df['nivel_ingles_encoded']
        processed_input['vaga_sap_encoded'] = input_df['vaga_sap_encoded']

        # 4.7 Assegurar que quaisquer NaNs restantes (se o usuário não preencheu algo) sejam 0
        processed_input.fillna(0, inplace=True)

        try:
            prediction = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)[:, 1]
            st.success(f"Previsão: {'Sim' if prediction[0] == 1 else 'Não'}")
            st.write(f"Probabilidade de ser 'Sim': {prediction_proba[0]:.2f}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao fazer a previsão. Verifique o console para detalhes.")
            st.error(f"Detalhes do erro: {e}")
            st.write("Verifique se todas as colunas estão no formato correto e na ordem esperada pelo modelo.")
            st.write("Colunas esperadas pelo modelo:")
            st.code(feature_columns_order)
            st.write("Colunas do input processado:")
            st.code(processed_input.columns.tolist())
            st.write("Dados do input processado:")
            st.dataframe(processed_input)