# ------------------------------------------------------------

# *Operacional Volume Completo*

# ------------------------------------------------------------

def segundos_para_hh_mm_ss(segundos_float):
    segundos = int(segundos_float)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos = segundos % 60

    return f'{horas:02d}:{minutos:02d}:{segundos:02d}'

def resample(df, time_frame):
    ##### Filtro de Data

    #df = df.loc[(df.index.year >= min_year_selected) & (df.index.year <= max_year_selected)]

    ##### 1.2.3.1. Antes do Resample

    import random
    specific_day = random.choice(df.index.date)

    # Filter DataFrame for the specific day
    df1 = df.loc[df.index.date == specific_day]

    # Gráfico de candles antes do resample

    # Criando o gráfico de candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(name='WDO', x=df1.index,
                                 close=df1['Close'], open=df1['Open'],
                                 high=df1['High'], low=df1['Low']))

    # Configurando o layout do gráfico
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title_text='Gráfico de Candlestick 1 min WDO',
                      paper_bgcolor='black',
                      plot_bgcolor='black',
                      title_font_color='white',
                      width=800,
                      height=500,
                      font_color='white')
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    st.subheader('Resample')
        
    resample = st.expander('Antes do Resample')

    with resample:
        # Exibindo o gráfico na aplicação Streamlit
        st.plotly_chart(fig)

    # Resample

    df = df.resample(f'{str(time_frame)}T').agg({\
             'Open' : 'first',\
             'High' : 'max',\
             'Low' : 'min',\
             'Close' : 'last',\
             'Volume': 'sum'})
    df.dropna(inplace=True)

    ##### 1.2.3.2. Depois do Resample

    # # Filter DataFrame for the specific day
    df2 = df.loc[df.index.date == specific_day]

    # Gráfico de candles depois do resample

    fig = go.Figure()
    fig.add_trace(go.Candlestick(name='WDO', x=df2.index,
                                 close=df2['Close'], open=df2['Open'],
                                 high = df2['High'], low=df2['Low'] ))
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title_text=f'Gráfico de Candlestick {time_frame} min <b>WDO<b>',
                      paper_bgcolor='black',
                      plot_bgcolor='black',
                      title_font_color='white',
                      width=800,
                      height=500,
                      font_color='white')
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    resample = st.expander('Depois do Resample')

    with resample:
        # Exibindo o gráfico na aplicação Streamlit
        st.plotly_chart(fig)

    # ------------------

def etl(uploaded_file, continuar, time_frame, inicio_abertura, limite_abertura, limite_fechamento):

    ## *1. ETL*

    # -----------------------------

    ### *1.1. Extração dos dados*

    # ------------------

    #### 1.1.3. Importando Data Frame

    df = pd.read_csv(uploaded_file, sep = ';', index_col = 0)

    # -----------------------------

    ### *1.2. Transformação dos dados*

    # ------------------

    #### 1.2.1. Alterando nome das colunas

    df.rename(columns = {df.columns[0]:'Open',\
                     df.columns[1]:'High',\
                     df.columns[2]:'Low',\
                     df.columns[3]:'Close',\
                     df.columns[5]:'Volume'}, inplace= True)

    df.drop(['Data.1'], axis=1, inplace=True)

    # ------------------

    #### 1.2.2. Alterando tipo das colunas

    df.index = pd.to_datetime(df.index, format = '%d/%m/%Y %H:%M')
    #df.dropna(inplace = True)

    # List of columns to convert to float64
    floats = ['Open', 'High', 'Low', 'Close']

    # Convert decimal separator and change data type to float
    for col in floats:
        df[col] = df[col].str.replace(',', '.').astype(float)

    df.sort_index(ascending = True, inplace = True)

    # ------------------

    #### 1.2.3. Resample
    if continuar:
        resample(df, time_frame)

        #### 1.2.4. Dados de Data

        df['Ano'] = df.index.year
        df['Dia'] = df.index.date
        df['Hora'] = df.index.hour
        df['Minuto'] = df.index.minute
        df['Week Day'] = df.index.strftime('%A')

        # ------------------

        #### 1.2.5. Criando coluna que classifica Manhã ou Tarde

        df['Turno'] = np.where(df.index.hour < 12, 'Manhã', 'Tarde')

        # -----------------------------

        ### *1.3. Tempo de Execução*
        segundos_float = time.time() - start_time  # Substitua isso pelo seu valor de tempo em segundos como float
        tempo.append(segundos_para_hh_mm_ss(segundos_float))

        # Imprima o tempo de execução

        #print(tempo[0])
        
        analise(df, uploaded_file, inicio_abertura, limite_abertura, limite_fechamento)

# ------------------------------------------------------------

def analise(df, uploaded_file, inicio_abertura, limite_abertura, limite_fechamento):
    ## *2. Análise*

    start_time = time.time()

    # -----------------------------

    ### *2.1. Definindo propriedades do ativo*

    # ------------------

    #### 2.1.1. Ativo

    # Usando slice para obter os 3 primeiros caracteres
    tres_caracteres_apos_ultima_barra = uploaded_file.name[0:3]

    # ------------------

    #### 2.1.1. Tick do Ativo

    tick_wdo = 0.5
    tick_win = 5
    if tres_caracteres_apos_ultima_barra == 'WDO':
        tick = tick_wdo
    elif tres_caracteres_apos_ultima_barra == 'WIN':
        tick = tick_win
    else:
        print('Dados de ativo não registrados, escolhendo tick de uma ação (0.01)')
        tick = 0.01

    # ------------------    

    #### 2.1.2. Custos do Ativo

    ##### 2.1.2.1. Custos por Contrato (R$)

    custo_wdo = 1.1
    custo_win = 0.4
    if tres_caracteres_apos_ultima_barra == 'WDO':
        custo_por_contrato = custo_wdo
    elif tres_caracteres_apos_ultima_barra == 'WIN':
        custo_por_contrato = custo_win
    else:
        print('Dados de ativo não registrados, desconsiderando custos do ativo')
        custo_por_contrato = 0

    ##### 2.1.2.2. Custos por Ponto por Contrato (R$)

    custo_wdo_pt = 10
    custo_win_pt = 0.25
    if tres_caracteres_apos_ultima_barra == 'WDO':
        custo_por_ponto = custo_wdo_pt
    elif tres_caracteres_apos_ultima_barra == 'WIN':
        custo_por_ponto = custo_win_pt
    else:
        print('Dados de ativo não registrados')
        custo_por_ponto = None

    # -----------------------------

    ### *2.2. Definindo horários limites*

    # ------------------



    # -----------------------------

    ### *2.3. Definindo Classes*

    # ------------------

    #### 2.3.1. Classificando Triggers

    psv_c = ((df['Open'] > df['Close']) & (df['Volume'] < df['Volume'].shift())) #compra

    psv_v = ((df['Open'] < df['Close']) & (df['Volume'] < df['Volume'].shift())) #venda


    defesa_c = ((df['Open'] < df['Close']) & (df['Open'].shift() >= df['Close'].shift()) &\
         ((df['Close'] - df['Low'] > (df['High'] - df['Low']) * 2 / 3)) & (df['Volume'] > df['Volume'].shift())) #compra

    defesa_v = ((df['Open'] > df['Close']) & (df['Open'].shift() <= df['Close'].shift()) &\
         (df['High'] - df['Close'] > (df['High'] - df['Low']) * 2 / 3) & (df['Volume'] > df['Volume'].shift()))   #venda


    ler_c = ((df['Open'] > df['Close']) & (df['High'] - df['Close'] < (df['High'] - df['Low']) * 0.5) &\
         (df['Close']) & (df['Volume'] > df['Volume'].shift())) #compra

    ler_v = ((df['Open'] < df['Close']) & (df['Close'] - df['Low'] < (df['High'] - df['Low']) * 0.5) &\
         (df['Volume'] > df['Volume'].shift())) #venda


    df['Trigger'] = ''
    df.loc[psv_c | psv_v, 'Trigger'] = 'PSV'
    df.loc[defesa_c | defesa_v, 'Trigger'] = 'Defesa'
    df.loc[ler_c | ler_v, 'Trigger'] = 'LER'

    reset_condition = (df['Dia'] != df['Dia'].shift())
    df.loc[reset_condition, 'Trigger'] = ''

    # ------------------

    #### 2.3.2. Compra ou Venda

    compra = psv_c | defesa_c | ler_c

    venda = psv_v | defesa_v | ler_v

    df['C/V'] = ''
    df.loc[compra, 'C/V'] = 'Compra'
    df.loc[venda, 'C/V'] = 'Venda'

    df.loc[reset_condition, 'C/V'] = ''

    # ------------------

    #### 2.3.3. Classificando Volume

    #Média do Volume do Dia
    df['Avg_Volume_Day'] = df.groupby('Dia')['Volume'].transform(lambda x: x.expanding().mean())

    hora_volume = '10:30'
    h_vol = int(hora_volume[:2])
    m_vol = int(hora_volume[3:])

    # Condition based on time
    time_condition = (df['Hora'] < h_vol) & (df['Minuto'] < m_vol)

    # Condition for 'Trigger' == 'PSV'
    trigger_condition_psv = (df['Trigger'] == 'PSV')

    # Condition for 'Trigger' == 'Defesa'
    trigger_condition_defesa = (df['Trigger'] == 'Defesa')

    # Condition for 'Trigger' == 'LER'
    trigger_condition_ler = (df['Trigger'] == 'LER')

    # Calculate the expanding maximum and minimum volume for each day
    expanding_max_volume = df.groupby('Dia')['Volume'].transform(lambda x: x.expanding().max())
    expanding_min_volume = df.groupby('Dia')['Volume'].transform(lambda x: x.expanding().min())

    # Condition for 'Volume' comparison - PSV
    if time_condition.all():  # Check if the time condition holds for all rows
        # Compare 'Volume' with 50% of the expanding maximum volume for the day
        volume_condition_psv = df['Volume'] < expanding_max_volume * 0.7
    else:
        # Compare 'Volume' with 50% of the 'Avg_Volume_Day'
        volume_condition_psv = df['Volume'] < df['Avg_Volume_Day']

    # Condition for 'Volume' comparison - Defesa
    if time_condition.all():  # Check if the time condition holds for all rows
        # Compare 'Volume' with 50% of the expanding maximum volume for the day
        volume_condition_defesa = df['Volume'] > expanding_min_volume / 0.7
    else:
        # Compare 'Volume' with 50% of the 'Avg_Volume_Day'
        volume_condition_defesa = df['Volume'] > df['Avg_Volume_Day']

    # Combine conditions using '&' (AND) operator
    v_psv = trigger_condition_psv & volume_condition_psv
    v_defesa = trigger_condition_defesa & volume_condition_defesa
    v_ler = trigger_condition_ler & volume_condition_defesa

    df['V'] = 0
    df.loc[v_psv, 'V'] = 1
    df.loc[v_defesa, 'V'] = 1
    df.loc[v_ler, 'V'] = 1

    # ------------------

    #### 2.3.4. Calculando Fibo

    # Extract unique days from the DataFrame
    dias = df['Dia'].unique()
    # Calculate Fibonacci levels for each unique day
    for i in range(1, len(dias)):
        previous_day = dias[i - 1]

        # Extract data for the previous day
        previous_day_data = df[df['Dia'] == previous_day]

        previous_dia_max_high = previous_day_data['High'].max()
        previous_dia_min_low = previous_day_data['Low'].min()

        previous_dia_amplitude = previous_dia_max_high - previous_dia_min_low

        # Define Fibonacci levels
        fibonacci_levels = [-0.618, -0.5, -0.236, 0, 0.382, 0.618, 1, 1.236, 1.5, 1.618]

        # Calculate Fibonacci levels based on the previous 'Dia' amplitude
        for level in fibonacci_levels:
            df.loc[df['Dia'] == dias[i], f'Fibo_{level}'] = (previous_dia_amplitude * level) + previous_dia_min_low

    df.dropna(inplace=True)

    fibo = [f'Fibo_{level}' for level in fibonacci_levels]

    # ------------------

    #### 2.3.5. Classificando Região

    df['Region'] = 0
    df['Qual_Fibo1'] = ''
    df['Qual_Fibo2'] = ''
    for index, row in df.iterrows():
        operation = row['C/V']
        high_or_low = 'High' if operation == 'Compra' else 'Low'

        for i, fibo_col in enumerate(fibo[:-1]):  # Exclude the last Fibonacci column
            next_fibo_col = fibo[i + 1]
            if ((operation == 'Compra' and row[high_or_low] >= row[fibo_col] and row[high_or_low] < row[next_fibo_col]) or
                (operation == 'Venda' and row[high_or_low] > row[fibo_col] and row[high_or_low] <= row[next_fibo_col])):
                df.at[index, 'Region'] = i + 1
                df.at[index, 'Qual_Fibo1'] = fibo[i]
                df.at[index, 'Qual_Fibo2'] = fibo[i+1]
                break

    # ------------------

    #### 2.3.6. Classificando Contexto

    df['C'] = 0

    valid_columns1 = df.loc[df['Qual_Fibo1'] != '', 'Qual_Fibo1'].tolist()
    valid_columns2 = df.loc[df['Qual_Fibo2'] != '', 'Qual_Fibo2'].tolist()

    buy_conditions = (
        (df['C/V'] == 'Compra') &
        (
            (df['Low'] <= df.apply(lambda row: row[fibo[row['Region']-1]], axis=1)) |
            (df['Low'].shift(1) <= df.apply(lambda row: row[fibo[row['Region']-1]], axis=1))
        ) &
        (df['Close'] >= df.apply(lambda row: row[fibo[row['Region']-1]], axis=1))
    )

    sell_conditions = (
        (df['C/V'] == 'Venda') &
        (
            (df['High'] >= df.apply(lambda row: row[fibo[row['Region']]], axis=1)) |
            (df['High'].shift(1) >= df.apply(lambda row: row[fibo[row['Region']]], axis=1))
        ) &
        (df['Close'] <= df.apply(lambda row: row[fibo[row['Region']]], axis=1))
    )


    contexto = buy_conditions | sell_conditions

    df.loc[contexto, 'C'] = 1

    # ------------------

    #### 2.3.7. Classificando Risco:Retorno "Payoff"

    df['P'] = 0

    buy_conditions = (
        (df['C/V'] == 'Compra') &
        ((df['High'] - df['Low']) + 2*tick <= df.apply(lambda row: row[fibo[row['Region']]], axis=1) - df['High'] + tick)
    )

    sell_conditions = (
        (df['C/V'] == 'Venda') &
        ((df['High'] - df['Low']) + 2*tick <= df['Low'] - tick - df.apply(lambda row: row[fibo[row['Region']-1]], axis=1))
    )

    contexto = buy_conditions | sell_conditions

    df.loc[contexto, 'P'] = 1

    # ------------------

    #### 2.3.8. Calculando Estocástico (falta ajustar para o Lento)

    from ta.momentum import StochasticOscillator
    from ta.trend import SMAIndicator
    stoch = StochasticOscillator(close = df['Close'], high = df['High'], low = df['Low'], window = 9, smooth_window = 3)
    df['Stoch_i'] = stoch.stoch()
    mm_stoch = SMAIndicator(close = df['Stoch_i'], window = 3)
    df['MM_Stoch'] = mm_stoch.sma_indicator()

    df['Stoch'] = 0

    buy_conditions = (
        (df['C/V'] == 'Compra') &
        (df['Stoch_i'] >= df['MM_Stoch'])
    )

    sell_conditions = (
        (df['C/V'] == 'Venda') &
        (df['Stoch_i'] <= df['MM_Stoch'])
    )

    contexto = buy_conditions | sell_conditions

    df.loc[contexto, 'Stoch'] = 1

    # # Filter DataFrame for the specific day
    # df2 = df.loc[df.index.date == specific_day]
    #
    # fig = make_subplots(rows = 2, cols = 1)
    # fig.add_trace(go.Candlestick(name='WDO', x=df2.index,
    #                              close=df2['Close'], open=df2['Open'],
    #                              high = df2['High'], low=df2['Low']),
    #                              row = 1, col = 1)
    # fig.add_trace(go.Scatter(name='Stoch', x=df2.index, y = df2['Stoch_i']),
    #                              row = 2, col = 1)
    # fig.add_trace(go.Scatter(name='MM3', x=df2.index, y = df2['MM_Stoch']),
    #                              row = 2, col = 1)
    #
    # fig.update_layout(xaxis_rangeslider_visible=False,
    #                   title_text=f'Gráfico de {time_frame} min <b>WDO<b>',
    #                   paper_bgcolor='black',
    #                   plot_bgcolor='black',
    #                   title_font_color='white',
    #                   width=1200,
    #                   height=500,
    #                   font_color='white')
    # fig.update_xaxes(showgrid=False, zeroline=False)
    # fig.update_yaxes(showgrid=False, zeroline=False)
    # fig.show()

    # ------------------

    #### 2.3.9. Criando IDs com Classes

    df['VCP_ID'] = df['V'].astype(str) + df['C'].astype(str) + df['P'].astype(str)
    df['VCP_Stoch_ID'] = df['V'].astype(str) + df['C'].astype(str) + df['P'].astype(str) + df['Stoch'].astype(str)
    df['ID'] = df['Trigger'] + df['VCP_Stoch_ID']

    # ------------------

    #### 2.3.10. Criando Scores com Soma de Classes

    df['VCP'] = df['V'] + df['C'] + df['P']
    df['VCP_Stoch'] = df['V'] + df['C'] + df['P'] + df['Stoch']

    # -----------------------------

    ### *2.4. Verificando se o trade ativou*

    #linhas para avaliar
    linhas = len(df)

    #Shift Columns OHLC
    columns_to_shift = ['Open', 'High', 'Low', 'Close']

    df[['Open1', 'High1', 'Low1', 'Close1']] = df[columns_to_shift].shift(-1)

    # Where

    df['Ativou'] = np.where(
        (df['C/V'] == 'Compra') & (df['High1'] > df['High']) &
        (df.index.hour >= inicio_abertura) & (df.index.hour < limite_abertura),
        np.where(
            df['Low1'] < df['Low'],
            np.where(
                df['Close1'] > df['Open1'],
                'Ativou e nao stopou',
                'Ativou e stopou'
            ),
            'Ativou'
        ),
        np.where(
            (df['C/V'] == 'Venda') & (df['Low1'] < df['Low']) &
            (df.index.hour >= inicio_abertura) & (df.index.hour < limite_abertura),
            np.where(
                df['High1'] > df['High'],
                np.where(
                    df['Close1'] < df['Open1'],
                    'Ativou e nao stopou',
                    'Ativou e stopou'
                ),
                'Ativou'
            ),
            ''
        )
    )

    # ------------------

    #### 2.4.1. Ativou e Stopou / Ativou [%]
    
    st.markdown('---')
    ativou = st.expander('Ativou e Stopou / Ativou [%]')
    
    with ativou:
        st.write(str(round(len(df[df['Ativou'] == 'Ativou e stopou'])*100/len(df[df['Ativou'] != '']),2)) + '%')


    # -----------------------------

    ### *2.5. Máxima Exposição Positiva - MEP e Custos Operacionais*

    # ------------------

    #### 2.5.1. Ajustando índice de df

    df['Data'] = df.index
    df['Hora'] = df.index.hour
    df.reset_index(drop=True,inplace=True)
    df.sort_values('Data',inplace = True, ascending = True)

    # ------------------

    #### 2.5.2. Calculando MEP

    def calculate_var_gain(row):
        if row['Ativou'] == 'Ativou':
            if row['C/V'] == 'Compra':
                j = row.name + 1
                amplitude = row['High'] - row['Low']
                entrada = row['High'] + tick
                max_high = 0
                while (j < len(df)) and not ((df.at[j, 'Low'] <= row['Low'] - tick) or (df.Hora[j] >= limite_fechamento)):
                    if df.at[j, 'High'] > max_high:
                        max_high = df.at[j, 'High']
                        var_gain = (max_high - entrada) / (amplitude + (2 * tick)) if (amplitude + (2 * tick)) != 0 else 0
                    j += 1
                return var_gain if 'var_gain' in locals() else 0
            elif row['C/V'] == 'Venda':
                j = row.name + 1
                amplitude = row['High'] - row['Low']
                entrada = row['Low'] - tick
                min_low = float('inf')
                while (j < len(df)) and not ((df.at[j, 'High'] >= row['High'] + tick) or (df.Hora[j] >= limite_fechamento)):
                    if df.at[j, 'Low'] < min_low:
                        min_low = df.at[j, 'Low']
                        var_gain = (entrada - min_low) / (amplitude + (2 * tick)) if (amplitude + (2 * tick)) != 0 else 0
                    j += 1
                return var_gain if 'var_gain' in locals() else 0
            else:
                return 0
        elif row['Ativou'] == 'Ativou e stopou':
            if row['C/V'] == 'Compra':
                j = row.name + 1
                amplitude = row['High'] - row['Low']
                entrada = row['High'] + tick
                max_high = df.at[j, 'High']
                var_gain = (max_high - entrada) / (amplitude + (2 * tick)) if (amplitude + (2 * tick)) != 0 else 0
                return var_gain if 'var_gain' in locals() else 0
            elif row['C/V'] == 'Venda':
                j = row.name + 1
                amplitude = row['High'] - row['Low']
                entrada = row['Low'] - tick
                min_low = df.at[j, 'Low']
                var_gain = (entrada - min_low) / (amplitude + (2 * tick)) if (amplitude + (2 * tick)) != 0 else 0
                return var_gain if 'var_gain' in locals() else 0

        elif row['Ativou'] == 'Ativou e nao stopou':
            if row['C/V'] == 'Compra':
                j = row.name + 2
                i = j - 1
                amplitude = row['High'] - df.at[i, 'Low'] if (j + 1 < len(df)) else 0
                entrada = row['High'] + tick if (j + 1 < len(df)) else 0
                max_high = 0
                while (j < len(df)) and not ((df.at[j, 'Low'] <= row['Low'] - tick) or (df.Hora[j] >= limite_fechamento)):
                    if df.at[j, 'High'] > max_high:
                        max_high = df.at[j, 'High']
                        var_gain = (max_high - entrada) / (amplitude + (2 * tick)) if (amplitude + (2 * tick)) != 0 else 0
                    j += 1
                return var_gain if 'var_gain' in locals() else 0
            elif row['C/V'] == 'Venda':
                j = row.name + 2
                i = j - 1
                amplitude = df.at[i, 'High'] - row['Low'] if (j + 1 < len(df)) else 0
                entrada = row['Low'] - tick if (j + 1 < len(df)) else 0
                min_low = float('inf')
                while (j < len(df)) and not ((df.at[j, 'High'] >= row['High'] + tick) or (df.Hora[j] >= limite_fechamento)):
                    if df.at[j, 'Low'] < min_low:
                        min_low = df.at[j, 'Low']
                        var_gain = (entrada - min_low) / (amplitude + (2 * tick)) if (amplitude + (2 * tick)) != 0 else 0
                    j += 1
                return var_gain if 'var_gain' in locals() else 0
            else:
                return 0
        else:
            return 0

    # Apply the function to create the 'MEP' column
    df['MEP'] = df.apply(calculate_var_gain, axis=1)

    df.index = df['Data']
    df['Data'] = df.index

    # ------------------

    #### 2.5.3. Criando novo DF com apenas trades que ativaram (df_ativou)

    df_ativou = df[df['Ativou'] != ''].copy()

    # ------------------

    #### 2.5.4. Custos

    ##### 2.5.4.1. Amplitude da entrada (risco)

    df_ativou['Risco_pts'] = np.where(
        (df_ativou['Ativou'] == 'Ativou') | (df_ativou['Ativou'] == 'Ativou e stopou'),
        df_ativou['High'] - df_ativou['Low'] + 2 * tick,
        np.where(
            (df_ativou['Ativou'] == 'Ativou e nao stopou') & (df_ativou['C/V'] == 'Compra'),
            df_ativou['High'] - df_ativou['Low1'] + 2 * tick,
            np.where(
                (df_ativou['Ativou'] == 'Ativou e nao stopou') & (df_ativou['C/V'] == 'Venda'),
                df_ativou['High1'] - df_ativou['Low'] + 2 * tick,
                0
            )
        )
    )

    ##### 2.5.4.2. Número de Contratos para um Risco de R$1000,00

    df_ativou['Contratos'] = 1000//(df_ativou['Risco_pts']*custo_por_ponto)

    ##### 2.5.4.3. Custo para um Risco de R$1000,00

    df_ativou['Custo'] = df_ativou['Contratos']*custo_por_contrato*2 # Nos custos, considera-se a abertura e fechamento da operação. Logo, *2.

    # ------------------

    #### 2.5.4. Funções personalizadas para cálculo geral

    def q1(x):
        return x.quantile(0.25)
    def q3(x):
        return x.quantile(0.75)
    def trades(x):
        return x.count()
    #resultado total dividido pelo número de trades
    def saldo_trades(x):
        return x.sum()/x.count()

    # ------------------

    #### 2.5.5. Groupby com IDs

    df_GB = df_ativou.groupby('ID')
    column = df_GB['MEP']
    GB = round(column.agg([np.mean, np.std, q1, np.median, q3, np.min, np.max, trades,saldo_trades]),2)
    #GB[GB['trades']>30].sort_values('saldo_trades',ascending = False).head(15)

    # ------------------

    #### 2.5.6. Concatenando df_ativou com alvos

    df_ativou.reset_index(drop=True,inplace=True)

    op = []
    columns = []

    for j in range(2, 52, 2):
        j /= 10.0
        op_aux = []
        columns.append(str(round(j,1)))
        for i in range(0,len(df_ativou)):
            if (df_ativou.MEP[i]) >= j:
                op_aux.append(j)

            else: 
                op_aux.append(-1)

        op.append(op_aux)


    alvos = pd.DataFrame(np.transpose(op),columns=columns)
    results = pd.concat([df_ativou,alvos],axis=1,copy=True)

    # -----------------------------

    ### *2.6. Adicionando Colunas com Resultado Acumulado, Custos e Drawdown por ID*

    # ------------------

    #### 2.6.1. Adicionando colunas com resultado acumulado em cada alvo

    acumulado = []

    for j in range(0,len(columns)):
        results['Acum_' + columns[j]] = results.groupby(['ID'])[columns[j]].cumsum()
        acumulado.append('Acum_' + columns[j])

    # ------------------

    #### 2.6.2. Adicionando colunas com picos acumulados em cada alvo

    picos = []

    for j in range(0,len(columns)):
        results['Picos_' + columns[j]] = results.groupby(['ID'])['Acum_' + columns[j]].cummax()
        picos.append('Picos_' + columns[j])

    # ------------------

    #### 2.6.3. Adicionando colunas com drawdown acumulado em cada alvo

    dd = []
    for j in range(0,len(columns)):
        results['DD_' + columns[j]] = results['Acum_' + columns[j]] - results['Picos_' + columns[j]]
        dd.append('DD_' + columns[j])

    results.drop(picos, axis = 1, inplace = True)
    results.drop(['Open', 'High', 'Low', 'Close'], axis = 1, inplace = True)

    # ------------------

    #### 2.6.4. Adicionando colunas com variação para quartis de resultado acumulado

    periodo = 15

    acum_q = []
    for j in range(0,len(columns)):
        results['Acum_q' + columns[j]] = results.groupby('ID')['Acum_' + columns[j]].diff(periods=periodo)
        acum_q.append('Acum_q' + columns[j])

    # ------------------

    #### 2.6.5. Adicionando colunas com variação para quartis de drawdown

    dd_q = []
    for j in range(0,len(columns)):
        results['DD_q' + columns[j]] = results.groupby('ID')['DD_' + columns[j]].diff(periods=periodo)
        dd_q.append('DD_q' + columns[j])

    # -----------------------------

    ### *2.7. Criando tabelas por ID*

    # ------------------

    #### 2.7.1. Tabela Saldo Trades

    tabela_saldo_trades = pd.pivot_table(results, index = 'ID',values = columns, aggfunc = [saldo_trades])
    tabela_saldo_trades.columns = tabela_saldo_trades.columns.droplevel()

    ##### 2.7.1.1. Adicionando colunas Max_saldo_trades e Alvo

    tabela_saldo_trades['Max_saldo_trades'] = tabela_saldo_trades.max(axis=1)
    column_names = list(tabela_saldo_trades.columns.values)


    tabela_saldo_trades['Alvo_saldo'] = tabela_saldo_trades.idxmax(axis = 1)
    tabela_saldo_trades = tabela_saldo_trades.sort_values('Max_saldo_trades',ascending=False)

    # ------------------

    #### 2.7.2. Tabela Razão Drawdown

    ##### 2.7.2.1. Tabelão Resultado Acumulado + Drawdown

    tabela_acumulado = pd.pivot_table(results, index = 'ID',values = acumulado, aggfunc = 'max')
    #tabela_acumulado.columns = tabela_acumulado.columns.droplevel()

    tabela_dd = pd.pivot_table(results, index = 'ID',values = dd, aggfunc = 'min')
    #tabela_dd.columns = tabela_dd.columns.droplevel()

    tabelao = pd.concat([tabela_acumulado,tabela_dd],axis=1,copy=True)
    trades = column.agg(['count'])
    trades.rename(columns={"count": "Trades"}, inplace = True)
    tabelao = pd.merge(tabelao,  trades,\
              how = 'left', left_index = True, right_index = True)
    trades.sort_values('Trades',ascending=False)

    ##### 2.7.2.2. Adicionando colunas Razão Drawdown

    razao_dd = []
    for i in range(0,len(columns)):
        tabelao['Razao_DD_' + columns[i] + ' '] = (-tabelao['DD_' + columns[i]])/tabelao['Acum_' + columns[i]]
        razao_dd.append('Razao_DD_' + columns[i] + ' ')

    tabelao.drop(acumulado, axis=1,inplace = True)
    tabelao.drop(dd, axis=1,inplace = True)

    ###### 2.7.2.2.1. Visualizando Razão DD por ID

    # plt.figure(figsize = (20,8))
    #
    # sns.heatmap(tabelao,
    #             vmin = -1, vmax = 1, annot = True, cmap = 'BrBG'
    # )

    ##### 2.7.2.3. Removendo valores incoerentes

    razao_dd = tabelao._get_numeric_data()

    razao_dd[(razao_dd <= 0) | (razao_dd > 100000)] = 1

    ###### 2.7.2.3.1. Visualizando Razão DD por ID

    # plt.figure(figsize = (20,8))
    #
    # sns.heatmap(tabelao,
    #             vmin = -1, vmax = 1, annot = True, cmap = 'BrBG'
    # )

    ##### 2.7.3.1. Adicionando colunas Min_Razao_DD e Alvo

    razao_dd['Min_Razao_DD'] = razao_dd.min(axis=1)
    column_names = list(tabela_saldo_trades.columns.values)

    razao_dd['Min_Razao_DD'] = razao_dd.min(axis=1)
    razao_dd['Razao_DD'] = razao_dd.idxmin(axis=1)
    razao_dd['Alvo_rdd'] = razao_dd['Razao_DD'].str.slice(-4, -1)
    razao_dd.drop(columns=razao_dd.columns.difference(['Min_Razao_DD', 'Alvo_rdd', 'Trades']), inplace=True)
    razao_dd = razao_dd.sort_values('Min_Razao_DD', ascending=True)

    ##### 2.7.3.1.1. Visualizando Min_Razao_DD e Alvo por ID

    #print(razao_dd.style.format({'Min_Razao_DD': '{:,.2%}'}))

    # ------------------

    #### 2.7.3. Tabela Quartis

    tabela_acum_q = pd.pivot_table(results, index = 'ID',values = acum_q, aggfunc=lambda x: x.quantile(0.25))
    tabela_dd_q = pd.pivot_table(results, index = 'ID',values = dd_q, aggfunc=lambda x: x.quantile(0.25))

    quartis = pd.concat([tabela_acum_q,tabela_dd_q],axis=1,copy=True)

    for column_name in columns:
        # Calculando a razão entre Acum_30d e DD_30d para cada coluna em columns
        quartis['R_Quartil_' + column_name + ' '] = (-quartis['DD_q' + column_name]) / quartis['Acum_q' + column_name]

    quartis.drop(acum_q, axis=1,inplace = True)
    quartis.drop(dd_q, axis=1,inplace = True)

    ###### 2.7.3.1. Visualizando tabela Quartis

    # plt.figure(figsize = (20,8))
    #
    # sns.heatmap(quartis,
    #             vmin = -1, vmax = 1, annot = True, cmap = 'BrBG'
    # )

    ##### 2.7.3.2. Removendo valores incoerentes

    quartis[(quartis <= 0) | (quartis > 100000)] = 1

    ##### 2.7.3.3. Calculando Resultados mínimos para Alvos

    quartis['Min_R_Quartil'] = quartis.min(axis=1)
    quartis['R_Quartil'] = quartis.idxmin(axis=1)
    quartis['Alvo_rq'] = quartis['R_Quartil'].str.slice(-4, -1)
    quartis = pd.merge(quartis,  trades,\
              how = 'left', left_index = True, right_index = True)

    quartis.drop(columns=quartis.columns.difference(['Min_R_Quartil', 'Alvo_rq', 'Trades']), inplace=True)
    quartis = quartis.sort_values('Min_R_Quartil', ascending=True)

    ##### 2.7.3.1.1. Visualizando Min_R_Quartil e Alvo por ID

    #print(quartis.style.format({'Min_R_Quartil': '{:,.2%}'}))

    # -----------------------------

    ### 2.8. *Retornando melhores alvos para o DF "results"*

    # ------------------

    #### 2.8.1. Por Critério Saldo Trades

    results.index = results['ID']

    columns.append('Trigger')
    columns.append('Data')
    columns.append('Alvo_saldo')
    columns.append('Max_saldo_trades')

    results = pd.merge(results,  tabela_saldo_trades[['Alvo_saldo','Max_saldo_trades']],\
              how = 'right', left_index = True, right_index = True)[columns]

    # ------------------

    #### 2.8.2. Por Critério Razão Drawdown

    columns.append('Trades')
    columns.append('Alvo_rdd')
    columns.append('Min_Razao_DD')
    results = pd.merge(results,  razao_dd[['Trades','Alvo_rdd','Min_Razao_DD']],\
              how = 'right', left_index = True, right_index = True)[columns]

    # ------------------

    #### 2.8.3. Por Critério Razão Quartil

    columns.append('Alvo_rq')
    columns.append('Min_R_Quartil')
    results = pd.merge(results,  quartis[['Alvo_rq','Min_R_Quartil']],\
              how = 'right', left_index = True, right_index = True)[columns]

    # -----------------------------

    ### 2.9. *Calculando resultado para melhores alvos em DF "results"*

    results['ID'] = results.index
    results.index = results.Data
    results = results.sort_index(ascending=True)

    # ------------------

    #### 2.9.1. Por Critério Saldo Trades

    def obter_valor(row):
        # Verifica se o valor em 'Alvo' é uma coluna válida no DataFrame
        alvo = row['Alvo_saldo']
        if alvo in row.index:
            return row[alvo]
        else:
            return None  # Ou qualquer valor padrão que você deseje atribuir quando 'Alvo' não é uma coluna válida

    # Crie uma nova coluna 'Result_Min_DD' aplicando a função à linha
    results['Result_Saldo'] = results.apply(obter_valor, axis=1)

    # ------------------

    #### 2.9.2. Por Critério Razão Drawdown

    def obter_valor(row):
        # Verifica se o valor em 'Alvo' é uma coluna válida no DataFrame
        alvo = row['Alvo_rdd']
        if alvo in row.index:
            return row[alvo]
        else:
            return None  # Ou qualquer valor padrão que você deseje atribuir quando 'Alvo' não é uma coluna válida

    # Crie uma nova coluna 'Result_Min_DD' aplicando a função à linha
    results['Result_Min_DD'] = results.apply(obter_valor, axis=1)

    # ------------------

    #### 2.9.3. Por Critério Razão Quartil

    def obter_valor(row):
        # Verifica se o valor em 'Alvo' é uma coluna válida no DataFrame
        alvo = row['Alvo_rq']
        if alvo in row.index:
            return row[alvo]
        else:
            return None  # Ou qualquer valor padrão que você deseje atribuir quando 'Alvo' não é uma coluna válida

    # Crie uma nova coluna 'Result_Min_DD' aplicando a função à linha
    results['Result_Min_Quartil'] = results.apply(obter_valor, axis=1)

    # -----------------------------

    ### 2.10. *Adicionando Custos*

    # ------------------

    #### 2.10.1. *Buscando Coluna 'Custos' de df_ativou*

    df_ativou.index = df_ativou.Data
    columns.append('Result_Saldo')
    columns.append('Result_Min_DD')
    columns.append('Result_Min_Quartil')
    columns.append('Custo')
    columns.append('ID')
    results = pd.merge(results,  df_ativou['Custo'],\
              how = 'right', left_index = True, right_index = True)[columns]

    # ------------------

    #### 2.10.2. *Adicionando Custos aos Resultados*

    results['Result_Saldo'] = results['Result_Saldo'] - (results['Custo']/1000)
    results['Result_Min_DD'] = results['Result_Min_DD'] - (results['Custo']/1000)
    results['Result_Min_Quartil'] = results['Result_Min_Quartil'] - (results['Custo']/1000)

    # -----------------------------

    ### 2.11. *Tempo de Execução*

    segundos_float = time.time() - start_time  # Substitua isso pelo seu valor de tempo em segundos como float
    tempo.append(segundos_para_hh_mm_ss(segundos_float))

    print(tempo[1])

# ------------------------------------------------------------

def resultados():

    ## *3. Resultados*

    # -----------------------------

    ### *3.1. Testes de Média*

    #/////////////////////////

    #/////////////////////////

    # -----------------------------

    ### *3.2. Saldo Trades*

    # Criar o vetor com arredondamento
    vetor_Max_saldo_trades = np.arange(0.02, 0.42, 0.02)
    vetor_Max_saldo_trades = np.round(vetor_Max_saldo_trades, 2).tolist()

    # ------------------

    #### 3.2.1. Criando Dicionários com Resultados e Drawdown para Vetor de Max_saldo_trades

    res_saldo = {}  # Dicionário para armazenar os resultados
    picos_saldo = {}  # Dicionário para armazenar os picos de saldo
    dd_saldo = {}  # Dicionário para armazenar os drawdowns de saldo
    max_res_saldo = {}  # Dicionário para armazenar o máximo de resultados de saldo
    max_dd_saldo = {}  # Dicionário para armazenar o máximo de drawdown
    cont_operacoes = {} # Dicionário para armazenar a contagem de operações
    saldo_saldo = {} # Dicionário para armazenar o saldo de saldo
    r_dd_saldo = {}  # Dicionário para armazenar a relação de drawdown de saldo

    res_saldo_periodos = {} # Dicionário para armazenar os resultados avaliados no período selecionado
    dd_saldo_periodos = {} # Dicionário para armazenar os drawdowns avaliados no período selecionado

    quartil_res_saldo = {} # Dicionário para armazenar o primeiro quartil dos resultados
    quartil_dd_saldo = {} # Dicionário para armazenar o primeiro quartil dos drawdowns

    r_quartil_saldo = {} # Dicionário para armazenar a razão quartil de saldo

    for i, valor in enumerate(vetor_Max_saldo_trades):
        valor_arredondado = round(valor, 2)
        res_saldo[valor_arredondado] = results['Result_Saldo'].loc[
            (results['Max_saldo_trades'] >= valor) & (results['Trades'] >= 50)
        ].cumsum()

        picos_saldo[valor_arredondado] = res_saldo[valor_arredondado].cummax()
        dd_saldo[valor_arredondado] = res_saldo[valor_arredondado] - picos_saldo[valor_arredondado]

    # ------------------

    #### 3.2.2. Por Critério Saldo Trades

    for j, valor in enumerate(vetor_Max_saldo_trades):
        valor_arredondado = round(valor, 2)

        max_res_saldo[valor_arredondado] = res_saldo[valor_arredondado].max()
        cont_operacoes[valor_arredondado] = res_saldo[valor_arredondado].count()

        # Avoid division by zero
        if (cont_operacoes[valor_arredondado] != 0).any():
            saldo_saldo[valor_arredondado] = abs(max_res_saldo[valor_arredondado] / cont_operacoes[valor_arredondado])
        else:
            saldo_saldo[valor_arredondado] = 0

    # Encontre a chave com o maior valor
    chave_maior_valor_saldo_saldo = max(saldo_saldo, key=saldo_saldo.get)

    # ------------------

    #### 3.2.3. Por Critério Razão Drawdown

    for j, valor in enumerate(vetor_Max_saldo_trades):
        valor_arredondado = round(valor, 2)


        max_dd_saldo[valor_arredondado] = dd_saldo[valor_arredondado].min()

        # Avoid division by zero
        if (max_res_saldo[valor_arredondado] != 0):
            r_dd_saldo[valor_arredondado] = abs(max_dd_saldo[valor_arredondado] / max_res_saldo[valor_arredondado])
        else:
            r_dd_saldo[valor_arredondado] = 0

    # Encontre a chave com o menor valor
    chave_menor_valor_r_dd_saldo = min(r_dd_saldo, key=r_dd_saldo.get)

    # ------------------

    #### 3.2.4. Por Critério Razão Quartil

    for j, valor in enumerate(vetor_Max_saldo_trades):
        valor_arredondado = round(valor, 2)

        res_saldo_periodos[valor_arredondado] = res_saldo[valor_arredondado].diff(periods = periodo)
        dd_saldo_periodos[valor_arredondado] = dd_saldo[valor_arredondado].diff(periods = periodo)

        quartil_res_saldo[valor_arredondado] = res_saldo_periodos[valor_arredondado].quantile(0.25)
        quartil_dd_saldo[valor_arredondado] = dd_saldo_periodos[valor_arredondado].quantile(0.25)

        if quartil_res_saldo[valor_arredondado] > 0:
            r_quartil_saldo[valor_arredondado] = (-quartil_dd_saldo[valor_arredondado]) / quartil_res_saldo[valor_arredondado]
        else:
            r_quartil_saldo[valor_arredondado] = 1

    # Encontre a chave com o menor valor
    chave_menor_valor_r_q_saldo = min(r_quartil_saldo, key=r_quartil_saldo.get)

    # ------------------

    #### 3.2.5. Comparação

    fig = plt.figure(figsize = (10,5))
    gs = fig.add_gridspec(nrows = 10, ncols = 1)

    ax1 = fig.add_subplot(gs[0:5, 0])
    ax2 = fig.add_subplot(gs[5:11, 0])

    # Plots
    line1, = ax1.plot(res_saldo[chave_maior_valor_saldo_saldo],color = 'blue', label='Saldo Trades')
    line2, = ax1.plot(res_saldo[chave_menor_valor_r_dd_saldo], color='red', label='R_DD Saldo')
    line3, = ax1.plot(res_saldo[chave_menor_valor_r_q_saldo], color='green', label='R_Q Saldo')

    line4, = ax2.plot(dd_saldo[chave_maior_valor_saldo_saldo], color = 'blue', label='Saldo Trades')
    line5, = ax2.plot(dd_saldo[chave_menor_valor_r_dd_saldo], color="red", label='R_DD Saldo')
    line6, = ax2.plot(dd_saldo[chave_menor_valor_r_q_saldo], color="green", label='R_Q Saldo')


    # Legendas
    ax1.legend(handles=[line1, line2, line3], loc='upper left')

    ax1.set_title('Retorno Acumulado - Saldo Trades')
    ax2.set_title('Drawdown')

    # -----------------------------

    ### *3.3. Razão DD*

    # Criar o vetor com arredondamento
    vetor_Min_Razao_DD = np.arange(0.26, 0.92, 0.02)
    vetor_Min_Razao_DD = np.round(vetor_Min_Razao_DD, 2).tolist()

    # ------------------

    #### 3.3.1. Criando Dicionários com Resultados e Drawdown para Vetor de Max_saldo_trades

    res_rdd = {}  # Dicionário para armazenar os resultados
    picos_rdd = {}  # Dicionário para armazenar os picos de saldo
    dd_rdd = {}  # Dicionário para armazenar os drawdowns de saldo
    max_res_rdd = {}  # Dicionário para armazenar o máximo de resultados de saldo
    max_dd_rdd = {}  # Dicionário para armazenar o máximo de drawdown
    cont_operacoes = {} # Dicionário para armazenar a contagem de operações
    saldo_res_rdd = {} # Dicionário para armazenar o saldo de saldo
    r_dd_res_rdd = {}  # Dicionário para armazenar a relação de drawdown de saldo

    res_rdd_periodos = {} # Dicionário para armazenar os resultados avaliados no período selecionado
    dd_rdd_periodos = {} # Dicionário para armazenar os drawdowns avaliados no período selecionado

    quartil_res_rdd = {} # Dicionário para armazenar o primeiro quartil dos resultados
    quartil_dd_rdd = {} # Dicionário para armazenar o primeiro quartil dos drawdowns

    r_quartil_rdd = {} # Dicionário para armazenar a razão quartil de saldo

    for i, valor in enumerate(vetor_Min_Razao_DD):
        valor_arredondado = round(valor, 2)
        res_rdd[valor_arredondado] = results['Result_Min_DD'].loc[
            (results['Min_Razao_DD'] <= valor) & (results['Trades'] >= 50)
        ].cumsum()

        picos_rdd[valor_arredondado] = res_rdd[valor_arredondado].cummax()
        dd_rdd[valor_arredondado] = res_rdd[valor_arredondado] - picos_rdd[valor_arredondado]

    # ------------------

    #### 3.3.2. Por Critério Saldo Trades

    for j, valor in enumerate(vetor_Min_Razao_DD):
        valor_arredondado = round(valor, 2)

        max_res_rdd[valor_arredondado] = res_rdd[valor_arredondado].max()
        cont_operacoes[valor_arredondado] = res_rdd[valor_arredondado].count()

        # Avoid division by zero
        if (cont_operacoes[valor_arredondado] != 0):
            saldo_res_rdd[valor_arredondado] = abs(max_res_rdd[valor_arredondado] / cont_operacoes[valor_arredondado])
        else:
            saldo_res_rdd[valor_arredondado] = 0

    # Encontre a chave com o maior valor
    chave_maior_valor_saldo_res_rdd = max(saldo_res_rdd, key=saldo_res_rdd.get)

    # ------------------

    #### 3.2.3. Por Critério Razão Drawdown

    for j, valor in enumerate(vetor_Min_Razao_DD):
        valor_arredondado = round(valor, 2)


        max_dd_rdd[valor_arredondado] = dd_rdd[valor_arredondado].min()

        # Avoid division by zero
        if (max_res_rdd[valor_arredondado] != 0):
            r_dd_res_rdd[valor_arredondado] = abs(max_dd_rdd[valor_arredondado] / max_res_rdd[valor_arredondado])
        else:
            r_dd_res_rdd[valor_arredondado] = 0

    # Encontre a chave com o menor valor
    chave_menor_valor_r_dd_res_rdd = min(r_dd_res_rdd, key=r_dd_res_rdd.get)

    # ------------------

    #### 3.2.4. Por Critério Razão Quartil

    for j, valor in enumerate(vetor_Min_Razao_DD):
        valor_arredondado = round(valor, 2)

        res_rdd_periodos[valor_arredondado] = res_rdd[valor_arredondado].diff(periods = periodo)
        dd_rdd_periodos[valor_arredondado] = dd_rdd[valor_arredondado].diff(periods = periodo)

        quartil_res_rdd[valor_arredondado] = res_rdd_periodos[valor_arredondado].quantile(0.25)
        quartil_dd_rdd[valor_arredondado] = dd_rdd_periodos[valor_arredondado].quantile(0.25)

        if quartil_res_rdd[valor_arredondado] > 0:
            r_quartil_rdd[valor_arredondado] = (-quartil_dd_rdd[valor_arredondado]) / quartil_res_rdd[valor_arredondado]
        else:
            r_quartil_rdd[valor_arredondado] = 1

    # Encontre a chave com o menor valor
    chave_menor_valor_r_q_res_rdd = min(r_quartil_rdd, key=r_quartil_rdd.get)

    # ------------------

    #### 3.2.5. Comparação

    fig = plt.figure(figsize = (10,5))
    gs = fig.add_gridspec(nrows = 10, ncols = 1)

    ax1 = fig.add_subplot(gs[0:5, 0])
    ax2 = fig.add_subplot(gs[5:11, 0])

    # Plots
    line1, = ax1.plot(res_rdd[chave_maior_valor_saldo_res_rdd],color = 'blue', label='Saldo Trades')
    line2, = ax1.plot(res_rdd[chave_menor_valor_r_dd_res_rdd], color='red', label='R_DD Saldo')
    line3, = ax1.plot(res_rdd[chave_menor_valor_r_q_res_rdd], color='green', label='R_Q Saldo')

    line4, = ax2.plot(dd_rdd[chave_maior_valor_saldo_res_rdd], color = 'blue', label='Saldo Trades')
    line5, = ax2.plot(dd_rdd[chave_menor_valor_r_dd_res_rdd], color="red", label='R_DD Saldo')
    line6, = ax2.plot(dd_rdd[chave_menor_valor_r_q_res_rdd], color="green", label='R_Q Saldo')


    # Legendas
    ax1.legend(handles=[line1, line2, line3], loc='upper left')

    ax1.set_title('Retorno Acumulado - Saldo Trades')
    ax2.set_title('Drawdown')

    # -----------------------------

    ### *3.4. Razão Quartil*

    # Criar o vetor com arredondamento
    vetor_Min_R_Quartil = np.arange(0.02, 0.9, 0.02)
    vetor_Min_R_Quartil = np.round(vetor_Min_R_Quartil, 2).tolist()

    # ------------------

    #### 3.4.1. Criando Dicionários com Resultados e Drawdown para Vetor de Max_saldo_trades

    res_rq = {}  # Dicionário para armazenar os resultados
    picos_rq = {}  # Dicionário para armazenar os picos de saldo
    dd_rq = {}  # Dicionário para armazenar os drawdowns de saldo
    max_res_rq = {}  # Dicionário para armazenar o máximo de resultados de saldo
    max_dd_rq = {}  # Dicionário para armazenar o máximo de drawdown
    cont_operacoes = {} # Dicionário para armazenar a contagem de operações
    saldo_res_rq = {} # Dicionário para armazenar o saldo de saldo
    r_dd_res_rq = {}  # Dicionário para armazenar a relação de drawdown de saldo

    res_rq_periodos = {} # Dicionário para armazenar os resultados avaliados no período selecionado
    dd_rq_periodos = {} # Dicionário para armazenar os drawdowns avaliados no período selecionado

    quartil_res_rq = {} # Dicionário para armazenar o primeiro quartil dos resultados
    quartil_dd_rq = {} # Dicionário para armazenar o primeiro quartil dos drawdowns

    r_quartil_rq = {} # Dicionário para armazenar a razão quartil de saldo

    for i, valor in enumerate(vetor_Min_R_Quartil):
        valor_arredondado = round(valor, 2)
        res_rq[valor_arredondado] = results['Result_Min_Quartil'].loc[
            (results['Min_R_Quartil'] <= valor) & (results['Trades'] >= 50)
        ].cumsum()

        picos_rq[valor_arredondado] = res_rq[valor_arredondado].cummax()
        dd_rq[valor_arredondado] = res_rq[valor_arredondado] - picos_rq[valor_arredondado]

    # ------------------

    #### 3.4.2. Por Critério Saldo Trades

    for j, valor in enumerate(vetor_Min_R_Quartil):
        valor_arredondado = round(valor, 2)

        max_res_rq[valor_arredondado] = res_rq[valor_arredondado].max()
        cont_operacoes[valor_arredondado] = res_rq[valor_arredondado].count()

        # Avoid division by zero
        if (cont_operacoes[valor_arredondado] != 0):
            saldo_res_rq[valor_arredondado] = abs(max_res_rq[valor_arredondado] / cont_operacoes[valor_arredondado])
        else:
            saldo_res_rq[valor_arredondado] = 0

    # Encontre a chave com o maior valor
    chave_maior_valor_saldo_res_rq = max(saldo_res_rq, key=saldo_res_rq.get)

    # ------------------

    #### 3.4.3. Por Critério Razão Drawdown

    for j, valor in enumerate(vetor_Min_R_Quartil):
        valor_arredondado = round(valor, 2)


        max_dd_rq[valor_arredondado] = dd_rq[valor_arredondado].min()

        # Avoid division by zero
        if (max_res_rq[valor_arredondado] != 0):
            r_dd_res_rq[valor_arredondado] = abs(max_dd_rq[valor_arredondado] / max_res_rq[valor_arredondado])
        else:
            r_dd_res_rq[valor_arredondado] = 0

    # Encontre a chave com o menor valor
    chave_menor_valor_r_dd_res_rq = min(r_dd_res_rq, key=r_dd_res_rq.get)

    # ------------------

    #### 3.4.4. Por Critério Razão Quartil

    for j, valor in enumerate(vetor_Min_R_Quartil):
        valor_arredondado = round(valor, 2)

        res_rq_periodos[valor_arredondado] = res_rq[valor_arredondado].diff(periods = periodo)
        dd_rq_periodos[valor_arredondado] = dd_rq[valor_arredondado].diff(periods = periodo)

        quartil_res_rq[valor_arredondado] = res_rq_periodos[valor_arredondado].quantile(0.25)
        quartil_dd_rq[valor_arredondado] = dd_rq_periodos[valor_arredondado].quantile(0.25)

        if quartil_res_rq[valor_arredondado] > 0:
            r_quartil_rq[valor_arredondado] = (-quartil_dd_rq[valor_arredondado]) / quartil_res_rq[valor_arredondado]
        else:
            r_quartil_rq[valor_arredondado] = 1

    # Encontre a chave com o menor valor
    chave_menor_valor_r_q_res_rq = min(r_quartil_rq, key=r_quartil_rq.get)

    # ------------------

    #### 3.4.5. Comparação

    fig = plt.figure(figsize = (10,5))
    gs = fig.add_gridspec(nrows = 10, ncols = 1)

    ax1 = fig.add_subplot(gs[0:5, 0])
    ax2 = fig.add_subplot(gs[5:11, 0])

    # Plots
    line1, = ax1.plot(res_rq[chave_maior_valor_saldo_res_rq],color = 'blue', label='Saldo Trades')
    line2, = ax1.plot(res_rq[chave_menor_valor_r_dd_res_rq], color='red', label='R_DD Saldo')
    line3, = ax1.plot(res_rq[chave_menor_valor_r_q_res_rq], color='green', label='R_Q Saldo')

    line4, = ax2.plot(dd_rq[chave_maior_valor_saldo_res_rq], color = 'blue', label='Saldo Trades')
    line5, = ax2.plot(dd_rq[chave_menor_valor_r_dd_res_rq], color="red", label='R_DD Saldo')
    line6, = ax2.plot(dd_rq[chave_menor_valor_r_q_res_rq], color="green", label='R_Q Saldo')


    # Legendas
    ax1.legend(handles=[line1, line2, line3], loc='upper left')

    ax1.set_title('Retorno Acumulado - Saldo Trades')
    ax2.set_title('Drawdown')
    
def main():
    # ------------------

    # Inputs

    #///////// Inserir formulário para streamlit

    # Título do aplicativo
    st.title('Operacional Volume Completo')
    st.markdown('---')
    
    # Time Frame Desejado
    st.sidebar.markdown('---')
    time_frame = st.sidebar.number_input('Insira o Time Frame', value=1, step=1)
    time_frame = int(time_frame)  # Converter para int
    
    #### 2.2.1. Início e limite de abertura de posições [horas (24h)] - valores inclusos

    inicio_abertura = st.sidebar.number_input('Horário de ínicio de abertura de posições', value=9, step=1 ,min_value = 9, max_value = 18)
    limite_abertura = st.sidebar.number_input('Horário limite de abertura de posições', value=11, step=1 ,min_value = inicio_abertura+1, max_value = 18)

    # ------------------

    #### 2.2.2. Limite de fechamento de posições [horas (24h)] - valores não-inclusos

    limite_fechamento = st.sidebar.number_input('Horário limite de fechamento de posições', value=17, step=1 ,min_value = inicio_abertura+1, max_value = 18)
    
    # ------------------

    # Widget para upload de arquivo
    uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type=["csv"])
    
    continuar = st.sidebar.button('Analisar')
    
    if uploaded_file:
        etl(uploaded_file, continuar, time_frame, inicio_abertura, limite_abertura, limite_fechamento)
        
    
def menu():
    st.sidebar.title('Operacional Volume Completo')
    st.sidebar.markdown('---')
    
    lista = ['Início', 'Backtest']
    
    choice = st.sidebar.radio('Menu', lista)
    
    if choice == 'Backtest':
        main()
        
    else:
        st.title('Operacional Volume Completo')
        st.markdown('---')
        st.write('App criado com o intuito de trazer uma visualização detalhada da análise de um operacional baseado em análise técnica e volume financeiro')

#### Importando bibliotecas

import streamlit as st
import time
tempo = []
start_time = time.time()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from scipy import stats
import pylab
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sweetviz as sv

menu()