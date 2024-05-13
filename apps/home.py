import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots



@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info

@st.cache_data
def create_histogram(df, column_name):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",
        title=f"Распределение {column_name}",
        template="plotly"
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, features):
    corr = df[features].corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='ice',
        annotation_text=corr.values
    )
    fig.update_layout(height=500)
    return fig


@st.cache_data
def create_correlation_df(df, features, target_feature):
    correlation_matrix = df[features].corr()
    correlation_with_target = correlation_matrix[target_feature].round(2)
    correlation_df = pd.DataFrame({
        'Признак': correlation_with_target.index,
        'Корреляция с ' + target_feature: correlation_with_target.values
    })
    return correlation_df


@st.cache_data
def create_countplot(df, categorical_features):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5, bottom=0)

    for ax, catplot in zip(axes.flatten(), categorical_features):
        sns.countplot(x=catplot, data=df, ax=ax)
        ax.set_title(catplot.upper(), fontsize=18)
        ax.set_ylabel('Count', fontsize=16)
        ax.set_xlabel(f'{catplot} Values', fontsize=15)
        ax.tick_params(axis='x', rotation=45)
    return fig


@st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        palette='viridis',
        plot_kws={'alpha': 0.5, 's': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


def display_scatter_plot(df, numerical_features, categorical_features):
    from scipy.stats import stats
    c1, c2, c3, c4 = st.columns(4)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='scatter_feature1')
    feature2 = c2.selectbox('Второй признак', numerical_features, index=2,
                            key='scatter_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, *categorical_features],
                             key='scatter_filter_by')

    correlation = round(stats.pearsonr(df[feature1], df[feature2])[0], 4)
    c4.metric("Корреляция", correlation)

    fig = px.scatter(
        df,
        x=feature1, y=feature2,
        color=filter_by, trendline='ols',
        opacity=0.5,
        template='plotly',
        title=f'Корреляция между {feature1} и {feature2}'
    )
    st.plotly_chart(fig, use_container_width=True)


def viz1(df):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Рассчитаем ширину столбцов
    bar_width = 0.35

    # Первый столбчатый график для средней цены
    color1 = 'tab:red'
    ax1.set_xlabel('Brand')
    ax1.set_ylabel('Mean Price', color=color1)
    bars1 = ax1.bar(np.arange(len(df.index)), df['Mean Price'], width=bar_width, color=color1, label='Mean Price')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(np.arange(len(df.index)))
    ax1.set_xticklabels(df.index, rotation=90)  # Поворачиваем подписи на оси X

    # Второй столбчатый график для среднего пробега
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Mean Mileage', color=color2)
    bars2 = ax2.bar(np.arange(len(df.index)) + bar_width, df['Mean Mileage'], width=bar_width, color=color2,
                    label='Mean Mileage')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Добавляем легенду
    bars = [bars1, bars2]
    labels = [bar.get_label() for bar in bars]
    plt.legend(bars, labels, loc='upper right')

    # Заголовок
    plt.title('Mean Price vs Mean Mileage by Popular Brand')
    return fig

def app(df, current_dir: Path):
    cm = sns.light_palette("seagreen", as_cmap=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Главная страница                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.title("Анализ рынка подержанных автомобилей")
    st.markdown("## Область применения")
    markdown_col1, markdown_col2 = st.columns(2)

    markdown_col1.markdown(
        """
        Система анализа данных находит широкое применение в изучении рынка подержанных автомобилей. Это позволяет потенциальным покупателям, продавцам, а также компаниям, работающим в автомобильной индустрии, лучше понимать текущие тенденции, ценообразование и предпочтения потребителей. Анализ данных может помочь выявить наиболее популярные марки и модели автомобилей, определить средние цены на различные типы транспортных средств, а также предоставить информацию о сезонности продаж и другие важные рыночные инсайты.
        """
    )
    markdown_col2.image(str(current_dir / 'images' / 'i.webp'), width=500, use_column_width='auto')

    tab1, tab2, tab3 = st.tabs(["Описание данных", "Подробное описание", "Пример данных"])

    with tab1:
        st.markdown(
            r"""
            ## Ключевые параметры и характеристики данных
            Из оригинального датасета было взято только 50000 данных, чтобы обеспечить быстрое выполнение кода ниже.
            
            Файл данных представляет собой следующее:
            
            | Поле                 | Описание                                                               |
            |----------------------|------------------------------------------------------------------------|
            | dateCrawled          | Дата скачивания анкеты из базы                                         |
            | name                 | Название автомобиля                                                    |
            | seller               | Частный продавец или дилер                                             |
            | offerType            | Тип предложения                                                        |
            | price                | Цена в объявлении                                                      |
            | abtest               | Участие объявления в A/B тесте                                         |
            | vehicleType          | Тип транспортного средства                                             |
            | yearOfRegistration   | Год первой регистрации автомобиля                                      |
            | gearbox              | Тип коробки передач                                                     |
            | powerPS              | Мощность автомобиля в PS                                               |
            | model                | Модель автомобиля                                                      |
            | odometer             | Пробег автомобиля                                                      |
            | monthOfRegistration  | Месяц первой регистрации автомобиля                                    |
            | fuelType             | Тип топлива                                                            |
            | brand                | Марка автомобиля                                                       |
            | notRepairedDamage    | Наличие непочиненных повреждений                                       |
            | dateCreated          | Дата создания объявления на eBay                                       |
            | nrOfPictures         | Количество фотографий в объявлении                                     |
            | postalCode           | Почтовый индекс местонахождения автомобиля                             |
            | lastSeenOnline       | Последний раз объявление было замечено в онлайне                       |
            """
        )
    with tab2:
        st.markdown(
            """
            ### Ключевые параметры и характеристики данных
            
            1. **Цена (price)**: Цена является одним из самых важных параметров, влияющих на выбор покупателя. Анализ распределения цен на различные марки и модели, а также их зависимость от года выпуска, пробега и состояния автомобиля позволит понять текущие тенденции рынка.
            
            2. **Марка и модель (brand, model)**: Популярность определённых марок и моделей может сильно варьироваться в зависимости от региона и времени года. Анализ этих данных поможет выявить наиболее востребованные автомобили на рынке.
            
            3. **Год выпуска (yearOfRegistration)**: Год выпуска автомобиля влияет на его стоимость и популярность. Анализируя эти данные, можно определить, какие возрастные группы автомобилей пользуются наибольшим спросом.
            
            4. **Пробег (odometer)**: Пробег напрямую связан с состоянием автомобиля и его рыночной стоимостью. Изучение этого параметра позволяет оценить ожидаемую продолжительность эксплуатации автомобиля и его ценность.
            
            5. **Тип топлива (fuelType)** и **Тип коробки передач (gearbox)**: Эти параметры влияют на эксплуатационные характеристики и стоимость обслуживания автомобиля. Анализ распределения автомобилей по типу топлива и коробки передач может выявить предпочтения потребителей и тенденции рынка.
            
            6. **Повреждения (notRepairedDamage)**: Наличие или отсутствие повреждений и их характер могут значительно повлиять на цену автомобиля и его привлекательность для покупателей.
            
            7. **Регион продажи (postalCode)**: Региональный анализ может выявить географические особенности спроса и предложения, что особенно важно для крупных стран с разнообразным климатом и экономическими условиями.
            
            Анализ этих ключевых параметров и характеристик данных позволит получить комплексное представление о рынке подержанных автомобилей, выявить текущие тренды и предпочтения покупателей, а также помочь в принятии обоснованных решений как для покупателей, так и для продавцов подержанных автомобилей.
            """
        )
    with tab3:
        st.header("Пример данных")
        st.dataframe(df.head(15))

    # categorical_features = ['vehicleType', 'gearbox', 'fuelType', 'brand']
    numerical_features = ['price', 'powerPS', 'odometer', 'yearOfRegistration']
    categorical_features = df.select_dtypes(include='object').columns.tolist()

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃               Предварительный анализ данных                 ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.subheader("Предварительный анализ данных")
    # Отображение метрик в колонках
    col1, col2, col3 = st.columns(3)
    average_price = df['price'].mean()
    average_odometer = df['odometer'].mean()
    unique_brands = df['brand'].nunique()
    with col1:
        st.metric("Средняя цена", f"${average_price:,.2f}")
    with col2:
        st.metric("Средний пробег", f"{average_odometer:,.2f} км")
    with col3:
        st.metric("Количество уникальных брендов", unique_brands)

    st.dataframe(get_data_info(df), use_container_width=True)

    st.markdown("""
        Предварительный анализ данных показал следующее:
        * всего в датасете 50,000 записей и 20 колонок.
        * не обнаружено пропущенных значений в данных;
    """)

    st.header("Основные статистики для признаков")

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.header("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe(), use_container_width=True)
    with tab2:
        st.header("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include='object'), use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Анализ                           ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.markdown(
            """
            ## Изучение брендов по цене
            Рассмотрим среднюю цену наиболее популярных 15 брендов, представленных в нашем наборе данных, чтобы выявить любые тенденции.
            """
        )
        popBrands = df['brand'].value_counts().head(15).index
        brandPriceData = {}

        for b in popBrands:
            brandPrices = df.loc[df['brand'] == b, 'price']
            brandPriceData[b] = int(brandPrices.mean())

        avgBrandPrices = pd.Series(brandPriceData)
        df_avg = pd.DataFrame(
            avgBrandPrices,
            columns=['Mean Price']

        ).sort_values(by=['Mean Price'], ascending=False)
        st.dataframe(df_avg.style.background_gradient(cmap=cm), use_container_width=True)
    with ccol2:
        st.markdown("""
            ## Изучение брендов по пробегу
            Рассмотрим средний пробег тех же 15 самых популярных марок. Это позволит легко сравнивать среднюю цену и средний пробег автомобилей различных марок.
        """)
        brandMileData = {}
        for b in popBrands:
            brandMiles = df.loc[df['brand'] == b, 'odometer']
            brandMileData[b] = int(brandMiles.mean())

        brandMileData_series = pd.Series(brandMileData)
        df_avg['Mean Mileage'] = brandMileData_series
        st.dataframe(df_avg, use_container_width=True)

    # st.pyplot(viz1(df_avg), use_container_width=True)
    cccol1, cccol2 = st.columns(2)
    with cccol1:
        st.markdown('Корреляция Пирсона между mean price & mean mileage - Luxury German Cars')
        st.write(df_avg.loc[['audi', 'mercedes_benz', 'bmw'], :].corr())
    with cccol2:
        st.markdown('Корреляция Пирсона между mean price & mean mileage - Non-Luxury Cars')
        st.write(df_avg.iloc[3:, :].corr())

    st.markdown("""
        Учитывая только самые дорогие немецкие марки автомобилей (Audi, Mercedes Benz, BMW), кажется, что их средняя цена уменьшается при более высоких значениях пробега. Это согласуется с интуицией, так как автомобиль с более высоким пробегом, скорее всего, использовался больше, что приводит к снижению цены продажи. Эта обратная корреляция подтверждается высоким отрицательным коэффициентом Пирсона (r = -0.99).

        Для остальных автомобилей в списке не наблюдается явной тенденции в отношении их указанной цены и пробега. Коэффициент Пирсона (r = -0.24) указывает на очень слабую отрицательную корреляцию между ценой и пробегом. Это отсутствие корреляции может быть обусловлено различными мнениями продавцов о относительной стоимости различных марок автомобилей не класса люкс. Кроме того, другие факторы, помимо пробега, вероятно, также влияют на ценообразование автомобилей.
    
        Направление, выбранное для исследования, представляло собой анализ средней цены и среднего пробега 15 самых популярных марок автомобилей в наборе данных. Было установлено, что роскошные немецкие марки автомобилей, как правило, имеют самую высокую цену, причем их цены снижаются с увеличением пробега. Остальные популярные марки автомобилей не демонстрировали определенной тенденции в отношении корреляции между ценой и пробегом.
    """)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Визуализация                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox"
    )
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("""
        ## Ящики с усами для числовых признаков
        На представленных ящиках с усами (`boxplot`) для числовых признаков в датасете автомобилей можно наблюдать следующее:
        
        1. **Цена (`price`)**: Распределение цен на автомобили имеет значительное количество выбросов, что указывает на наличие очень дорогих автомобилей по сравнению с большинством предложений. Основная масса цен сосредоточена в более низком диапазоне, что типично для рынка подержанных автомобилей.
        
        2. **Год регистрации (`yearOfRegistration`)**: График показывает, что большинство автомобилей были зарегистрированы в период с начала 1990-х до 2010 года. Существуют также выбросы, указывающие на очень старые автомобили, а также на некоторые аномальные значения, такие как автомобили с годом регистрации в будущем, что явно является ошибкой в данных.
        
        3. **Мощность (`powerPS`)**: Мощность автомобилей также демонстрирует широкий разброс значений с большим количеством выбросов. Это указывает на наличие автомобилей с очень высокой мощностью. Вероятно, существуют некорректные или завышенные значения, требующие дополнительной очистки данных.
        
        4. **Пробег (`odometer`)**: Распределение пробега показывает, что большинство автомобилей имеют пробег выше среднего, с меньшим количеством автомобилей с низким пробегом. Это ожидаемо для рынка подержанных автомобилей. Присутствие выбросов не очень заметно на этом графике, что указывает на относительно однородное распределение пробега.
        
        
        Эти графики помогают выявить потенциальные аномалии и распределения значений по различным числовым признакам в датасете.
    """)

    st.image(str(current_dir / 'images' / 'boxplot.png'), use_column_width='auto')

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Корреляция                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Корреляционный анализ")
    st.subheader("Распределение по различным признакам")
    display_scatter_plot(df, numerical_features, categorical_features)


    markdown_col1, markdown_col2 = st.columns(2)
    markdown_col1.markdown("""
      Корреляционная матрица представляет связь между различными числовыми параметрами. В данном случае:
      - **Цена (Price)** сильно положительно коррелирует с мощностью двигателя (PowerPS) и годом регистрации (YearOfRegistration), а отрицательно с пробегом (Odometer).
      - **Мощность двигателя (PowerPS)** имеет сильную положительную корреляцию с ценой.
      - **Год регистрации (YearOfRegistration)** также положительно коррелирует с ценой, но слабее, чем мощность двигателя.
      - **Пробег (Odometer)** сильно отрицательно коррелирует с ценой, что означает, что автомобили с меньшим пробегом имеют более высокую цену.
        - **Месяц регистрации (MonthOfRegistration)** и **Почтовый индекс (PostalCode)** имеют слабую корреляцию с остальными параметрами. 
    """)
    markdown_col2.plotly_chart(create_correlation_matrix(df, numerical_features), use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Диаграммы                        ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown(
        """
        ## Столбчатые диаграммы для категориальных признаков
        """
    )
    st.markdown("""
        На столбчатых диаграммах для категориальных признаков в датасете автомобилей наблюдаются следующие особенности:
        
        1. **Тип автомобильного кузова (`vehicleType`)**: Наибольшее количество автомобилей приходится на категории лимузин (`limousine`), маленький автомобиль (`kleinwagen`), и универсал (`kombi`). Это отражает популярные выборы среди автомобилей на вторичном рынке.
        
        2. **Тип коробки передач (`gearbox`)**: Автомобили с механической коробкой передач (`manuell`) значительно преобладают над автомобилями с автоматической коробкой передач (`automatik`). Это может быть связано с более низкой стоимостью и большей распространенностью механических коробок передач на рынке подержанных автомобилей.
        
        3. **Тип топлива (`fuelType`)**: Бензиновые автомобили (`benzin`) занимают большую часть рынка, за ними следуют дизельные (`diesel`). Другие типы топлива, такие как газ (`lpg`), гибридные (`hybrid`), электрические (`elektro`), и другие варианты занимают значительно меньшую долю рынка. Это подтверждает доминирование бензиновых и дизельных двигателей на рынке подержанных автомобилей.
        
        4. **Марка автомобиля (`brand`)**: Volkswagen является самой популярной маркой, за ним следуют другие популярные бренды, такие как BMW, Opel, и Mercedes-Benz. Распределение подтверждает сильные позиции немецких автомобильных брендов на рынке.
        
        Эти диаграммы предоставляют важную информацию о распределении автомобилей по различным категориальным признакам и позволяют выявить популярные тенденции на рынке подержанных автомобилей.
    """)
    st.pyplot(create_countplot(df, ['vehicleType', 'gearbox', 'fuelType', 'brand']))

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        Данный вид графика важен для оценки распределения каждого признака и исследования возможных взаимосвязей между различными переменными в данных о подержанных автомобилях. Вот подробное описание графиков и их значение в контексте выбранной области:
        
        **Описание графиков на изображении:**
        
        - **Диагональ (верхний ряд слева направо, нижний ряд слева направо)**: На диагонали изображены распределения для каждого из трех признаков. Для `price` и `powerPS` мы видим явно скошенные вправо распределения, что указывает на наличие небольшого числа автомобилей с очень высокой ценой и мощностью. Распределение `odometer` также показывает скошенность, отражая факт, что большинство автомобилей имеют высокий пробег.
        
        - **Недиагональные графики (парные точечные диаграммы)**: Эти графики показывают отношения между двумя различными числовыми переменными. Например, график в первой строке и втором столбце показывает отношение между `price` и `powerPS`. Здесь мы можем видеть, что более мощные автомобили часто имеют более высокую цену, хотя существует и много исключений из этого правила. Аналогичные взаимосвязи видны на других недиагональных графиках между `price` и `odometer`, а также `powerPS` и `odometer`.
        
        Эти графики очень полезны для потенциальных покупателей и продавцов подержанных автомобилей, так как они предоставляют визуальное представление о том, как различные характеристики автомобиля связаны с его ценой и пробегом. Например, покупатели могут использовать эту информацию для определения разумного диапазона цен на автомобили с определенной мощностью или пробегом. Продавцы могут использовать эти данные для определения конкурентной цены на их автомобиль, учитывая его мощность и пробег.
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features,
        default=numerical_features,
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=0,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
