import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
from pygwalker.api.streamlit import StreamlitRenderer
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def get_profile_report(df):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(
        df,
        title="Анализ данных электроэнергии",
        dataset={
            "description": "Этот репорт представляет анализ данных по потреблению и ценам на электроэнергию, включая информацию о температуре, времени суток и других факторах, влияющих на потребление и стоимость."
        },
        variables={
            "descriptions": {
                "Время": "Время, в течение которого были зарегистрированы данные.",
                "Цена": "Стоимость электроэнергии за данный час в рублях.",
                "ДаммиЧасы": "Дамми переменные для каждого часа дня.",
                "ДаммиНомерНедели": "Дамми переменные для номера недели в году.",
                "T": "Температура воздуха на момент измерения в градусах Цельсия.",
                "ДаммиПогодныеУсловия": "Дамми переменные, описывающие погодные условия.",
                "Доллар": "Курс доллара к рублю на момент сбора данных.",
                "Длина ночи": "Продолжительность ночи в часах на дату сбора данных.",
                "Потребление": "Общее потребление электроэнергии за час в мегаватт-часах."
            }
        }
    )
    return profile


@st.cache_resource
def display_pygwalker(df) -> "StreamlitRenderer":
    vis_spec = r"""{"config":[{"config":{"defaultAggregated":false,"geoms":["auto"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0},"encodings":{"dimensions":[{"dragId":"gw_VeNF","fid":"Время","name":"Время","basename":"Время","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_mzZI","fid":"0:00-1:00","name":"0:00-1:00","basename":"0:00-1:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_JfLw","fid":"10:00-11:00","name":"10:00-11:00","basename":"10:00-11:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_EMbW","fid":"11:00-12:00","name":"11:00-12:00","basename":"11:00-12:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_ZeJ4","fid":"12:00-13:00","name":"12:00-13:00","basename":"12:00-13:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_g6JZ","fid":"13:00-14:00","name":"13:00-14:00","basename":"13:00-14:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_IrjC","fid":"14:00-15:00","name":"14:00-15:00","basename":"14:00-15:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_GVjF","fid":"15:00-16:00","name":"15:00-16:00","basename":"15:00-16:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_BRya","fid":"16:00-17:00","name":"16:00-17:00","basename":"16:00-17:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_VuG7","fid":"17:00-18:00","name":"17:00-18:00","basename":"17:00-18:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_W1Fl","fid":"18:00-19:00","name":"18:00-19:00","basename":"18:00-19:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_0Sdp","fid":"19:00-20:00","name":"19:00-20:00","basename":"19:00-20:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_8znI","fid":"1:00-2:00","name":"1:00-2:00","basename":"1:00-2:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_XG4n","fid":"20:00-21:00","name":"20:00-21:00","basename":"20:00-21:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_-V8-","fid":"21:00-22:00","name":"21:00-22:00","basename":"21:00-22:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_T0Gy","fid":"22:00-23:00","name":"22:00-23:00","basename":"22:00-23:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_TMq1","fid":"23:00-0:00","name":"23:00-0:00","basename":"23:00-0:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_pOat","fid":"2:00-3:00","name":"2:00-3:00","basename":"2:00-3:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_TA4n","fid":"3:00-4:00","name":"3:00-4:00","basename":"3:00-4:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_HPt5","fid":"4:00-5:00","name":"4:00-5:00","basename":"4:00-5:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_lucU","fid":"5:00-6:00","name":"5:00-6:00","basename":"5:00-6:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_hnbG","fid":"6:00-7:00","name":"6:00-7:00","basename":"6:00-7:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_0W2Y","fid":"7:00-8:00","name":"7:00-8:00","basename":"7:00-8:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_umCG","fid":"8:00-9:00","name":"8:00-9:00","basename":"8:00-9:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_vqhB","fid":"9:00-10:00","name":"9:00-10:00","basename":"9:00-10:00","semanticType":"nominal","analyticType":"dimension","offset":0},{"dragId":"gw_HeuD","fid":"ДаммиНомерНедели","name":"ДаммиНомерНедели","basename":"ДаммиНомерНедели","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_PwV3","fid":"ДаммиПогодныеУсловия","name":"ДаммиПогодныеУсловия","basename":"ДаммиПогодныеУсловия","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_mea_key_fid","fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"dragId":"gw_wJbY","fid":"Цена","name":"Цена","basename":"Цена","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw__ftY","fid":"ДаммиЧасы","name":"ДаммиЧасы","basename":"ДаммиЧасы","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_Tc9p","fid":"T","name":"T","basename":"T","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_O01l","fid":"Доллар","name":"Доллар","basename":"Доллар","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_D4We","fid":"Длина ночи","name":"Длина ночи","basename":"Длина ночи","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_0uAN","fid":"Потребление","name":"Потребление","basename":"Потребление","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_count_fid","fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"dragId":"gw_mea_val_fid","fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"dragId":"gw_tAzA","fid":"Цена","name":"Цена","basename":"Цена","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw__PKp","fid":"Потребление","name":"Потребление","basename":"Потребление","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0}],"columns":[{"dragId":"gw_jIoW","fid":"ДаммиЧасы","name":"ДаммиЧасы","basename":"ДаммиЧасы","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0}],"color":[{"dragId":"gw_Tgqg","fid":"ДаммиНомерНедели","name":"ДаммиНомерНедели","basename":"ДаммиНомерНедели","semanticType":"quantitative","analyticType":"dimension","offset":0}],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"full","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_5QSY","name":"Chart 1"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"view","query":[{"op":"raw","fields":["ДаммиНомерНедели","ДаммиЧасы","Цена","Потребление"]}]}]}],"version":"0.4.7"}"""
    return StreamlitRenderer(df, spec=vis_spec, dark='light', spec_io_mode="rw")


@st.cache_data
def create_plot(df, feature):
    fig = go.Figure(go.Scatter(
        x=df['Время'],
        y=df[feature],
        line=dict(color='black', width=3)
    ))

    fig.update_layout(
        title=f"Распределение {feature}",
        template="plotly_white",
        xaxis_title='Время',
        yaxis_title=feature,
        font=dict(size=12),
        xaxis=dict(
            tickangle=45,
            type='category',
            ticklabelstep=3
        )
    )
    return fig


@st.cache_data
def create_box_plot(df, feature):
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=df[feature],
            name=feature,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=2
        )
    )

    # Настройка внешнего вида графика
    fig.update_layout(
        title=f"Ящик с усами для {feature}",
        yaxis=dict(
            title='Значения',
            zeroline=False
        ),
        xaxis=dict(
            title='Признак',
            tickangle=45,
            type='category'
        ),
        boxmode='group',
        template='plotly_white'
    )
    return fig


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
    fig.update_layout(height=800)
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
def plot_energy_prices(df):
    plt.figure(figsize=(15, 8))
    for i in range(len(df) // 24):
        colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
        daily_data = df.iloc[i * 24: (i + 1) * 24]
        date = pd.to_datetime(daily_data["Время"]).dt.strftime('%Y-%m-%d').iloc[0]
        plt.plot(
            daily_data['ДаммиЧасы'], daily_data['Цена'],
            alpha=0.5, color=colors[i % len(colors)],
            marker='.', markersize=12, label=date
        )

    plt.title('Цены на электроэнергию по часам', fontsize=16)
    plt.xlabel('Час', fontsize=14)
    plt.ylabel('Цена', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 24))
    plt.grid(True)
    plt.legend(loc='lower right')
    st.pyplot(plt)


@st.cache_data
def create_countplot(df, categorical_features):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 35))
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
    feature2 = c2.selectbox('Второй признак', numerical_features, index=4, key='scatter_feature2')
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


def app(df, current_dir: Path):
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Главная страница                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.title("Прогнозирование цен на электроэнергию")
    st.markdown("## Область применения")
    markdown_col1, markdown_col2 = st.columns(2)

    markdown_col1.markdown(
        """
        Эта страница предназначена для описания использования данных и общего контекста проекта. 
        Примеры анализа данных и прогнозирования включают использование машинного обучения и статистического анализа для 
        прогнозирования цен на электроэнергию.
        """
    )
    markdown_col2.image(str(current_dir / 'images' / 'logo.png'), width=150)

    tab1, tab2 = st.tabs(["Описание данных", "Пример данных"])

    with tab2:
        st.header("Пример данных")
        st.dataframe(df.head())
    with tab1:
        st.markdown(
            """
            | Параметр                   | Описание                                                                                     |
            |----------------------------|----------------------------------------------------------------------------------------------|
            | Время                      | Час дня, в течение которого были зарегистрированы данные.                                     |
            | Цена                       | Стоимость электроэнергии за данный час в рублях.                                              |
            | 0:00-1:00 - 23:00-0:00     | Данные по потреблению электроэнергии в каждый часовой интервал за день.                       |
            | ДаммиЧасы                  | Индикаторные переменные для каждого часа дня.                                                 |
            | ДаммиНомерНедели           | Индикаторные переменные для номера недели в году.                                             |
            | T                          | Температура воздуха на момент измерения в градусах Цельсия.                                   |
            | ДаммиПогодныеУсловия       | Индикаторные переменные, описывающие погодные условия (например, дождь, снег).                 |
            | Доллар                     | Курс доллара к рублю на момент сбора данных.                                                  |
            | Длина ночи                 | Продолжительность ночи в часах на дату сбора данных.                                          |
            | Потребление                | Общее потребление электроэнергии за час в мегаватт-часах.                                     |
            """,
            unsafe_allow_html=True
        )

    numerical_features   = ['Цена', 'T', 'Доллар', 'Длина ночи', 'Потребление']
    categorical_features = ['ДаммиЧасы', 'ДаммиНомерНедели', 'ДаммиПогодныеУсловия']
    categorical_features += [f'{hour}:00-{hour + 1}:00' for hour in range(23)] + ['23:00-0:00']

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃               Предварительный анализ данных                 ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.subheader("Предварительный анализ данных")
    # Отображение метрик в колонках
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Максимальная цена", f"{df['Цена'].max():.2f} ₽")
    col2.metric("Минимальная цена", f"{df['Цена'].min():.2f} ₽")
    col3.metric("Средняя температура", f"{df['T'].mean():.2f} °C")
    col4.metric("Среднее потребление", f"{df['Потребление'].mean():.2f}")
    st.dataframe(get_data_info(df), use_container_width=True)

    st.subheader("Основные статистики для признаков")

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.subheader("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe())
    with tab2:
        st.subheader("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include='bool'))

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Интерактивные отчёты                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.subheader("Анализ данных")
    tab1, tab2 = st.tabs(["Редактор графиков", "Показать отчет о данных"])
    with tab1:
        renderer = display_pygwalker(df)
        renderer.render_explore()
    with tab2:
        if st.button("Сформировать отчёт", type='primary', use_container_width=True):
            st_profile_report(get_profile_report(df))

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Визуализация                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox"
    )
    # Построение и отображение графика
    plot_fig = create_plot(
        df,
        selected_feature
    )
    st.plotly_chart(plot_fig, use_container_width=True)
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Корреляция                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Корреляционный анализ")
    st.subheader("Распределение по различным признакам")
    display_scatter_plot(df, numerical_features, categorical_features)

    if st.button('Показать все переменные на корреляционной матрице', use_container_width=True):
        corr_features = numerical_features + categorical_features
    else:
        corr_features = numerical_features

    st.plotly_chart(
        create_correlation_matrix(df, corr_features),
        use_container_width=True
    )
    markdown_col1, markdown_col2 = st.columns(2)
    markdown_col1.markdown(
        """
        На рисунке представлена корреляционная тепловая карта. Это визуальный инструмент, используемый для представления коэффициентов корреляции между различными переменными. 
        *	Существует сильная положительная корреляция между "Ценой" и "Потреблением" (0.78), что может указывать на то, что увеличение потребления сопровождается повышением цены.
        *	"Цена" также сильно коррелирует с "ДаммиЧасы" (0.55), что может быть связано с тем, что в определённые часы (возможно, пиковые часы) цена выше.
        *	Между "Ценой" и "Температурой" наблюдается умеренная положительная корреляция (0.14), что может свидетельствовать о том, что изменения температуры влияют на цену.
        Основываясь на этих данных, можно предположить, что временные факторы и указанные дамми переменные имеют различное влияние на цену и другие переменные. Корреляционная тепловая карта помогает быстро идентифицировать потенциальные связи для дальнейшего более глубокого анализа.
        """
    )
    markdown_col2.dataframe(
        create_correlation_df(df, numerical_features + categorical_features, 'Цена')
        .style.background_gradient(cmap='coolwarm', subset=['Корреляция с Цена']),
        use_container_width=True,
        hide_index=True,
        height=500
    )
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Диаграммы                        ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.subheader("График исторических данных цен на электроэнергию, сгруппированных по дням")
    plot_energy_prices(df)
    st.markdown(
        """
        На рисунке представлен график, который иллюстрирует изменение цен на электроэнергию по часам. Каждая линия на графике соответствует определённой дате в указанном диапазоне. Ось Y (вертикальная) отображает цен, в то время как ось X (горизонтальная) представляет различные временные интервалы в течение суток (00:00, 01:00, и так далее до 23:00).
        Из графика видно, что цены колеблются в течение дня, при этом можно заметить определённые закономерности:
        1.	В большинстве дней наблюдается рост цен, начиная примерно с 06:00, достигающий пика в период с 09:00 до 19:00.
        2.	Затем следует спад цен после 20:00, который продолжается до 00:00 и далее до 03:00.
        3.	Наименьшие цены, как правило, наблюдаются в интервале времени с 00:00 до 05:00 утра.
        4.	После 05:00 цены снова начинают расти.
        Эти наблюдения могут указывать на то, что спрос на электроэнергию, и, соответственно, её цена, увеличиваются в дневные часы и снижаются ночью, что согласуется с типичными паттернами потребления электроэнергии.
        """
    )
    st.markdown(
        """
        ## Ящики с усами для числовых признаков
        Эти графики позволяют наглядно оценить распределение основных числовых параметров.
        """
    )
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_box_plots"
    )
    st.plotly_chart(create_box_plot(df, selected_feature), use_container_width=True)

    st.markdown(
        """
        ## Столбчатые диаграммы для категориальных признаков
        """
    )
    st.pyplot(create_countplot(df, categorical_features))

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features + categorical_features,
        default=numerical_features,
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=1,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
