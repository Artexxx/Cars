import sys

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction
# from rich.traceback import install

# install(show_locals=True)
from rich.console import Console

console = Console()

# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="Анализ рынка подержанных автомобилей",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding = 'Latin-1')

    # Преобразование колонок `price` и `odometer` к числовому типу
    df['price'] = df['price'].str.strip().str.replace('$', '').str.replace(',', '').astype(int)
    df['odometer'] = df['odometer'].str.strip().str.replace('km', '').str.replace(',', '').astype(int)

    # Удаление колонок, не несущих значимой информации для анализа
    df = df.drop(columns=['seller', 'offerType', 'nrOfPictures'])

    # Удаление аномальных значений
    df = df[df['yearOfRegistration'].between(1900, 2016)]
    df = df[df['price'].between(10, 350001)]
    df = df[df['powerPS'].between(10, 1000)]

    # Заменим пропущенные значения в категориальных данных на 'not_specified'
    columns_to_fill = ['vehicleType', 'gearbox', 'model', 'fuelType', 'notRepairedDamage']
    df[columns_to_fill] = df[columns_to_fill].fillna('not_specified')

    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        Q3 = df[column].quantile(0.75)
        Q1 = df[column].quantile(0.25)
        IQR = Q3 - Q1
        upper = Q3 + (1.5 * IQR)
        lower = Q1 - (1.5 * IQR)
        df = df[(df[column] > lower) & (df[column] < upper)]
    return df


class Menu:
    apps = [
        {
            "func": home.app,
            "title": "Главная",
            "icon": "house-fill"
        },
        {
            "func": prediction.app,
            "title": "Прогнозирование",
            "icon": 'graph-up'
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons = [app["icon"] for app in self.apps]
            st.image('images/main.webp', use_column_width='auto')

            selected = option_menu(
                "Меню",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )

            st.info("""
                 ### Область применения
                Система анализа данных помогает потенциальным покупателям, продавцам и компаниям в автоиндустрии понимать текущие тенденции и ценообразование. 
                Это инструмент для изучения популярности различных марок, моделей, влияния года выпуска и пробега на стоимость автомобилей.
            """)
        return selected


if __name__ == '__main__':
    # try:
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    df = load_data(current_dir / 'autos.csv')

    menu = Menu()
    selected = menu.run()

    # Добавление интерфейса для загрузки файла
    st.sidebar.header('Загрузите свой файл')
    uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif df is None:
        st.sidebar.warning("Пожалуйста, загрузите файл данных.")

    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break

    # except Exception as e:
    #     console.print_exception()
    #     st.error("Произошла ошибка во время исполнения приложения.", icon="🚨")
    #     raise
