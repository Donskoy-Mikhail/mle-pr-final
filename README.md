
# Итоговый проект: Рекомендательная система для электронной коммерции

**Цель:** построить прототип рекомендательной системы, ориентированной на метрику **добавления в корзину (add-to-cart)**, провести эксперименты с MLflow, подготовить сервис на FastAPI в Docker, описать пайплайн дообучения в Airflow и мониторинг.

## Стек
- Python 3.11
- pandas, numpy, scipy, scikit-learn
- implicit (ALS для неявной обратной связи)
- MLflow (эксперименты, артефакты)
- FastAPI + Uvicorn (инференс)
- Prometheus client (метрики сервиса)
- Docker / docker-compose
- Airflow

## Структура репозитория
```
ecomm_recsys_project/
├── config/
│   └── config.yaml                  # Конфигурация обучения/сервиса
├── prometheus/
│   └── prometheus.yaml              # Конфигурация prometheus
├── data/                            # Папка в которую нужно поместить данные
├── ml-airflow/                      # Папка с сервисом airflow и дагом для обучения модели
├── src/
│   ├── common/
│   │   ├── io_utils.py
│   │   └── metrics.py
│   ├── pipeline/
│   │   ├── train.py                 # Тренировка и сохранение артефактов
│   │   ├── train_cli.py             # CLI-обёртка обучения
│   │   └── utils.py                 # Вспомогательные функции 
├── .gitignore
├── Dockerfile_ml_service             # Dockerfile для fastapi
├── docker-compose.yml               # docker-compose для fastapi/prometheus/grafana
├── requirements.txt                 # Полный список зависимостей
└── app.py                           # FastAPI
└── run_mlflow_server.sh             # запуск MLflow
└── recsys_ecommerce_capstone.ipynb  # ноутбук с эксперементами
└── dashboard_png.png                # Скрин мониторинга
└── dashboard.json                   # Файл для импорта мониторинга в grafana
└── README.md
```



# ПОДГОТОВКА
## Данные
Ожидается датасет с тремя таблицами в папке data:
- `events.csv` 
- `item_properties*.csv` (part1, part2)
- `category_tree.csv`

Важно чтобы в директории с docker-compose.yaml был .env с содержанием таким

```bash
DB_SOURCE_HOST=
DB_SOURCE_PORT=
DB_SOURCE_NAME=
DB_SOURCE_USER=
DB_SOURCE_PASSWORD=

DB_DESTINATION_HOST=
DB_DESTINATION_PORT=
DB_DESTINATION_NAME=
DB_DESTINATION_USER=
DB_DESTINATION_PASSWORD=

MLFLOW_S3_ENDPOINT_URL=
S3_BUCKET_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Тут изменения не нужны
EXPERIMENT_NAME=course_yandex
RUN_NAME=run_yandex


APP_DOCKER_PORT=8081
APP_VM_PORT=8081
GRAFANA_PORT=3000
GRAFANA_USER="admin"
GRAFANA_PASS="grafana" 
```


## Быстрый старт (локально)
```bash
cd ecomm_recsys_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# запустить MLflow
source .venv/bin/activate
export $(cat .env | xargs)
sh run_mlflow_server.sh

# запустить ноутбук
export $(cat .env | xargs)
jupyter notebook notebooks/recsys_ecommerce_capstone.ipynb


## Сервис FastApi/Prometheus/Grafana
```bash
docker compose up -d --build
```

## Сервис Airflow
```bash
cd ml_airflow
echo -e "\nAIRFLOW_UID=$(id -u)" >> .env 
docker compose up airflow-init 
docker compose down --volumes --remove-orphans 
docker compose up -d --build
```

# Пример запроса в сервис FastApi
curl -X POST "http://0.0.0.0:8081/recommend"   -H "Content-Type: application/json"   -d '{"user_id": 123, "k": 20, "recent_item_ids": [111,222,333]}' 


## Руководство по проект

### Трансляция бизнес-задачи
Цель— максимизировать вероятность добавления товара в корзину Офлайн-метрики: Recall@K / MAP@K / NDCG@K. В онлайне — CTR, Add-to-Cart Rate, конверсия по сессиям/пользователям.

### Инфраструктура обучения
- MLflow: локальный сервер `run_mlflow_server.sh` (артефакты в `/models`).
- Конфиг `config/config.yaml` управляет путями, параметрами и логированием.

### EDA
В ноутбуке анализ распределений событий, воронка (view → cart → purchase), активность по времени, доли новых/возвратных пользователей, сводки по товарам/категориям.

### Генеарция признаков и обучение
- Взвешенные взаимодействия (view=1, cart=5, purchase=10).
- Разделение по времени: у каждого пользователя последние взаимодействия в валидацию/тест.
- Кандидаты: популярность, ковизитация; модель ALS (implicit) с BM25/TF-IDF взвешиванием.
- Логирование метрик и артефактов в MLflow.

### Продуктивизация (сервис)
FastAPI сервис загружает артефакты модели (факторы пользователей/товаров, маппинги). Поддержан холодный старт по списку `recent_item_ids` и фоллбек на популярность.

### Мониторинг
Прометей-метрики в сервисе: latency, количество запросов

### Airflow
Сервис позволяет положить данные в папку data (events item_properties category_tree) и после отработки дага получить итоговую модель из этой же директории по пути data/models 

