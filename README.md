# classification-of-oil-and-gas

## Подготовка проекта

Склонируйте репозиторий:

```shell
git clone https://github.com/AlekseiAgarkov/classification-of-oil-and-gas.git
cd classification-of-oil-and-gas
```

Установите uv: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

Запустите:

```shell
uv venv --python 3.11
```

Активация Linux/Mac:

```shell
source .venv/bin/activate
```

Активация в Windows:

```shell
.venv\Scripts\activate
```

Установите зависимости:

```shell
uv pip install -e .
```

# Citation / Цитирование
* Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.

![NEV-Logo-color.png](doc%2Fimg%2FNEV-Logo-color.png)

## Конфигурация Kaggle Submitter Module

Kaggle Submitter Module - это модуль для автоматической отправки сабмишнов в Kaggle и взаимодействия с метаданными
моделей.

### Создание API ключ на Kaggle

Создайте API ключ на Kaggle:

1. Зайдите в Settings -> API
2. Нажмите "Create New API Token"
3. Следуйте инструкции с сохранением API Token'а в ~/.kaggle/

### Использование

Минимальный пример использования KaggleSubmitter:

```python
from src.submission.kaggle_integration import KaggleSubmitter

submitter = KaggleSubmitter(competition="classification-of-oil-and-gas")
best_submission_kaggle_info = submitter.get_submission_by_hash("some_hash")
print(best_submission_kaggle_info)
```
