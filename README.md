# desafioBBCpuAfetiva






```
git clone https://github.com/IvanAmador/desafioBBCpuAfetiva.git
```
Instalar dependências em ambiente virtual com [Pipenv](https://pypi.org/project/pipenv/) (recomendado)

```
pipenv install -r requirements.txt
```

ou
```
pip install -r requirements.txt
```


Utilizando pelo Prompt de Comando
```
python main_predict.py -f "<filepath>"
```


Importando
```python
import main_predict as mp

mp.fer(r"<filepath>")
```



