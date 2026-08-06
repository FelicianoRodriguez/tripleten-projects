# Bootcamp TripleTen

Proyectos realizados en el bootcamp TripleTen


# Crear el entorno virtual

```bash
conda env create -f environment.yml
```

# Actulizar el entorno virtual

```bash
conda env create --file=environment.yml --prune

conda env update -f environment.yml --prune

conda env update --file environment.yml  --prune #oficial
```

# No olvidarse de descargar el modelo de lenguaje spicy

```bash
python -m spacy download es_core_news_sm
```

# Eliminar un entorno virtual
```bash
conda env remove --name env_name
```

---

# Proyecto film

batch_size = 1000, n_jobs=-1 - 3.30 minutos
batch_size = 100, n_jobs=-1 = 1.30 minutos
batch_size = 100, n_jobs = sin parametro (default) - 9:30 minutos