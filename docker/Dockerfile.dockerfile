# La imagen usa Ubuntu 18.04
FROM ubuntu:18.04
LABEL maintainer=@naperez

# Evita que el frontend de APT requiera interacción
ENV DEBIAN_FRONTEND=noninteractive

# Instalación de dependencias necesarias para el proyecto
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean

# Establece 'python3' como el comando por defecto para 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualiza pip a la última versión
RUN pip3 install --upgrade pip

# Instalación de librerías requeridas de Python para el proyecto
RUN pip3 install \
    tensorflow==2.4.0 \
    keras==2.4.3 \
    matplotlib \
    pandas \
    seaborn \
    numpy \
    scikit-learn \
    scipy \
    nltk \
    gensim \
    spacy==3.7.4 \
    annotated-types==0.6.0 \
    blis==0.7.11 \
    catalogue==2.0.10 \
    certifi==2024.2.2 \
    charset-normalizer==3.3.2 \
    click==8.1.7 \
    cloudpathlib==0.16.0 \
    colorama==0.4.6 \
    confection==0.1.4 \
    contourpy==1.2.1 \
    cycler==0.12.1 \
    cymem==2.0.8 \
    fonttools==4.51.0 \
    idna==3.7 \
    Jinja2==3.1.4 \
    joblib==1.4.2 \
    kiwisolver==1.4.5 \
    langcodes==3.4.0 \
    language_data==1.2.0 \
    marisa-trie==1.1.1 \
    MarkupSafe==2.1.5 \
    matplotlib==3.8.4 \
    murmurhash==1.0.10 \
    packaging==24.0 \
    pillow==10.3.0 \
    preshed==3.0.9 \
    pydantic==2.7.1 \
    pydantic_core==2.18.2 \
    pyparsing==3.1.2 \
    python-dateutil==2.9.0.post0 \
    pytz==2024.1 \
    regex==2024.5.10 \
    requests==2.31.0 \
    scikit-learn==1.4.2 \
    six==1.16.0 \
    smart-open==6.4.0 \
    spacy-legacy==3.0.12 \
    spacy-loggers==1.0.5 \
    srsly==2.4.8 \
    thinc==8.2.3 \
    threadpoolctl==3.5.0 \
    tqdm==4.66.4 \
    typer==0.9.4 \
    typing_extensions==4.11.0 \
    tzdata==2024.1 \
    urllib3==2.2.1 \
    wasabi==1.1.2 \
    weasel==0.3.4

# Descarga el modelo de Spacy
RUN pip install en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl#sha256=86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889

ENTRYPOINT ["/bin/bash"]
