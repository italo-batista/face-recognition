

Here we will use a virtual enviroment

## Executing in a Virtual Enviroment

Install virtualvenv

```sudo apt-get install python-dev python3-dev python-pip python3-pip```

Create your enviroment (here we will use _venv_)

```python -m virtualenv -p python3 venv```

* To enter your envirement run at root dir:

```source venv/bin/activate```

* To exit run:

```deactivate```

### Instaling depdendencies

```python -m pip3 install -r requirements.txt``` 