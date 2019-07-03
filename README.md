Here we will use a virtual enviroment

## Executing in a Virtual Enviroment

Install virtualvenv

`sudo apt-get install python-dev python3-dev python-pip python3-pip`

Create your enviroment (here we will use _venv_)

`python -m virtualenv -p python3 venv`

- To enter your envirement run at root dir:

`source venv/bin/activate`

- To exit run:

`deactivate`

### Instaling depdendencies

`python -m pip3 install -r requirements.txt`

#### Also install tensorflow

If you use an Ubuntu/Linux 64-bit machine, with CPU only, Python 2.7, do:

- `export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl`
- `sudo pip install $TF_BINARY_URL`

For others machines, [see this](http://tflearn.org/installation/)
