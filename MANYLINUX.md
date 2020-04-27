[Install docker](https://docs.docker.com/get-docker/).

Get the [manylinux docker environment](https://github.com/pypa/manylinux). 
At the time of this writing, `manylinux1` is compatible; however, I used `manylinx2014`. 
There is a tool which will label the manylinux binary with the oldest compatible standard. 

```bash
docker pull quay.io/pypa/manylinux2014_x86_64
```

Run an interactive bash shell in the manylinux docker environment.

```bash
docker run -it quay.io/pypa/manylinux2014_x86_64 /bin/bash
```

Inside the interactive bash shell for the docker environment, download lie_learn and change to the source directory.

```bash
git clone https://github.com/AMLab-Amsterdam/lie_learn.git
cd lie_learn
```

Create wheels. You have to determine which versions of python are appropriate.

```bash
/opt/python/cp35-cp35m/bin/python setup.py bdist_wheel
/opt/python/cp36-cp36m/bin/python setup.py bdist_wheel
/opt/python/cp37-cp37m/bin/python setup.py bdist_wheel
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel
```

Use auditwheel to check for success and modify the binaries to be labeled with the oldest compatible standard (lowest
 priority).

```bash
auditwheel repair ./dist/lie_learn-0.0.1.post1-cp35-cp35m-linux_x86_64.whl -w ./manylinux
auditwheel repair ./dist/lie_learn-0.0.1.post1-cp36-cp36m-linux_x86_64.whl -w ./manylinux
auditwheel repair ./dist/lie_learn-0.0.1.post1-cp37-cp37m-linux_x86_64.whl -w ./manylinux
auditwheel repair ./dist/lie_learn-0.0.1.post1-cp38-cp38-linux_x86_64.whl -w ./manylinux
``` 

Open a new terminal window (host environment) and get the running docker `CONTAINER ID`.

```bash
docker ps
```

yields

```
CONTAINER ID        IMAGE                               COMMAND             CREATED             STATUS              PORTS               NAMES
8e2b2c3baa8e        quay.io/pypa/manylinux2014_x86_64   "/bin/bash"         30 minutes ago      Up 30 minutes                           charming_shannon
```

In my case, the `CONTAINER ID` is `8e2b2c3baa8e`. 
In the new terminal window, copy the manylinux wheels from the running container to a folder you'll remember.

```bash
mkdir ~/manylinux
docker cp 8e2b2c3baa8e:/lie_learn/manylinux/lie_learn-0.0.1.post1-cp35-cp35m-manylinux1_x86_64.whl ~/manylinux/
docker cp 8e2b2c3baa8e:/lie_learn/manylinux/lie_learn-0.0.1.post1-cp36-cp36m-manylinux1_x86_64.whl ~/manylinux/
docker cp 8e2b2c3baa8e:/lie_learn/manylinux/lie_learn-0.0.1.post1-cp37-cp37m-manylinux1_x86_64.whl ~/manylinux/
docker cp 8e2b2c3baa8e:/lie_learn/manylinux/lie_learn-0.0.1.post1-cp38-cp38-manylinux1_x86_64.whl ~/manylinux/
```

First do a test by uploading to test pypi. 

```bash
twine upload --repository-url https://test.pypi.org/legacy/ ~/manylinux/*
```

Try downloading and testing `lie_learn` from there before proceeding. 
This is easier said than done. You will need to download all of the dependencies manually then download from test 
pypi without any dependencies using 
`pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --no-deps lie_learn`.
 
Once you know it's working, upload the wheels to pypi with twine.

```bash
twine upload ~/manylinux/*
```

For a bit more info, another useful resource is https://opensource.com/article/19/2/manylinux-python-wheels.