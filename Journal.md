### @2018-12-15
#### How to install OpenFOAM on Arch linux
1. Enable the Loop Module

check:
```shell
# lsmod | grep loop
```
If the loop module doesn't loaded, run the code below:
```shell
# tee /etc/modules-load.d/loop.conf <<< "loop"
# modprobe loop
```

2. Install Docker
```shell
# pacman -S docker
```

3. start/enable/stop Docker
```shell
# systemctl start docker.service
 
# systemctl enable docker.service

# systemctl stop docker.service
```
Then we check the docker info with
```shell
# docker info
```

4. Grand user with docker group
```shell
# groupadd docker
 
# gpasswd -a user docker [replace user with your username]
``` 

5. Download OpenFoam scripts to load and run OpenFoam
* Download two scripts
[installOpenFOAM](https://sourceforge.net/projects/openfoamplus/files/v1806/installOpenFOAM)
[startOpenFOAM](https://sourceforge.net/projects/openfoamplus/files/v1806/startOpenFOAM)

* Make two scripts executable
```shell
chmod +x installOpenFOAM 
chmod +x startOpenFOAM 
```

* Run install script
```shell
./installOpenFOAM
```

* Run start script
```shell
./startOpenFOAM
```

6. Test
```shell
mkdir -p $FOAM_RUN 
run 
cp -r $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity . 
cd cavity 
blockMesh 
icoFoam 
```
7. Copy tutorial
``` shell
mkdir -p $FOAM_RUN 
cp -r $FOAM_TUTORIALS $FOAM_RUN
```

