# LEARN_Assignment1
Assignment 1 of Learning over massive data course
#### Cluster address

```
didavhpc01.dais.unive.it
```

#### How to transfer files from local to cluster

```bash
scp -r /home/vito/LEARN_Assignment1/ 904120@didavhpc01.dais.unive.it:/home/904120
```

#### How to run the code in ~/LEARN_Assignment1/build$ : folders before perf my change or not necessary

```bash
/usr/lib/linux-tools/5.15.0-101-generic/perf stat -d -B -o perf_out.txt ./LEARN_Assignment1 ../datasets/p2p-Gnutella25.txt 12
```
```bash
srun -v -I -c 20 ./LEARN_Assignment1 ../datasets/p2p-Gnutella25.txt 20
```