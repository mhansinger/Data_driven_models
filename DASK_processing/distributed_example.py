'''
Example how to use dask with multiple workers and machines
'''

from distributed import Client, LocalCluster

if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client()
    print(client)
