from dgl.data import GINDataset

data = GINDataset('MUTAG', self_loop=True)

g, l = data[0]

print(g)
print(l)