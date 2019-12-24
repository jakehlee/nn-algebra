import sys, os
import numpy as np
import csv
from scipy import spatial
import matplotlib.pyplot as plt


def usage():
    print("Usage: python plot_cosine.py feats.csv model dataset layer [tl, tr, c, bl, br] min=auto, max=auto")
    sys.exit(0)

def cosine_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

if __name__ == "__main__":
    if len(sys.argv) != 6 and len(sys.argv) != 8:
        usage()

    if sys.argv[5] not in ['tl', 'tr', 'c', 'bl', 'br']:
        usage()

    csvfile = sys.argv[1]
    model = sys.argv[2]
    dataset = sys.argv[3]
    layer = sys.argv[4]
    comp = sys.argv[5]

    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        all_feats = np.array(list(reader))

    max_r = 0
    max_c = 0
    for row in all_feats:
        name = row[0]
        r = int(name.split('_')[0])
        c = int(name.split('_')[1])
        if r > max_r:
            max_r = r
        if c > max_c:
            max_c = c

    all_vects = all_feats[:,1:]
    feat_len = len(all_vects[0])

    feat_array = np.zeros((max_r+1, max_c+1, feat_len))
    cs_array = np.zeros((max_r+1, max_c+1))
    
    PROB = False
    COSINE = False
    if feat_len == 1:
        PROB = True
    else:
        COSINE = True

    if COSINE:
        for row in all_feats:
            name = row[0]
            feat = np.array(row[1:]).astype(float)
            r = int(name.split('_')[0])
            c = int(name.split('_')[1])
            feat_array[r,c] = feat

        if comp == 'tl':
            compvect = feat_array[0, 0]
        elif comp == 'tr':
            compvect = feat_array[0, max_c]
        elif comp == 'c':
            compvect = feat_array[max_r // 2, max_c // 2]
        elif comp == 'bl':
            compvect = feat_array[max_r, 0]
        elif comp == 'br':
            compvect = feat_array[max_r, max_c]

        for i in range(max_r+1):
            for j in range(max_c+1):
                cs_array[i,j] = cosine_sim(compvect, feat_array[i,j])

    elif PROB:
        for row in all_feats:
            name = row[0]
            feat = float(row[1])
            r = int(name.split('_')[0])
            c = int(name.split('_')[1])
            cs_array[r,c] = feat

    print("min:", np.min(cs_array))
    print("max:", np.max(cs_array))

    fig, ax = plt.subplots()
    pos = ax.imshow(cs_array, cmap='hot')

    if COSINE:
        ax.set_title('Cosine Similarity of Shifted Object Features\n{} {} on {}, comp. to {}'.format(model, layer, dataset, comp))
    if PROB:
        ax.set_title('Correct Class Prob. of Shifted Object Features\n {} on {}'.format(model, dataset))

    ax.set_xlabel('Horizontal Shift in Pixels')
    ax.set_ylabel('Vertical Shift in Pixels')
    if len(sys.argv) == 8:
        cmap_min = float(sys.argv[6])
        cmap_max = float(sys.argv[7])
        pos.set_clim(cmap_min, cmap_max)

    fig.colorbar(pos)
    plt.savefig('{}-{}-{}-{}-plot.png'.format(model, layer, dataset, comp))
