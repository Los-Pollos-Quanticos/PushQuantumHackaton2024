from config import hamiltonian_matrix

import numpy as np
import matplotlib.pyplot as plt

def compute_gershgorin_disks(H):
    diagonal_elements = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(np.real(diagonal_elements))

    gershgorin_disks = [(i, diagonal_elements[i], radii[i]) for i in range(H.shape[0])]

    gershgorin_disks.sort(key=lambda d: d[1] - d[2])
    
    return gershgorin_disks

def disks_overlap(d1, d2):
    center1, radius1 = d1[1], d1[2]
    center2, radius2 = d2[1], d2[2]
    return abs(center1 - center2) <= (radius1 + radius2)

def select_disks(gershgorin_disks):
    selected_indices = set()
    selected_disks = []

    for i in range(len(gershgorin_disks)):
        if gershgorin_disks[i][0] in selected_indices:
            continue
        
        current_disk = gershgorin_disks[i]
        selected_indices.add(current_disk[0])
        selected_disks.append(current_disk)

        for j in range(len(gershgorin_disks)):
            if gershgorin_disks[j][0] in selected_indices:
                continue
            if disks_overlap(current_disk, gershgorin_disks[j]):
                selected_indices.add(gershgorin_disks[j][0])
                selected_disks.append(gershgorin_disks[j])

        if len(selected_disks) >= 5:
            break
    
    return selected_disks

def plot_disks(selected_disks, diagonal_elements, radii):
    fig, ax = plt.subplots(figsize=(10, 6))

    for index, center, radius in selected_disks:
        circle = plt.Circle((center, 0), radius, color='blue', alpha=0.1, linewidth=1.5)
        ax.add_patch(circle)
        
        plt.scatter(center, 0, color='black', marker='o', s=30)
        plt.text(center, 0.05, f"{index}", ha='center', fontsize=9, color='black')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis (ignored)")
    plt.title("Selected Gershgorin Disks")
    plt.xlim(min(diagonal_elements - radii) - 1, max(diagonal_elements + radii) + 1)
    plt.ylim(-max(radii) - 1, max(radii) + 1)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    H = hamiltonian_matrix
    gershgorin_disks = compute_gershgorin_disks(H)
    selected_disks = select_disks(gershgorin_disks)

    for (i, center, radius) in selected_disks:
        hamiltonian_row = H[i]
        binary_representation = format(i, f"0{int(np.log2(H.shape[0]))}b")
        print(f"Disk {i+1} (Row {i}): Center = {center:.4f}, Radius = {radius:.4f}, Range = [{center - radius:.4f}, {center + radius:.4f}]")
        print(f"  Corresponding Hamiltonian Row (Index {i}): {hamiltonian_row}")
        print(f"  Computational Basis State (Binary): |{binary_representation}>")
        print()

    diagonal_elements = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(np.real(diagonal_elements))
    plot_disks(selected_disks, diagonal_elements, radii)
