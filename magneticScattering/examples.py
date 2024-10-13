import scatter
import structures
import plot
import matplotlib.pyplot as plt


pol_dict = {'LH': [1, 1, 0, 0], 'LV': [1, -1, 0, 0],
            'CL': [1, 0, 0, 1], 'CR': [1, 0, 0, -1]}


def sk_lattice_dichroism():
    mag_config = structures.skyrmion(10, 10)
    mag_config = structures.tessellate(mag_config, [20, 20], 'hex')

    sample = scatter.Sample(50e-6, [1, 1, 1], mag_config)
    beam_cp = scatter.Beam(scatter.en2wave(706), [25e-6, 25e-6], pol_dict['CR'])
    beam_cl = scatter.Beam(scatter.en2wave(706), [25e-6, 25e-6], pol_dict['CL'])
    s_cp = scatter.Scatter(beam_cp, sample, scatter.Geometry(0, 10))
    s_cl = scatter.Scatter(beam_cl, sample, scatter.Geometry(0, 10))

    plot.structure(sample)
    plot.intensity(s_cl, log=True)
    plot.difference(s_cl, s_cp, log=True)
    plt.show()


if __name__ == '__main__':
    sk_lattice_dichroism()
