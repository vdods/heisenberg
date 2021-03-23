import heisenberg.self_similar.kh_poster_periodic_orbits
import heisenberg.self_similar.kh_self_similar_pdf_plots
import heisenberg.self_similar.kh_self_similar_more_pdf_plots
import heisenberg.self_similar.kh_z_axis_pdf_plots
import heisenberg.self_similar.kh_z_axis_png_plots

def main ():
    heisenberg.self_similar.kh_poster_periodic_orbits.main()
    heisenberg.self_similar.kh_self_similar_pdf_plots.main()
    heisenberg.self_similar.kh_self_similar_more_pdf_plots.main()
    heisenberg.self_similar.kh_z_axis_pdf_plots.main()
    heisenberg.self_similar.kh_z_axis_png_plots.main()

if __name__ == '__main__':
    print('running everything...')
    main()
    print('done running everything.')
