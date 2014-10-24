# Add our library code directory to the import search path.  This allows things like
# `import fourier` to work.  It relies on the relative filesystem position of this
# file to the `library` directory.
import sys
sys.path.append('library')

def main ():
    import fourier
    fourier.Transform.test_partial_inverse()

if __name__ == "__main__":
    main()
