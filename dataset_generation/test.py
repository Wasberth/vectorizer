from resize_svg import polygon_to_path
from argparse import ArgumentParser

def main():
    arg_parser = ArgumentParser(description="Resize SVG files.")
    arg_parser.add_argument(
        'source',
        metavar='SOURCE_SVG',
        help='Original SVG to resize'
    )
    args = arg_parser.parse_args()
    return polygon_to_path(open(args.source).read())

if __name__ == "__main__":
    pathed = main()

    with open('pathed.svg', 'w') as f:
        f.write(pathed)