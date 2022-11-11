import sys
from elftools.elf.elffile import ELFFile

def process_file(filename):
    with open(filename, 'rb') as f:
        elffile = ELFFile(f)

        dynsym = elffile.get_section_by_name('.dynsym')
        dynsym_offset = dynsym.header.sh_addr
        dynsym_idx = []  # addresses of Elf64_Sym
        for idx, sb in enumerate(dynsym.iter_symbols()):
            bind = sb.entry.st_info.bind
            if "UNIQUE" in bind or "LOOS" in bind:
                print("Found UNIQUE symbol: ", sb.name[:60])
                dynsym_idx.append(dynsym_offset + idx * 24)  # 24=sizeof(Elf64_Sym)

    print(f"Patching {len(dynsym_idx)} symbols ...")
    with open(filename, 'rb+') as f:
        for sym_idx in dynsym_idx:
            f.seek(sym_idx + 4)  # 4=sizeof(st_name)
            old = ord(f.read(1))
            assert old // 16 == 10, hex(old)  # STB_GNU_UNIQUE==10
            f.seek(sym_idx + 4)
            f.write(bytes([old % 16 + 2 * 16]))  # STB_WEAK==2
            f.write(bytes([2]))  # STV_HIDDEN=2

process_file(sys.argv[1])