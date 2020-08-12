# The MIT License (MIT)
# Copyright (c) 2014 Eero af Heurlin
# https://github.com/rambo/python-gpmf

#!/usr/bin/env python3
import hachoir.parser
from hachoir.field import MissingField
from hachoir.field.string_field import String


def get_raw_content(met):
    """Reads the raw bytes from the stream for this atom/field"""
    if hasattr(met, 'stream'):
        stream = met.stream
    else:
        stream = met.parent.stream
    return stream.read(met.absolute_address, met.size)


def get_gpmf_payloads_from_file(filepath):
    """Get payloads from file, returns a tuple with the payloads iterator and the parser instance"""
    parser = hachoir.parser.createParser(filepath)
    return (get_payloads(find_gpmd_stbl_atom(parser)), parser)


def get_gpmf_payloads(parser):
    """Shorthand for finding the GPMF atom to be passed to get_payloads"""
    return get_payloads(find_gpmd_stbl_atom(parser))


def get_payloads(stbl):
    """Get payloads by chunk from stbl, with timing info"""
    # Locate needed subatoms
    try:
        for subatom in stbl:
            tag = subatom['tag']
            if tag.value == 'stsz':
                stsz = subatom['stsz']
            if tag.value == 'stco':
                stco = subatom['stco']
            if tag.value == 'stts':
                stts = subatom['stts']
    except TypeError:
        raise Exception("Problem parsing metadata")

    # Generate start and end timestamps for all chunks
    timestamps = []
    for idx in range(stts['count'].value):
        sample_delta = stts["sample_delta[{}]".format(idx)].value
        for idx2 in range(stts["sample_count[{}]".format(idx)].value):
            if idx == 0 and idx2 == 0:
                sampletimes = (0, sample_delta)
            else:
                sampletimes = (timestamps[-1][1], timestamps[-1][1] + sample_delta)
            timestamps.append(sampletimes)

    # Read chunks, yield with timing data
    num_samples = stsz['count'].value
    for idx in range(num_samples):
        offset = stco["chunk_offset[{}]".format(idx)].value
        size = stsz["sample_size[{}]".format(idx)].value
        data = stbl.stream.read(offset * 8, size * 8)[1]
        yield (data, timestamps[idx])


def get_stream_data(stbl):
    """Get raw payload bytes from stbl atom offsets"""
    ret_bytes = b''
    for payload in get_payloads(stbl):
        ret_bytes += payload[0]
    return ret_bytes


def find_gpmd_stbl_atom(parser):
    """Find the stbl atom"""
    minf_atom = find_gpmd_minf_atom(parser)
    if not minf_atom:
        return None
    try:
        for minf_field in minf_atom:
            tag = minf_field['tag']
            if tag.value != 'stbl':
                continue
            return minf_field['stbl']
    except MissingField:
        pass


def find_gpmd_minf_atom(parser):
    """Find minf atom for GPMF media"""
    def recursive_search(atom):
        try:
            subtype = atom['hdlr/subtype']
            if subtype.value == 'meta':
                meta_atom = atom.parent
                # print(meta_atom)
                for subatom in meta_atom:
                    tag = subatom['tag']
                    if tag.value != 'minf':
                        continue
                    minf_atom = subatom['minf']
                    #print("  {}".format(minf_atom))
                    for minf_field in minf_atom:
                        tag = minf_field['tag']
                        #print("    {}".format(tag))
                        if tag.value != 'gmhd':
                            continue
                        if b'gpmd' in minf_field['data'].value:
                            return minf_atom
        except MissingField:
            pass
        try:
            for x in atom:
                ret = recursive_search(x)
                if ret:
                    return ret
        except KeyError as e:
            pass
        return None
    return recursive_search(parser)


def recursive_print(input):
    """Recursively print hachoir parsed state"""
    print(repr(input))
    if isinstance(input, String):
        print("  {}".format(input.display))
    try:
        for x in input:
            recursive_print(x)
    except KeyError as e:
        pass


if __name__ == '__main__':
    import sys
    parser = hachoir.parser.createParser(sys.argv[1])
    with open(sys.argv[2], 'wb') as fp:
        fp.write(
            get_stream_data(
                find_gpmd_stbl_atom(parser)
            )
        )
