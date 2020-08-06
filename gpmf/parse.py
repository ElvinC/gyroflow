# The MIT License (MIT)
# Copyright (c) 2014 Eero af Heurlin
# https://github.com/rambo/python-gpmf

#!/usr/bin/env python3
"""Parses the FOURCC data in GPMF stream into fields"""
import struct

import construct
import dateutil.parser

TYPES = construct.Enum(
    construct.Byte,
    int8_t=ord(b'b'),
    uint8_t=ord(b'B'),
    char=ord(b'c'),
    int16_t=ord(b's'),
    uint16_t=ord(b'S'),
    int32_t=ord(b'l'),
    uint32_t=ord(b'L'),
    float=ord(b'f'),
    double=ord(b'd'),
    fourcc=ord(b'F'),
    uuid=ord(b'G'),
    int64_t=ord(b'j'),
    uint64_t=ord(b'J'),
    Q1516=ord(b'q'),
    Q3132=ord(b'Q'),
    utcdate=ord(b'U'),
    complex=ord(b'?'),
    nested=0x0,
)

FOURCC = construct.Struct(
    "key" / construct.Bytes(4),
    "type" / construct.Byte,
    "size" / construct.Byte,
    "repeat" / construct.Int16ub,
    "data" / construct.Aligned(4, construct.Bytes(construct.this.size * construct.this.repeat))
)


def parse_value(element):
    """Parses element value"""
    type_parsed = TYPES.parse(bytes([element.type]))
    #print("DEBUG: type_parsed={}, element.repeat={}, element.size={}, len(element.data): {}".format(type_parsed, element.repeat, element.size, len(element.data)))

    # Special cases
    if type_parsed == 'char' and element.key == b'GPSU':
        return parse_goprodate(element)
    if type_parsed == 'utcdate':
        return parse_goprodate(element)

    # Basic number types
    struct_key = None
    struct_repeat = element.repeat
    if type_parsed == 'int32_t':
        struct_key = 'l'
        # It seems gopro is "creative" with grouped values and size vs repeat...
        if element.size > 4:
            struct_repeat = int(element.repeat * (element.size / 4))
    if type_parsed == 'uint32_t':
        struct_key = 'L'
        if element.size > 4:
            struct_repeat = int(element.repeat * (element.size / 4))

    if type_parsed == 'int16_t':
        struct_key = 'h'
        if element.size > 2:
            struct_repeat = int(element.repeat * (element.size / 2))
    if type_parsed == 'uint16_t':
        struct_key = 'H'
        if element.size > 2:
            struct_repeat = int(element.repeat * (element.size / 2))

    if type_parsed == 'float':
        struct_key = 'f'
        if element.size > 4:
            struct_repeat = int(element.repeat * (element.size / 4))

    if not struct_key:
        raise ValueError("{} does not have value parser yet".format(type_parsed))

    struct_format = ">{}".format(''.join([struct_key for x in range(struct_repeat)]))
    #print("DEBUG: struct_format={}".format(struct_format))
    try:
        value_parsed = struct.unpack(struct_format, element.data)
    except struct.error as e:
        #print("ERROR: {}".format(e))
        #print("DEBUG: struct_format={}, data (len: {}) was: {}".format(struct_format, len(element.data), element.data))
        raise ValueError("Struct unpack failed: {}".format(e))

    # Single value
    if len(value_parsed) == 1:
        return value_parsed[0]
    # Grouped values
    if len(value_parsed) > element.repeat:
        n = int(len(value_parsed) / element.repeat)
        return [value_parsed[i:i + n] for i in range(0, len(value_parsed), n)]
    return list(value_parsed)


def parse_goprodate(element):
    """Parses the gopro date string from element to Python datetime"""
    goprotime = element.data.decode('UTF-8')
    return dateutil.parser.parse("{}-{}-{}T{}:{}:{}Z".format(
        2000 + int(goprotime[:2]),  # years
        int(goprotime[2:4]),        # months
        int(goprotime[4:6]),        # days
        int(goprotime[6:8]),        # hours
        int(goprotime[8:10]),       # minutes
        float(goprotime[10:])       # seconds
    ))


def recursive(data, parents=tuple()):
    """Recursive parser returns depth-first traversing generator yielding fields and list of their parent keys"""
    elements = construct.GreedyRange(FOURCC).parse(data)
    for element in elements:
        if element.type == 0:
            subparents = parents + (element.key,)
            for subyield in recursive(element.data, subparents):
                yield subyield
        else:
            yield (element, parents)


if __name__ == '__main__':
    import sys
    from extract import get_gpmf_payloads_from_file
    payloads, parser = get_gpmf_payloads_from_file(sys.argv[1])
    for gpmf_data, timestamps in payloads:
        for element, parents in recursive(gpmf_data):
            try:
                value = parse_value(element)
            except ValueError:
                value = element.data
            print("{} {} > {}: {}".format(
                timestamps,
                ' > '.join([x.decode('ascii') for x in parents]),
                element.key.decode('ascii'),
                value
            ))
