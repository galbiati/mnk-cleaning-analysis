#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vidrovr Inc.
# By: Gianni Galbiati

# Standard Libraries

# Scientific Libraries
import numpy as np

# Custom Libraries


hstarts = [i for row in range(4) for i in range(9*row, 9*row + 6, 1)]
vstarts = list(range(9))
ddstarts = list(range(6))
dustarts = list(range(4, 9))


def _add_position_strings(bp, wp):
    """Add string representations of position channels"""
    return ''.join([str(int(b) + int(w)) for b, w in zip(bp, wp)])


def _count_feature(bp, wp, feature):
    """Count occurrences of feature in all orientations"""
    # Get the overall occupancy of position
    p = _add_position_strings(bp, wp)

    # Initialize count matrices
    bcounts = np.zeros(36, dtype=np.uint8)
    wcounts = np.zeros(36, dtype=np.uint8)

    # Helper function to detect matchs in different orientations
    def _orient_count(start, increment):

        end = start + 4 * increment

        for orientation in [1, -1]:
            total_match = p[start:end:increment] == feature[::orientation]

            if not total_match:
                # If the complete position is not the same as feature,
                #    it means that some locations that should have been
                #    empty were not, so just continue
                continue

            black_match = bp[start:end:increment] == feature[::orientation]

            if black_match:
                bcounts[start:end:increment] += 1

                # If we found a black_match, no need to check white position
                break

            white_match = wp[start:end:increment] == feature[::orientation]

            if white_match:
                wcounts[start:end:increment] += 1

        return None

    # For every horizontal starting value
    for start in hstarts:
        _orient_count(start, 1)

    # Etc
    for start in vstarts:
        _orient_count(start, 9)

    for start in dustarts:
        _orient_count(start, 8)

    for start in ddstarts:
        _orient_count(start, 10)

    return bcounts, wcounts


def count_all_features(row):
    features = ['1100', '1010', '1001', '1110', '1101', '1111']
    bp = row['Black Position']
    wp = row['White Position']

    output_dict = {}
    for feature in features:
        bcount, wcount = _count_feature(bp, wp, feature)

        output_dict[feature + 'b'] = bcount
        output_dict[feature + 'w'] = wcount

    return output_dict


def _detect_type_2_error(bi, bf, wi, wf):
    final_empty = ((bf == '0') and (wf == '0'))
    original_not_empty = ((bi == '1') or (wi == '1'))

    return int(original_not_empty and final_empty)


def _detect_type_3_error(bi, bf, wi, wf):
    b2w = ((bi == '1') and (wf == '1'))
    w2b = ((wi == '1') and (bf == '1'))

    return int(b2w or w2b)


def count_all_errors(row):
    bpi = row['Black Position']
    bpf = row['Black Position (final)']

    wpi = row['White Position']
    wpf = row['White Position (final)']

    type_2_errors = [
        _detect_type_2_error(bi, bf, wi, wf)
        for bi, bf, wi, wf in zip(bpi, bpf, wpi, wpf)
    ]

    type_3_errors = [
        _detect_type_3_error(bi, bf, wi, wf)
        for bi, bf, wi, wf in zip(bpi, bpf, wpi, wpf)
    ]

    return {'Type 2': type_2_errors, 'Type 3': type_3_errors}


def main():
    pass


if __name__ == '__main__':
    main()
