#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vidrovr Inc.
# By: Gianni Galbiati

# Standard Libraries

# Scientific Libraries

# Custom Libraries


def error_func(bi, bf, wi, wf, error_type='any'):
    """Checks for different error types at a single location

    A generic error occurs when the location contents don't match in the
        initial and final positions

    A Type 1 error ("false positive") occurs when a piece was placed where
        none was in the original board

    A Type 2 error ("false negative") occurs when a piece was forgotten

    A Type 3 error ("color switch") occurs when a piece was placed with the
        wrong color

    Arguments
    ---------
    bi : str
        original black state; ['0', '1']
    bf : str
        final black state; ['0', '1']
    wi : str
        original white state; ['0', '1']
    wf : str
        final white state; ['0', '1']

    error_type : str
        type of error to check for
        'any' -> any error at all
        '1' -> a Type I error (false positive)
        '2' -> a Type II error (false negative)
        '3' -> a Type III error (color switch)

    Returns
    -------
    int
        0 if there was no error; 1 if there was an error
    """

    if error_type == 'any':

        if (bi == bf) & (wi == wf):
            return 0
        else:
            return 1

    elif error_type == '1':

        if (bi == '0') & (wi == '0') & ((wf == '1') | (bf == '1')):
            return 1
        else:
            return 0

    elif error_type == '2':

        if (bf == '0') & (wf == '0') & ((bi == '1') | (wi == '1')):
            return 1
        else:
            return 0

    elif error_type == '3':

        if ((bi == '1') & (wf == '1')) | ((wi == '1') & (bf == '1')):
            return 1
        else:
            return 0

    else:
        raise ValueError('Error type must be one of "any", "1", "2", or "3"')


def get_errors_per_location(row, error_type,
                            bp_name='Black Position',
                            wp_name='White Position'):

    bp_string = row[bp_name]
    wp_string = row[wp_name]
    bp_string_final = row[bp_name + ' (final)']
    wp_string_final = row[wp_name + ' (final)']

    error = [
        error_func(bi, bf, wi, wf, error_type=error_type)
        for bi, wi, bf, wf in
        zip(bp_string, wp_string, bp_string_final, wp_string_final)
    ]

    return error


def main():
    pass


if __name__ == '__main__':
    main()
