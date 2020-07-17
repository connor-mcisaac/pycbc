import numpy, logging
from pycbc.waveform import FilterBank
from pycbc.waveform.utils import apply_fseries_time_shift
from pycbc.vetoes.bank_chisq import bank_chisq_from_filters
from pycbc.filter import overlap_cplx, sigmasq, matched_filter_core


def apply_shift(template, dt, df):
    shift = int(1.*df/template.delta_f)
    if df > 0:
        template[-shift:] = 0
    elif df < 0:
        template[:-shift] = 0
    template.roll(shift)
    template = apply_fseries_time_shift(template, dt)
    return template


def segment_snrs(filters, sigmasqs, stilde, psd, low_frequency_cutoff):
    """ This functions calculates the snr of each shifted template against
    the segment
    Parameters
    ----------
    filters: list of FrequencySeries
        The list of bank veto templates filters.
    stilde: FrequencySeries
        The current segment of data.
    psd: FrequencySeries
    low_frequency_cutoff: float
    Returns
    -------
    snr (list): List of snr time series.
    norm (list): List of normalizations factors for the snr time series.
    """
    snrs = []
    norms = []

    for template, sigmasq in zip(filters, sigmasqs):
        # For every template compute the snr against the stilde segment
        snr, _, norm = matched_filter_core(
                template, stilde, h_norm=sigmasq,
                psd=None, low_frequency_cutoff=low_frequency_cutoff)
        # SNR time series stored here
        snrs.append(snr)
        # Template normalization factor stored here
        norms.append(norm)

    return snrs, norms


def template_overlaps(shifted, template, psd, low_frequency_cutoff):
    """ This functions calculates the overlaps between the template and the
    shifted templates.
    Parameters
    ----------
    shifted: List of FrequencySeries
    template: FrequencySeries
    psd: FrequencySeries
    low_frequency_cutoff: float
    Returns
    -------
    overlaps: List of complex overlap values.
    """
    overlaps = []
    sigmasqs = []
    template_ow = template / psd
    for shift in shifted:
        overlap = overlap_cplx(template_ow, shift,
                               low_frequency_cutoff=low_frequency_cutoff,
                               normalized=False)
        shift_sigmasq = sigmasq(shift, psd=psd,
                                low_frequency_cutoff=low_frequency_cutoff)

        norm = numpy.sqrt(1. / template.sigmasq(psd) / shift_sigmasq)
        overlaps.append(overlap * norm)
        sigmasqs.append(shift_sigmasq)
        if (abs(overlaps[-1]) > 0.99):
            errMsg = "Overlap > 0.99 between bank template and filter. "
            errMsg += "This bank template will not be used to calculate "
            errMsg += "bank chisq for this filter template. The expected "
            errMsg += "value will be added to the chisq to account for "
            errMsg += "the removal of this template.\n"
            errMsg += "Masses of filter template: %e %e\n" \
                      %(template.params.mass1, template.params.mass2)
            errMsg += "Masses of bank filter template: %e %e\n" \
                      %(bank_template.params.mass1, bank_template.params.mass2)
            errMsg += "Overlap: %e" %(abs(overlaps[-1]))
            logging.info(errMsg)
    return overlaps, sigmasqs


class SingleDetShiftChisq(object):

    def __init__(self, shift_tuples, f_low, snr_threshold):
        if shift_tuples:
            self.do = True

            self.column_name = 'shift_chisq'
            self.table_dof_name = 'shift_chisq_dof'

            self.shift_tuples = shift_tuples
            self.f_low = f_low
            self.snr_threshold = snr_threshold

            self.dof = len(shift_tuples) * 2

            self._overlaps_cache = {}
        else:
            self.do = False
        
    def cache_overlaps(self, template, psd):
        key = (id(template.params), id(psd))
        if key not in self._overlaps_cache:
            logging.info("...Calculate bank veto overlaps")
            shifted = [apply_shift(template.copy(), s[0], s[1])
                       for s in self.shift_tuples]
            o, s = template_overlaps(shifted, template, psd, self.f_low)
            self._overlaps_cache[key] = (o, s)
        return self._overlaps_cache[key]

    def values(self, template, psd, stilde, snrv, norm, indices):
        if not self.do:
            return None, None

        logging.info("...Doing bank veto")        
        chisq = numpy.ones(len(snrv))
        dof = numpy.ones(len(snrv))

        snr = numpy.abs(snrv * norm)
        above = numpy.where(snr >= self.snr_threshold)[0]

        overlaps, sigmasqs = self.cache_overlaps(template, psd)
        
        shifted = [apply_shift(template.copy(), s[0], s[1])
                   for s in self.shift_tuples]
        shift_veto_snrs, shift_veto_norms = segment_snrs(shifted, sigmasqs,
                                                         stilde, psd, self.f_low)
        
        chisq[above] = bank_chisq_from_filters(snrv[above], norm, shift_veto_snrs,
                                               shift_veto_norms, overlaps, indices[above])
        dof[above] = numpy.repeat(self.dof, len(above))
        
        return chisq, dof

    @staticmethod
    def insert_option_group(parser):
        group = parser.add_argument_group("Shift Chisq")
        group.add_argument("--shift-chisq-snr-threshold", type=float,
            help="Minimum SNR threshold to use shift chisq")
        group.add_argument("--shift-chisq-dt-df", type=str, nargs='+',
            help='Time and Frequency offsets of the sine-Gaussiansbank template'
                 + ' to use, format "xdt:df".')

    @classmethod
    def from_cli(cls, args, f_low):
        shift_tuples = []
        for shift_string in args.shift_chisq_dt_df:
            dt, df = shift_string[1:].split(':')
            shift_tuples.append((float(dt), float(df)))
        return cls(shift_tuples, f_low, args.shift_chisq_snr_threshold)
