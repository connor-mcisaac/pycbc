import numpy, logging
from pycbc.waveform import FilterBank
from pycbc.waveform.utils import apply_fseries_time_shift
from pycbc.filter import overlap_cplx, sigma, matched_filter_core
from pycbc.types import real_same_precision_as, TimeSeries


def apply_shift(template, dt, df):
    shift = int(1.*df/template.delta_f)
    if df > 0:
        template[-shift:] = 0
    elif df < 0:
        template[:-shift] = 0
    template.roll(shift)
    template = apply_fseries_time_shift(template, dt)
    return template


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
    sigmas = []
    for shift in shifted:
        overlap = overlap_cplx(template, shift, psd=psd,
                               low_frequency_cutoff=low_frequency_cutoff,
                               normalized=False)
        shift_sigma = sigma(shift, psd=psd,
                            low_frequency_cutoff=low_frequency_cutoff)

        norm = 1. / numpy.sqrt(template.sigmasq(psd)) / shift_sigma
        overlaps.append(overlap * norm)
        sigmas.append(shift_sigma)
    return overlaps, sigmas


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


    def get_ortho(self, template, psd):
        shifted = [apply_shift(template.copy(), s[0], s[1])
                   for s in self.shift_tuples]

        key = (id(template.params), id(psd))
        if key not in self._overlaps_cache:
            o, s = template_overlaps(shifted, template, psd, self.f_low)
            self._overlaps_cache[key] = (o, s)
        else:
            o, s = self._overlaps_cache[key]

        orthos = []
        for j in range(len(shifted)):
            norm = ((1 - o[j] * o[j].conj()).real) ** 0.5
            ortho = (shifted[j] / s[j]
                     - o[j] * template / numpy.sqrt(template.sigmasq(psd)))
            ortho /= norm
            orthos.append(ortho)

        return orthos
        

    def values(self, template, psd, stilde, snrv, norm, indices):
        if not self.do:
            return None, None

        logging.info("...Doing shift veto")
        orthos = self.get_ortho(template, psd)

        chisq = numpy.zeros(len(snrv), dtype=real_same_precision_as(stilde))
        dof = numpy.repeat(self.dof, len(snrv))

        for j in range(len(orthos)):

            ortho_snr, _, ortho_norm = matched_filter_core(orthos[j], stilde, psd=None,
                                                           low_frequency_cutoff=self.f_low,
                                                           h_norm=1.)
            if indices is not None:
                ortho_snr = ortho_snr.take(indices)

            chisq += (ortho_snr * ortho_norm).squared_norm()
        
        if indices is None:
            chisq = TimeSeries(chisq, delta_t=snrv.delta_t,
                               epoch=snrv.start_time, copy=False)

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
