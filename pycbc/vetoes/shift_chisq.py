import numpy, logging
from pycbc.waveform import FilterBank
from pycbc.waveform.utils import apply_fseries_time_shift
from pycbc.filter import overlap_cplx, sigma, matched_filter_core
from pycbc.types import real_same_precision_as, TimeSeries


def apply_shift(template, dt, df):
    shift = int(1.*df/template.delta_f)
    if df > 0:
        template[-shift:] = 0.
    elif df < 0:
        template[:-shift] = 0.
    template.roll(shift)
    apply_fseries_time_shift(template, dt, copy=False)
    return template


def get_orthogonal(template, template_sigma, base, base_sigma, overlap):
    norm = ((1. - overlap * overlap.conj()).real) ** 0.5
    norm_temp = template / template_sigma
    norm_base = base / base_sigma
    ortho = (norm_temp - overlap * norm_base) / norm
    return ortho


def get_chisq_from_orthogonal(ortho_templates, stilde, psd, f_low):

    chisq = None
    for temp in ortho_templates:

        snr, _, norm = matched_filter_core(temp, stilde, psd=psd,
                                           low_frequency_cutoff=f_low,
                                           h_norm=1.)
        if chisq is None:
            chisq = (snr * norm).squared_norm()
        else:
            chisq += (snr * norm).squared_norm()

    dof = len(ortho_templates) * 2

    return chisq, dof


def get_chisq_from_shifts(shift_tuples, template, stilde, psd, f_low):
    
    template_sigma = sigma(template, psd=psd, low_frequency_cutoff=f_low)
    ortho_templates = []
    for shift in shift_tuples:
        shifted = apply_shift(template.copy(), shift[0], shift[1])
        shifted_sigma = sigma(shifted, psd=psd, low_frequency_cutoff=f_low)

        overlap = overlap_cplx(template, shifted, low_frequency_cutoff=f_low,
                               normalized=False, psd=psd)
        overlap *= 1. / template_sigma / shifted_sigma

        ortho = get_orthogonal(shifted, shifted_sigma,
                               template, template_sigma,
                               overlap)

        ortho_templates.append(ortho)

    chisq, dof = get_chisq_from_orthogonal(ortho_templates, stilde, psd, f_low)

    return chisq, dof


class SingleDetShiftChisq(object):

    def __init__(self, shift_tuples, f_low):
        if shift_tuples:
            self.do = True

            self.column_name = 'shift_chisq'
            self.table_dof_name = 'shift_chisq_dof'

            self.shift_tuples = shift_tuples
            self.f_low = f_low

            self.dof = len(shift_tuples) * 2

            self._overlaps_cache = {}
        else:
            self.do = False


    def get_ortho(self, template, psd):
        shifted = [apply_shift(template.copy(), s[0], s[1])
                   for s in self.shift_tuples]

        key = (id(template.params), id(psd))
        if key not in self._overlaps_cache:
            overlaps = []
            sigmas = []
            for shift in shifted:
                shift_sigma = sigma(shift, psd=psd, low_frequency_cutoff=self.f_low)
                overlap = overlap_cplx(template, shift, psd=psd, low_frequency_cutoff=self.f_low,
                                       normalized=False)
                overlap = overlap / shift_sigma / numpy.sqrt(template.sigmasq(psd))

                overlaps.append(overlap)
                sigmas.append(shift_sigma)
            self._overlaps_cache[key] = (overlaps, sigmas)
        else:
            overlaps, sigmas = self._overlaps_cache[key]

        orthos = []
        for j in range(len(shifted)):
            ortho = get_orthogonal(shifted[j], sigmas[j],
                                   template, numpy.sqrt(template.sigmasq(psd)),
                                   overlaps[j])
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
        group.add_argument("--shift-chisq-dt-df", type=str, nargs='+',
            help='Time and Frequency offsets of the sine-Gaussiansbank template'
                 + ' to use, format "xdt:df".')

    @classmethod
    def from_cli(cls, args, f_low):
        shift_tuples = []
        for shift_string in args.shift_chisq_dt_df:
            dt, df = shift_string[1:].split(':')
            shift_tuples.append((float(dt), float(df)))
        return cls(shift_tuples, f_low)
