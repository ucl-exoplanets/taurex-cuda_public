


def compute_h2():
    
    code = """

    double wave = 1e8/_wn;

    double wave_2 = 1.0/(wave*wave);
    double wave_4 = wave_2*wave_2;

    double result = ((8.14E-13)*(wave_4)*(1+(1.572E6)*(wave_2)+(1.981E12)*(wave_4)))*1E-4;
    """

    return code
def compute_he():
    code = """

    double wave = 1e8/_wn;

    double wave_2 = 1.0/(wave*wave);
    double wave_4 = wave_2*wave_2;

    double result = ((5.484E-14)*(wave_4)*(1+(2.44E5)*(wave_2)))*1E-4;
    """

    return code


def compute_ray_sigma(n=0.0,n_air = 2.6867805e25,king=1.0,):
    from taurex.constants import PI
    code = f"""
    double _wl = (10000.0/_wn)*1e-6;
    double _wl_2 = 1/(_wl*_wl);
    double _n = {n};
    double _nair = {n_air};
    double _king = {king};
    double n_factor = (_n*_n - 1)/(_nair*(_n*_n + 2.0));


    double result = 24.0*({PI**3.0})*_king*(n_factor*n_factor)/(_wl*_wl*_wl*_wl);
    """

    return code

_molecule_func_code = {
        'He' : compute_he(),
        'H2' : compute_h2(),
        'N2' : compute_ray_sigma(
                            n='1. + (6498.2 + (307.43305e12)/(14.4e9 - _wn*_wn))*1.0e-8',
                            king='1.034+3.17e-12*_wn*_wn'),
        'O2' : compute_ray_sigma(
                            n='1 + 1.181494e-4 + 9.708931e-3/(75.4-_wl_2)',
                            king = 1.096),
        'CO2' : compute_ray_sigma(
                        n='1 + 6.991e-2/(166.175-_wl_2)+1.44720e-3/(79.609-_wl_2)+6.42941e-5/(56.3064-_wl_2)'
                                +'+5.21306e-5/(46.0196-_wl_2)+1.46847e-6/(0.0584738-_wl_2)',king = 1.1364), 
        'CH4' : compute_ray_sigma(
                                n='1 + 1.e-8*(46662. + 4.02*1.e-6*pow(1/((10000/_wn)*1.e-4),2.0))'),
        'CO' : compute_ray_sigma(
                                n='1.0 + 32.7e-5 * (1.0 + 8.1e-3 / (100000000.0/(_wn*_wn)))',king=1.016),
        'NH3' : compute_ray_sigma(
                                n='1 + 37.0e-5 * (1. + 12.0e-3 / (100000000.0/(_wn*_wn)))'),
        'H2O' : (lambda ns_air='(1 + (0.05792105/(238.0185 - 1/(100000000.0/(_wn*_wn))) + 0.00167917/(57.362- 1.0/(100000000.0/(_wn*_wn)))))',delta=0.17:
                        compute_ray_sigma(n=f'0.85 * ({ns_air} - 1.) + 1',king=f'(6.+3.*{delta})/(6.-7.*{delta})'))(),
            


}