from math import floor

def truncate(obs, act, T):
    """Truncate and downsample the 1kHz act recording of actions and
    the 30Hz obs recording of images to match the period T (s).

    returns obs, act truncated.
    """
    truncto=floor(min(len(obs)*1000*T, len(act))/(T*1000))
    obstrunc=obs[:int(truncto)]
    acttrunc=act[:int(truncto*1000*T):int(1000*T)]
    return obstrunc, acttrunc
    
