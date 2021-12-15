#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def generate_fake_data(nchan, ntime, period, dm, noise_frac=.1):
    """Create an (nchan x ntime) array with simulated pulsar signal
       nchan (int): Number of frequency channels
       ntime (int): Number of time samples
       period (float): Pulsar period in units of time samples
       dm (float): DM in units of time samples across band
    """
    assert (ntime / period) % 1 == 0, "for now need integer periods in ntime"
    data = np.zeros((nchan, ntime), dtype=np.uint8)
    # get indices where pulsar is "on" in reference frequency channel (= highest frequency = last channel)
    nperiod = int(ntime / period) + 1
    pulsar_indices_highest_freq = (np.arange(nperiod) * period).astype(int)
    # calculate DM shifts for each channel (no shifts at highest channel, positive towards lower channels)
    for chan in range(nchan):
        # DM shift of this channel
        shift = dm * ((nchan - 1 - chan) / (nchan - 1)) ** 2
        # pulsar location in this channel
        pulsar_indices = (pulsar_indices_highest_freq + shift) % ntime  # without % ntime if not integer nr of periods in ntime
        data[chan, pulsar_indices.astype(int)] = 1

    # add some random noise, i.e. set given fraction of data to 1
    data += (np.random.random(data.shape) < noise_frac)
    return data


def generate_fake_data_2(nchan, ntime, period, dm, noise_frac=.1):
    """Create an (nchan x ntime) array with simulated pulsar signal
       nchan (int): Number of frequency channels
       ntime (int): Number of time samples
       period (float): Pulsar period in units of time samples
       dm (float): DM in units of time samples across band
    """
    # assert (ntime / period) % 1 == 0, "for now need integer periods in ntime"
    data = np.zeros((nchan, ntime), dtype=np.uint8)
    # get indices where pulsar is "on" in reference frequency channel (= highest frequency = last channel)
    nperiod = int(ntime / period) + 1
    pulsar_indices_highest_freq = (np.arange(nperiod) * period).astype(int)
    # calculate DM shifts for each channel (no shifts at highest channel, positive towards lower channels)
    for chan in range(nchan):
        # DM shift of this channel
        shift = dm * ((nchan - 1 - chan) / (nchan - 1)) ** 2
        # pulsar location in this channel
        # print(shift)
        pulsar_indices = (pulsar_indices_highest_freq + shift) #% ntime  # without % ntime if not integer nr of periods in ntime
        # print(pulsar_indices)
        pulsar_indices = pulsar_indices.astype(int)
        pulsar_indices = list(pulsar_indices)
        while pulsar_indices[-1]>=ntime:
            pulsar_indices.pop()

        while pulsar_indices[0]>=period:
            pulsar_indices = [pulsar_indices[0]-period] + pulsar_indices

        pulsar_indices = np.array(pulsar_indices)
        data[chan, pulsar_indices] = 1

    # add some random noise, i.e. set given fraction of data to 1
    data += (np.random.random(data.shape) < noise_frac)
    return data


if __name__ == '__main__':
    nchan = 64
    ntime = 64

    period_real = 6
    dm_real = 15.5
    noise_frac = .0

    # generate raw data, optionally with noise
    data = generate_fake_data_2(nchan, ntime, period_real, dm_real, noise_frac)

    # # dedispersion step
    # dms = range(15, 35)
    # ndm = len(dms)
    # timeseries = np.zeros((ndm, ntime), dtype=np.uint8)
    # for dm_index, dm in enumerate(dms):
    #     for channel_index, channel in enumerate(data):
    #         # DM shift of this channel
    #         shift = int(dm * ((nchan - 1 - channel_index) / (nchan - 1)) ** 2)
    #         timeseries[dm_index] += np.roll(channel, -shift)

    # # fft step
    # power_spectrum = np.abs(np.fft.fft(timeseries)) ** 2
    # # remove zero freq (is not related to real pulsar signal, but has high amplitude)
    # power_spectrum[..., 0] = 0

    # # find point with highest amplitude in (DM, fourier freq) space
    # df = 1. / ntime
    # freqs = np.arange(ntime) * df
    # dm_index, freq_index = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)

    # dm_guess = dms[dm_index]
    # freq_guess = freqs[freq_index]
    # period_guess = 1. / freq_guess
    # print('Param\tguess\treal')
    # print(f'DM\t{dm_guess:.3f}\t{dm_real:.3f}')
    # print(f'Freq\t{freq_guess:.3f}\t{1/period_real:.3f}')
    # print(f'Period\t{period_guess:.3f}\t{period_real:.3f}')

    # plots
    # raw data
    fig, ax = plt.subplots()
    ax.imshow(data, vmin=0, vmax=1, origin='lower', aspect='auto',
              cmap='gray', extent=[0, ntime-1, 0, nchan-1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.show()
    exit()

    fig.savefig("raw-data.svg")

    # timeseries
    fig, ax = plt.subplots()
    x = range(ntime)
    scale = 20
    for i, ts in enumerate(timeseries):
        ax.plot(x, ts.astype(float) + i * scale, c='k')
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    fig.savefig("timeseries.svg")

    # FFT of timeseries at DM closest to real DM
    fig, ax = plt.subplots()
    x = np.arange(ntime) * df
    idx = np.argmin(np.abs(np.array(dms) - dm_real))
    print(idx)
    ax.plot(x, power_spectrum[10])
    ax.axvline(1./period_real, ls='--', c='r')
    ax.set_xlabel('Fourier frequency')
    ax.set_ylabel('Amplitude')

    fig.savefig("fft.svg")
