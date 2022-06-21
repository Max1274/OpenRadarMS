import sys
import time
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mmwave.dsp.utils import Window
from mmwave.dataloader.adc import DCA1000
import gc

gc.disable()


plt.close('all')

# QOL settings
loadData = True
datapath = r"C:\\Users\\I009140\\Desktop\\03_Inbetriebnahme DCA\\DUMP\\leicht_schraeg\\"

numFrames = 900
numADCSamples = 256
numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 128

range_resolution, bandwidth, max_range = dsp.range_resolution(numADCSamples)
doppler_resolution, max_doppler = dsp.doppler_resolution(bandwidth)

print("range resolution: " + str(range_resolution))
print("doppler resolution: " + str(doppler_resolution))

max_range = 18
max_doppler = 7

plotRangeDopp = False
plot2DXYdoppler = False
plot2DXYClusters = False
plotmusic = True
saveVideo = True


def main():
    ims = []
    max_size = 0

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile(
            datapath + "adc_data.bin",
            dtype=np.uint16)

        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Data Loaded!")

    # (1.1) Required Plot Declarations
    if plot2DXYdoppler:
        fig, axes = plt.subplots(1, 2)
    elif plot2DXYClusters:
        fig, axes = plt.subplots(1, 2)
    elif plotmusic:
        fig, axes = plt.subplots(1, 2)
    elif plotRangeDopp:
        fig = plt.figure()

    for i, frame in enumerate(adc_data[50:400]):  # leicht_schraeg:50, gerade_vor:120, 90grad:50
        #        print(i,end=',') # Frame tracker
        # (2) Range Processing
        start = time.time()

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=2, clutter_removal_enabled=True,
                                                       window_type_2d=Window.HAMMING)

        #evaluate doppler variance for Micro-Doppler-recognition of wheels
        doppler_var = dsp.get_micro_doppler(aoa_input, numLoopsPerFrame)
        doppler_var_sum_angle = np.sum(doppler_var, axis=1)

        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            plt.imshow(det_matrix_vis / det_matrix_vis.max(), aspect='auto', \
                       extent=[-max_doppler, max_doppler, \
                               max_range, 0])
            plt.title("Range-Doppler plot " + str(i))
            plt.pause(0.01)
            plt.clf()

        # (4) Detection of Reflections
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        #thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.os_,
         #                                                        axis=0,
          #                                                      arr=fft2d_sum.T,
           #                                                    # l_bound=1.5,
            #                                                  guard_len=4,
             #                                                noise_len=20,
              #                                              scale=1.07
               #                                            )

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.os_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              # l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=10,
                                                              scale=1.07
                                                              )

        #thresholdRange = dsp.cfar_os(fft2d_sum, n=32, k=24, offset=1.02, axis=0)
        #thresholdRange, noiseRange = dsp.os_ms(fft2d_sum, guard_len=4, noise_len=10, scale=1.07, axis=0)
        #thresholdDoppler, noiseDoppler = dsp.os_ms(fft2d_sum, guard_len=4, noise_len=16, scale=1.1, axis=1)


        #thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (fft2d_sum > 0)
        det_range_mask = (fft2d_sum > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        #snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'azimuth', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<i4', '<f4', dtype_location, '<f4']})
        detRefl = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detRefl['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detRefl['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detRefl['peakVal'] = peakVals.flatten()
        #detRefl['SNR'] = snr.flatten()

        music_input = np.zeros((numTxAntennas*numRxAntennas, detRefl.shape[0]), dtype=np.complex128)

        n_insertions = 0
        for refs in range(detRefl.shape[0]):
            music_input = aoa_input[:, :, detRefl[refs]['dopplerIdx']]
            music_input = np.transpose(music_input, axes=(1,0))
            if (detRefl['dopplerIdx']==detRefl[refs]['dopplerIdx']).sum() <= 6-1:     #wie viele Ziele befinden sich in dopplerBin?
                music_spectrum = dsp.aoa_music_1D_sv(music_input, (detRefl['dopplerIdx']==detRefl[refs]['dopplerIdx']).sum())
            else:
                music_spectrum = dsp.aoa_music_1D_sv(music_input,7-1)
            peak_data, _ = dsp.peak_search_full_variance(music_spectrum, 181)

            detRefl[refs+n_insertions]['azimuth'] = peak_data[0]['peakLoc'] - 180/2-1
            peak_data = np.delete(peak_data, 0, axis=0)
            while peak_data.shape[0] >= 1:
                detRefl = np.insert(detRefl, refs+n_insertions, detRefl[refs+n_insertions], axis=0)
                n_insertions += 1
                detRefl[refs+n_insertions]['azimuth'] = peak_data[0]['peakLoc']-90
                peak_data = np.delete(peak_data, 0, axis=0)

        #adjust dopplerIdx to take negative values into account
        for j in range(detRefl.shape[0]):
            if detRefl[j]['dopplerIdx'] >= numLoopsPerFrame/2:
                detRefl[j]['dopplerIdx'] = detRefl[j]['dopplerIdx'] - numLoopsPerFrame

        numDetObjs = detRefl.shape[0]
        dtf = np.dtype({'names': ['location_x', 'location_y', 'azimuth', 'doppler', 'cluster', 'variance_D'],
                        'formats': ['<f4', '<f4', '<f4', '<f4', '<i4', '<f4']})
        detObj2D = np.zeros((numDetObjs,), dtype=dtf)

        detObj2D['location_x'] = detRefl['rangeIdx'] * range_resolution * np.sin(detRefl['azimuth']/180*np.pi)
        detObj2D['location_y'] = detRefl['rangeIdx'] * range_resolution * np.cos(detRefl['azimuth']/180*np.pi)
        detObj2D['azimuth'] = detRefl['azimuth']
        detObj2D['doppler'] = detRefl['dopplerIdx'] * doppler_resolution
        



        if plotmusic:
            ann0 = axes[0].annotate(f"XY scatter " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")
            ann1 = axes[1].annotate(f"velocity profile " + str(i), (0.5, 1.03), xycoords="axes fraction", ha="center")

            axes[0].set_ylim(bottom=0, top=max_range)
            axes[0].set_ylabel('x-range [m]')
            axes[0].set_xlim(left=-10, right=10)
            axes[0].set_xlabel('y-range [m]')
            axes[0].grid(visible=True)

            axes[1].set_xlabel('azimuth [°]')
            axes[1].set_ylabel('velocity [m/s]')
            axes[1].grid(visible=True)
            axes[1].set_xlim(left=-40, right=40)
            axes[1].set_ylim(bottom=-3, top=3)

            if not saveVideo:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=20, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['azimuth'], detObj2D['doppler'], marker='o', s=20)
                cb = plt.colorbar(p0, ax=axes[0])
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
                cb.remove()
            else:
                p0 = axes[0].scatter(detObj2D['location_x'], detObj2D['location_y'], c=detObj2D['doppler'],
                                     vmin=-max_doppler, vmax=max_doppler, marker='o', s=5, cmap=cm.jet)
                p1 = axes[1].scatter(detObj2D['azimuth'], detObj2D['doppler'], marker='o', s=5, c='#1f77b4')
                p = [p0, p1, ann0, ann1]
                ims.append(p)

        end = time.time()
        print("execution time " + str(i) + ": " + str(end - start))

    if saveVideo:
        cb = fig.colorbar(p0, ax=axes[0])
        ani = animation.ArtistAnimation(fig, ims, repeat=False, blit=True)
        ani.save(datapath + 'auswertung_music06_21scale1_07spatialsmooting.mp4', dpi=200, fps=1 / 0.035)





if __name__ == '__main__':
    main()