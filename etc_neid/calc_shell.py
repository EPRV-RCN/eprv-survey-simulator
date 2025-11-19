import numpy as np

import neid_calculator.neid_etcalc_public as etc

from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)

bp = Blueprint('calc_shell', __name__, url_prefix='/calc_shell')

@bp.route('/calculate_rv', methods=('GET', 'POST'))
def calculate_rv(temperature_in=5500, exptime_in=300, vmag_in=8, vsini_in=2.0, seeing_in=0.8):
    if request.method == 'POST':

        temperature = float(request.form['temperature'])
        vmag = float(request.form['vmag'])
        exptime = float(request.form['exptime'])
        vsini = float(request.form['vsini'])
        seeing = float(request.form['seeing'])
        if exptime==10:
            exptime=10.1
        error=None
        
        if not temperature:
            error = 'Effective Temperature is required'
        if not vmag:
            error = 'V Band Magnitude is required'
        if not exptime:
            error = 'Exposure time is required'
        if not vsini:
            error = 'Projected rotational velocity is required'
        if not seeing:
            error = 'Seeing is required'
            
        if error is None:
            rvprec=etc.NEID_RV_prec(temperature, vmag, exptime, vsini=vsini, seeing=seeing)
            if np.isnan(rvprec)==False:
                maxexp=etc.NEID_max_exptime(temperature, vmag, seeing=seeing)
                if maxexp==3600:
                    flash('Maximum recommended exposure time for this target is >3600 seconds', category='warning')
                else:
                    flash('Maximum recommended exposure time for this target is %d seconds' % maxexp, category='warning')
            flash('RV Precision = %4.3f m/s' % rvprec, category='message')
            
        temperature_in=request.form['temperature']
        exptime_in=request.form['exptime']
        vmag_in=request.form['vmag']
        vsini_in=request.form['vsini']
        seeing_in=request.form['seeing']
        
    return render_template('calc_shell/calculate_rv.html', temp=temperature_in, vm=vmag_in, et=exptime_in, vsi=vsini_in, see=seeing_in)

@bp.route('/calculate_snr', methods=('GET', 'POST'))
def calculate_snr(temperature_in=5500, exptime_in=300, vmag_in=8, wavelength_in=552.97, seeing_in=0.8):
    if request.method == 'POST':

        temperature = float(request.form['temperature'])
        vmag = float(request.form['vmag'])
        exptime = float(request.form['exptime'])
        wavelength = float(request.form['wavelength'])
        seeing = float(request.form['seeing'])
        if exptime==10:
            exptime=10.1
        error=None
        
        if not temperature:
            error = 'Effective Temperature is required'
        if not vmag:
            error = 'V Band Magnitude is required'
        if not exptime:
            error = 'Exposure time is required'
        if not seeing:
            error = 'Seeing is required'
        if wavelength==0:
            error = 'Wavelength required for SNR calculation'
            flash(error)
        if error is None:
            snr=etc.NEID_SNR(temperature, vmag, exptime, wavelength, seeing=seeing)
            maxexp=etc.NEID_max_exptime(temperature, vmag, seeing=seeing)
            if maxexp==3600:
                flash('Maximum recommended exposure time for this target is >3600 seconds', category='warning')
            else:
                flash('Maximum recommended exposure time for this target is %d seconds' % maxexp, category='warning')
            flash('SNR = %4.3f' % snr, category='message')
        temperature_in=request.form['temperature']
        exptime_in=request.form['exptime']
        vmag_in=request.form['vmag']
        wavelength_in=request.form['wavelength']
        seeing_in=request.form['seeing']
    return render_template('calc_shell/calculate_snr.html', temp=temperature_in, vm=vmag_in, et=exptime_in, wvl=wavelength_in, see=seeing_in)

@bp.route('/calculate_exp_rv', methods=('GET', 'POST'))
def calculate_exp_rv(temperature_in=5500, rvprec_in=1.0, vmag_in=8, vsini_in=2.0, seeing_in=0.8):
    if request.method == 'POST':

        temperature = float(request.form['temperature'])
        vmag = float(request.form['vmag'])
        rvprec = float(request.form['rvprec'])
        vsini = float(request.form['vsini'])
        seeing = float(request.form['seeing'])
        error=None
        
        if not temperature:
            error = 'Effective Temperature is required'
        if not vmag:
            error = 'V Band Magnitude is required'
        if not rvprec:
            error = 'Desired RV precision is required'
        if not vsini:
            error = 'Projected rotational velocity is required'
        if not seeing:
            error = 'Seeing is required'
            
        if error is None:
            exptime=etc.NEID_exptime_RV(temperature, vmag, rvprec, vsini=vsini, seeing=seeing)
            if np.isnan(exptime)==False:
                maxexp=etc.NEID_max_exptime(temperature, vmag, seeing=seeing)
                if maxexp==3600:
                    flash('Maximum recommended exposure time for this target is >3600 seconds', category='warning')
                else:
                    flash('Maximum recommended exposure time for this target is %d seconds' % maxexp, category='warning')

                flash('Exposure Time = '+str(exptime)+' s')
            else:
                flash('Maximum Exposure Time Exceeded (t>3600s)')
        temperature_in=request.form['temperature']
        rvprec_in=request.form['rvprec']
        vmag_in=request.form['vmag']
        vsini_in=request.form['vsini']
        seeing_in=request.form['seeing']
    return render_template('calc_shell/calculate_exp_rv.html', temp=temperature_in, rv=rvprec_in, vm=vmag_in, vsi=vsini_in, see=seeing_in)

@bp.route('/calculate_exp_snr', methods=('GET', 'POST'))
def calculate_exp_snr(temperature_in=5500, snr_in=100, vmag_in=8, wavelength_in=552.97, seeing_in=0.8):
    if request.method == 'POST':

        temperature = float(request.form['temperature'])
        vmag = float(request.form['vmag'])
        snr = float(request.form['snr'])
        wavelength = float(request.form['wavelength'])
        seeing = float(request.form['seeing'])
        error=None
        
        if not temperature:
            error = 'Effective Temperature is required'
        if not vmag:
            error = 'V Band Magnitude is required'
        if not snr:
            error = 'Desired SNR is required'
        if not seeing:
            error = 'Seeing is required'
        if wavelength==0:
            error = 'Wavelength required for SNR calculation'
            flash(error)
            
        if error is None:
            exptime=etc.NEID_exptime_SNR(temperature, vmag, snr, wavelength, seeing=seeing)
            if np.isnan(exptime)==False:
                maxexp=etc.NEID_max_exptime(temperature, vmag, seeing=seeing)
                if maxexp==3600:
                    flash('Maximum recommended exposure time for this target is >3600 seconds', category='warning')
                else:
                    flash('Maximum recommended exposure time for this target is %d seconds' % maxexp, category='warning')
                flash('Exposure Time = '+str(exptime)+' s')
            else:
                flash('Maximum Exposure Time Exceeded (t>3600s)')
        temperature_in=request.form['temperature']
        snr_in=request.form['snr']
        vmag_in=request.form['vmag']
        wavelength_in=request.form['wavelength']
        seeing_in=request.form['seeing']
    return render_template('calc_shell/calculate_exp_snr.html', temp=temperature_in, vm=vmag_in, sn=snr_in, wvl=wavelength_in, see=seeing_in)

@bp.route('/about', methods=('GET', 'POST'))
def about():
    return render_template('calc_shell/about.html')