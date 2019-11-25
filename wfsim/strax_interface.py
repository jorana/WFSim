import logging
import time
import pickle
import uproot
import nestpy

import numpy as np
import pandas as pd

import strax
from straxen.common import get_resource

from .core import RawData
from sklearn.cluster import DBSCAN

export, __all__ = strax.exporter()
__all__ += ['instruction_dtype', 'truth_extra_dtype']

instruction_dtype = [('event_number', np.int), ('type', np.int), ('t', np.int64), 
    ('x', np.float32), ('y', np.float32), ('z', np.float32), 
    ('amp', np.int), ('recoil', '<U2'), ('e_dep', np.float32)]


truth_extra_dtype = [
    ('n_electron', np.float),
    ('n_photon', np.float), ('n_photon_bottom', np.float),
    ('t_first_photon', np.float), ('t_last_photon', np.float), 
    ('t_mean_photon', np.float), ('t_sigma_photon', np.float), 
    ('t_first_electron', np.float), ('t_last_electron', np.float), 
    ('t_mean_electron', np.float), ('t_sigma_electron', np.float),]

log = logging.getLogger('SimulationCore')


@export
def rand_instructions(c):
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    instructions['t'] = np.repeat(uniform_times, 2) * int(1e9)
    instructions['event_number'] = np.digitize(instructions['t'], 
         1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1
    instructions['type'] = np.tile([1, 2], n)
    instructions['recoil'] = ['er' for i in range(n * 2)]

    r = np.sqrt(np.random.uniform(0, 2500, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-100, 0, n), 2)

    nphotons = np.random.uniform(2000, 2050, n)
    nelectrons = 10 ** (np.random.uniform(1, 4, n))
    instructions['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(int)

    return instructions

#Calculate the clustered values
def cluster_function(x):
    d = {}
    #use the average position for each cluster weighted by energy
    d["xp"] = np.average(x["xp"], weights=x["ed"])
    d["yp"] = np.average(x["yp"], weights=x["ed"])
    d["zp"] = np.average(x["zp"], weights=x["ed"])
    d["time"] = np.average(x["time"], weights=x["ed"])
    #Sum the energy
    d["ed"] = np.sum(x["ed"])
    
    #use for now the most abundant particle type for the cluster...
    types, counts = np.unique(x.type, return_counts=True)
    d["type"] = types[np.argmax(counts)]
    
    return pd.Series(d, index = ["xp", "yp", "zp", "time", "ed", "type"])


@export
def read_g4(file, eps=0.3):

    nc = nestpy.NESTcalc(nestpy.VDetector())
    A = 131.293
    Z = 54.
    density = 2.862  # g/cm^3   #SR1 Value
    drift_field = 82  # V/cm    #SR1 Values
    
    #get the simulation source
    source = uproot.open(file)["generator"]["SourceType"]._fTitle.decode("UTF-8")
    
    data = uproot.open(file)["events/events"]
    df = data.pandas.df(["xp","yp", "zp", "time", "ed", "nsteps", "eventid", "type"])
    #Add the interaction type in the correct format
    #this is not working when not loading the type in the line above??
    df["type"] = np.concatenate(data.array("type"))
    df["type"] = df["type"].apply(lambda x: x.decode("UTF-8"))

    df["time"] = df["time"]*1e9 # conversion to ns
         
    #Remove all values without energy depositon
    df = df[df.ed != 0 ]

    #Time Clustering
    time_scale = 10 #ns ## select some resonabel time later...
    dbscan_time_clustering = DBSCAN(eps=time_scale, min_samples=1)

    df["time_cluster"] = np.concatenate(df[["time"]].groupby("entry").apply(lambda x: dbscan_time_clustering.fit_predict(x.time.values.reshape(-1,1)) ))

    #Clustering in space
    #Cluster in xyz for each event(entry) and each time_cluster
    dbscan_clustering = DBSCAN(eps=eps, min_samples=1)
    df["cluster"] = np.concatenate(df.groupby(["entry", "time_cluster"]).apply(lambda x:  dbscan_clustering.fit_predict(np.stack(x[["xp", "yp", "zp"]].values))).values)

    #Apply the clustering for each event(entry), time_cluster and cluster
    df = df.groupby(["entry","time_cluster","cluster"]).apply(lambda x: cluster_function(x))

    df.xp /=10
    df.yp /=10
    df.zp /=10

    #Limit the interactions to the TPC
    tpc_radius_square = 2500
    z_lower = -100
    z_upper = 0
    df = df[(df.xp**2+df.yp**2<=47.9**2)&(df.zp<z_upper)&(df.zp>z_lower)]

    #Sort the df for each event in time
    df["index_dummy"] = df.index.get_level_values(0)
    df = df.sort_values(["index_dummy", "time"])

    #set time of first e deposition in each event to 0
    df["time"] = df.groupby(["entry"]).apply(lambda x: (x["time"]-x["time"].min())).values

    #try with longer times....
    #df["time"] = df["time"]*10

    # set the index to start from 0 and run to the final number of events 
    # This is important for simulations with external sources where a lot of events never reach the xenon
    idx = df.index.get_level_values(0)
    idx_lims = np.append(np.intersect1d(idx,np.unique(idx), return_indices = True)[1],len(idx))
    n_vals = [idx_lims[i+1] - idx_lims[i] for i in range(0,len(idx_lims)-1)]
    new_entry_idx = pd.Int64Index(np.repeat(np.arange(len(n_vals)), n_vals), dtype = "int64", name = "entry")
    
    df.index = pd.MultiIndex.from_arrays([new_entry_idx,
                                 df.index.get_level_values(1).values]
                                 )
    
    
    #lets reset the time cluster index aswell
    new_cluster_idx = pd.Int64Index(np.concatenate(df.groupby("entry").apply(lambda x: np.arange(len(x.index.get_level_values(1).values)))), dtype = "int64", name = "cluster")
    
    df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0).values,
                                 new_cluster_idx]
                                 )
    
    
    #and separate the events in time by one second
    event_spacing = 1e9
    df.time = np.cumsum(df.time+ (df.index.get_level_values(1).values == 0 )*event_spacing)

    #build the instructions
    n_instructions = len(df)
    ins = np.zeros(2*n_instructions, dtype=instruction_dtype)
    
    #shift the time by a constant offset...
    e_dep, ins['x'], ins['y'], ins['z'], ins['t'] = df.ed.values, \
                                                    np.repeat(df.xp.values, 2), \
                                                    np.repeat(df.yp.values, 2), \
                                                    np.repeat(df.zp.values, 2), \
                                                    np.repeat(df.time.values, 2)
                                                    
    ins["e_dep"] = np.repeat(df.ed.values, 2)
    ins["event_number"] = np.repeat(df.index.get_level_values(0).values,2)
    
    #try to set everythin exept the last event to event number = 0...
    #idx = np.repeat(df.index.get_level_values(0).values,2)
    #ins["event_number"] = np.clip(idx-idx[-1]+1, a_min = 0, a_max =np.inf)
    
    #ins["event_number"] = np.zeros(2*n_instructions)
    ins['type'] = np.tile((1, 2), n_instructions)
    
    p_type = df.type.values
    #lets take the gamma model for betas with low energies
    #https://indico.fnal.gov/event/18104/session/14/contribution/90/material/0/0.pdf
    #lets try 500 for now
    p_type = np.where((e_dep<500)&(p_type == "e-"), "gamma", p_type)

    
    #NEST handling...
    #https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp
    #line 406
    #https://arxiv.org/pdf/1106.1613.pdf page 18
    if "Kr83" in source:
        print("Kr83m in source, modify e_dep and types")
        
        #To use the special 9.4 keV case in NEST the energy has to be exactly 9.4 keV
        e_dep = np.where((e_dep<9.5)&(e_dep>9.3), 9.4 , e_dep)
        #change the particle type for the 9.4 keV line to Kr83m
        #p_type = np.where((e_dep<9.5)&(e_dep>9.3), "Kr83m" , p_type)
        #Why do i also have to apply the Kr83m model to the 32 keV line?
        p_type = np.array(["Kr83m"]*len(p_type))
    
    df["type"] = p_type
    #select the interaction type for NEST, use gamma (7) as default
    #add more interactions here...
    interaction_dict = {"e-": 8, "gamma": 7, "neutron": 0, "Kr83m": 11}
    interaction = [nestpy.INTERACTION_TYPE(interaction_dict.get(particle_type, 7)) for particle_type in p_type]

    #get the recoil type here, set "er" as default 
    recoil_dict = {"e-": "er", "gamma": "er", "neutron": "nr"}
    ins['recoil'] =np.repeat([recoil_dict.get(particle_type, "er") for particle_type in df.type.values],2)
    
    quanta = []

    for en, inter in zip(e_dep, interaction):
        y = nc.GetYields(inter,
                         en,
                         density,
                         drift_field,
                         A,
                         Z,
                         (1, 1))
        quanta.append(nc.GetQuanta(y, density).photons)
        quanta.append(nc.GetQuanta(y, density).electrons)
    ins['amp'] = quanta
    
    
    #lets interactions without electrons or photons
    ins = ins[ins["amp"] > 0]
    
    

    return ins

@export
def instruction_from_csv(file):
    return pd.read_csv(file).to_records(index=False)

@export
class ChunkRawRecords(object):
    def __init__(self, config):
        self.config = config
        self.rawdata = RawData(self.config)
        self.record_buffer = np.zeros(5000000, dtype=strax.record_dtype()) # 2*250 ms buffer
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)]) # 500 s1 + 500 s2

    def __call__(self, instructions):
        # Save the constants as privates
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        buffer_length = len(self.record_buffer)
        dt = self.config['sample_duration']
        rext = self.config['right_raw_extension']
        cksz = self.config['chunk_size'] * 1e9
        
        self.blevel = buffer_filled_level = 0
        self.chunk_time = np.min(instructions['t']) + cksz # Starting chunk

        for channel, left, right, data in self.rawdata(instructions, self.truth_buffer):
            pulse_length = right - left + 1
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            if left * 10 > self.chunk_time + 2 * rext:    
                yield from self.final_results()
                self.chunk_time += cksz

            if self.blevel + records_needed > buffer_length:
                log.warning('Chunck size too large, insufficient record buffer')
                yield from self.final_results()

            if self.blevel + records_needed > buffer_length:
                log.warning('Pulse length too large, insufficient record buffer, skipping pulse')
                continue

            # WARNING baseline and area fields are zeros before finish_results
            s = slice(self.blevel, self.blevel + records_needed)
            self.record_buffer[s]['channel'] = channel
            self.record_buffer[s]['dt'] = dt
            self.record_buffer[s]['time'] = dt * (left + samples_per_record * np.arange(records_needed))
            self.record_buffer[s]['length'] = [min(pulse_length, samples_per_record * (i+1)) 
                - samples_per_record * i for i in range(records_needed)]
            self.record_buffer[s]['pulse_length'] = pulse_length
            self.record_buffer[s]['record_i'] = np.arange(records_needed)
            self.record_buffer[s]['data'] = np.pad(data, 
                (0, records_needed * samples_per_record - pulse_length), 'constant').reshape((-1, samples_per_record))
            self.blevel += records_needed

        yield from self.final_results()

    def final_results(self):
        records = self.record_buffer[:self.blevel] # No copying the records from buffer
        maska = records['time'] < self.chunk_time
        records = records[maska]

        records = strax.sort_by_time(records) # Do NOT remove this line
        strax.baseline(records)
        strax.integrate(records)

        maskb = self.truth_buffer['fill'] & (self.truth_buffer['t_first_photon'] < self.chunk_time)
        truth = self.truth_buffer[maskb]
        self.truth_buffer[maskb]['fill'] = False

        truth.sort(order='t_first_photon')
        # Return truth without 'fill' field
        _truth = np.zeros(len(truth), dtype=instruction_dtype + truth_extra_dtype)
        for name in _truth.dtype.names:
            _truth[name] = truth[name]

        yield dict(raw_records=records, truth=_truth)
        
        self.record_buffer[:np.sum(~maska)] = self.record_buffer[:self.blevel][~maska]
        self.blevel = np.sum(~maska)

    def source_finished(self):
        return self.rawdata.source_finished


@strax.takes_config(
    strax.Option('fax_file', default=None, track=True,
                 help="Directory with fax instructions"),
    strax.Option('experiment', default='XENON1T', track=True,
                 help="Directory with fax instructions"),
    strax.Option('event_rate', default=5, track=False,
                 help="Average number of events per second"),
    strax.Option('chunk_size', default=5, track=False,
                 help="Duration of each chunk in seconds"),
    strax.Option('nchunk', default=4, track=False,
                 help="Number of chunks to simulate"),
    strax.Option('fax_config', 
                 default='https://raw.githubusercontent.com/XENONnT/'
                 'strax_auxiliary_files/master/fax_files/fax_config.json'),
    strax.Option('samples_to_store_before',
                 default=2),
    strax.Option('samples_to_store_after',
                 default=20),
    strax.Option('right_raw_extension', default=10000),
    strax.Option('trigger_window', default=50),
    strax.Option('zle_threshold', default=0))
class FaxSimulatorPlugin(strax.Plugin):
    depends_on = tuple()

    # Cannot arbitrarily rechunk records inside events
    rechunk_on_save = False

    # Simulator uses iteration semantics, so the plugin has a state
    # TODO: this seems avoidable...
    parallel = False

    # TODO: this state is needed for sorting checks,
    # but it prevents prevent parallelization
    last_chunk_time = -999999999999999

    # A very very long input timeout, our simulator takes time
    input_timeout = 3600 # as an hour

    def setup(self):
        c = self.config
        c.update(get_resource(c['fax_config'], fmt='json'))

        if c['fax_file']:
            if c['fax_file'][-5:] == '.root':
                self.instructions = read_g4(c['fax_file'],c['clustering_eps'])
                c['nevents'] = np.max(self.instructions['event_number'])
            else:
                self.instructions = instruction_from_csv(c['fax_file'])
                c['nevents'] = np.max(self.instructions['event_number'])

        else:
            self.instructions = rand_instructions(c)

        assert np.all(self.instructions['x']**2+self.instructions['y']**2 < 2500), "Interation is outside the TPC"
        assert np.all(self.instructions['z'] < 0) & np.all(self.instructions['z']>-100), "Interation is outside the TPC"
        assert np.all(self.instructions['amp']>0), "Interaction has zero size"


    def _sort_check(self, result):
        if result['time'][0] < self.last_chunk_time + 5000:
            raise RuntimeError(
                "Simulator returned chunks with insufficient spacing. "
                "Last chunk's max time was {timeA}, "
                "this chunk's first time is {timeB}.".format(timeA=self.last_chunk_time, 
        timeB=result['time'][0]))
        if np.diff(result['time']).min() < 0:
            raise RuntimeError("Simulator returned non-sorted records!")
        self.last_chunk_time = result['time'].max()

@export
class RawRecordsFromFax(FaxSimulatorPlugin):
    provides = ('raw_records', 'truth')
    data_kind = {k: k for k in provides}

    def setup(self):
        super().setup()
        self.sim = ChunkRawRecords(self.config)
        self.sim_iter = self.sim(self.instructions)

    def infer_dtype(self):
        dtype = dict(raw_records=strax.record_dtype(),
                     truth=instruction_dtype + truth_extra_dtype)
        return dtype

    def is_ready(self, chunk_i):
        """Overwritten to mimic online input plugin.
        Returns False to check source finished;
        Returns True to get next chunk.
        """
        if 'ready' not in self.__dict__: self.ready = False
        self.ready ^= True # Flip
        return self.ready

    def source_finished(self):
        """Return whether all instructions has been used."""
        return self.sim.source_finished()

    def compute(self, chunk_i):
        try:
            result = next(self.sim_iter)
        except StopIteration:
            raise RuntimeError("Bug in chunk count computation")
        # self._sort_check(result['raw_records'])
        return result
