import logging
import time
import pickle
import uproot
import nestpy

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

import strax
from straxen.common import get_resource

from .core import RawData

export, __all__ = strax.exporter()
__all__ += ['instruction_dtype', 'truth_extra_dtype']

instruction_dtype = [('event_number', np.int), ('type', np.int), ('t', np.float32), 
    ('x', np.float32), ('y', np.float32), ('z', np.float32), 
    ('amp', np.int), ('recoil', '<U2')]

truth_extra_dtype = [('n_photon', np.float), ('n_electron', np.float),
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
                                                    
    ins["event_number"] = np.repeat(df.index.get_level_values(0).values,2)
    ins['type'] = np.tile((1, 2), n_instructions)
    
    #select the interaction type for NEST, use gamma (7) as default
    #add more interactions here...
    interaction_dict = {"e-": 8, "gamma": 7, "neutron": 0}
    interaction = np.repeat([nestpy.INTERACTION_TYPE(interaction_dict.get(particle_type, 7)) for particle_type in df.type.values],2)

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

        chunk_i = record_j = 0 # Indices of chunk(event), record buffer
        for channel, left, right, data in self.rawdata(instructions, self.truth_buffer):
            pulse_length = right - left + 1
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            # Currently chunk is decided by instruction using event_number
            # TODO: mimic daq strax insertor chunking
            chunk_i =0
            if instructions['event_number'][self.rawdata.instruction_index] > chunk_i:
                yield self.final_results(record_j)
                record_j = 0 # Reset record buffer
                self.truth_buffer['fill'] = np.zeros_like(len(self.truth_buffer)) # Reset truth buffer
                chunk_i = instructions['event_number'][self.rawdata.instruction_index]

            if record_j + records_needed > buffer_length:
                log.warning('Chunck size too large, insufficient record buffer')
                yield self.final_results(record_j)
                record_j = 0
                self.truth_buffer['fill'] = np.zeros_like(len(self.truth_buffer))
            
            if record_j + records_needed > buffer_length:
                log.Warning('Pulse length too large, insufficient record buffer, skipping pulse')
                continue

            # WARNING baseline and area fields are zeros before finish_results
            s = slice(record_j, record_j + records_needed)
            self.record_buffer[s]['channel'] = channel
            self.record_buffer[s]['dt'] = dt
            self.record_buffer[s]['time'] = dt * (left + samples_per_record * np.arange(records_needed))
            self.record_buffer[s]['length'] = [min(pulse_length, samples_per_record * (i+1)) 
                - samples_per_record * i for i in range(records_needed)]
            self.record_buffer[s]['pulse_length'] = pulse_length
            self.record_buffer[s]['record_i'] = np.arange(records_needed)
            self.record_buffer[s]['data'] = np.pad(data, 
                (0, records_needed * samples_per_record - pulse_length), 'constant').reshape((-1, samples_per_record))

            record_j += records_needed

        yield self.final_results(record_j)

    def final_results(self, record_j):
        records = self.record_buffer[:record_j] # Copy the records from buffer
        records = strax.sort_by_time(records) # Must keep this for sorted output
        strax.baseline(records)
        strax.integrate(records)

        _truth = self.truth_buffer[self.truth_buffer['fill']]
        # Return truth without 'fill' field
        truth = np.zeros(len(_truth), dtype=instruction_dtype + truth_extra_dtype)
        for name in truth.dtype.names:
            truth[name] = _truth[name]

        return dict(raw_records=records, truth=truth)

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
                self.instructions = read_g4(c['fax_file'])
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
        #if result['time'][0] < self.last_chunk_time + 500:
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
        self._sort_check(result['raw_records'])
        return result
