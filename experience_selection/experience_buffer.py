from baselines.ddpg.memory import RingBuffer, array_min2d
import random
from collections import namedtuple
import numpy as np
import sortedcontainers
import tensorflow as tf
import math
from os import path, makedirs

class ESMemoryAdapter(object):
    """Adapter for the baselines DDPG code
    overwrite options: 'FIFO', 'expl_xx' (stochastic exploration magnitude based with alpha = xx),
    tde_xx (stochastic TDE based with alpha = xx), 'resv' (Reservoir sampling)
    sample options: 'uniform','PER_xx (TDE rank based with alpha = xx)
    """
    def __init__(self, limit, action_shape, observation_shape, overwrite_policy='FIFO',
                 sample_policy='uniform', batch_size=64, forgetting_factor=0.99):

        ow = overwrite_policy.lower().strip()
        if 'fifo' in ow:
            ow_tab = {'type': 'FIFO'}
        elif 'expl' in ow:
            _, alpha = ow.split('_')
            ow_tab = {'type': 'rank based stochastic',
                      'metric': 'exploration_magnitude',
                      'proportional': False, # lowest values have the highest chance of being
                      # overwritten
                      'alpha': float(alpha)}
        elif 'tde' in ow:
            _, alpha = ow.split('_')
            ow_tab = {'type': 'rank based stochastic',
                      'metric': 'tde',
                      'proportional': False,  # lowest values have the highest chance of being
                      # overwritten
                      'alpha': float(alpha)}
        elif 'resv' in ow or 'reservoir' in ow:
            ow_tab = {'type': 'Reservoir'}
        else:
            assert False, 'unknown overwrite policy: {:s}'.format(overwrite_policy)

        sa = sample_policy.lower().strip()
        if 'uniform' in sa:
            sa_tab = {'type': 'uniform'}
        elif 'per' in sa:
            _, alpha = ow.split('_')
            sa_tab = {'type': 'rank based stochastic',
                        'metric': 'tde',
                        'proportional': True,  # Samples with high TDE have a higher chance of
                            # being sampled again
                        'alpha': float(alpha)}
        else:
            assert False, 'unknown sample policy: {:s}'.format(sample_policy)

        settings = {
            'buffer_size': limit,
            'forgetting_factor': forgetting_factor,
            'batch_size': batch_size,
            'reuse': 32, # not used in the baselines version
            'experience_properties': {
                'observations': {
                    'state': {
                        'shape': observation_shape,
                        'dtype': np.float32,
                        'ttype': tf.float32,
                    },
                },
                'action': {
                    'shape': action_shape,
                    'dtype': np.float32,
                    'ttype': tf.float32,
                },
                'terminal': {
                    'shape': (1,),
                    'dtype': np.uint8,
                    'ttype': tf.float32,
                },
                'reward': {
                    'shape': (1,),
                    'dtype': np.float32,
                    'ttype': tf.float32,
                },
                'experience_meta_data': {
                    'tde': {
                        'shape': (1,),
                        'dtype': np.float16,
                        'default': np.inf,
                    },
                    'exploration_magnitude': {
                        'shape': (1,),
                        'dtype': np.float16,
                        'default': 0.0
                    },
                }
            },
            'buffer_properties': {
                'overwrite policy': ow_tab,
                'sample policy': sa_tab
                },
            }

        self.experience_selection_buffer = ExperienceBuffer(settings)

        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        return self.experience_selection_buffer.get_batch_baselines(batch_size=batch_size)

    def append(self, obs0, action, reward, obs1, terminal1, training=True,
               experience_meta_data=None):
        if not training:
            return

        self.experience_selection_buffer.add_experience(observation={'state': obs0},
                                                        action=action,
                                                        next_observation={'state': obs1},
                                                        reward=reward,
                                                        terminal=terminal1,
                                                        experience_meta_data=experience_meta_data)

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.experience_selection_buffer)



class ExperienceBuffer(object):
    """Class with methods for saving and replaying experiences.

    This class takes a description of the observations, actions, rewards and other signals that
    are relevant. It creates a series of placeholders. It further includes methods to store
    experiences, sample batches (return the placeholders and numpy arrays with values in a
    feed-dict) and store and restore to and from a file.
    """

    def __init__(self, properties):
        """Inits the buffer using the properties specified.

        Args:
            properties: a dictionary specifying the properties of the buffer to be made.
                Properties should have following structure:
                    {
                        'buffer_size': <int, required, number of experiences to store>
                        'batch_size': <int, optional, can also be specified in
                            the functions that require it>
                        'reuse': <int, required, number of times a sample will on average be used
                            if the get_available_batches method is used>
                        'experience_properties': <dict, required, describes the properties of
                        the signals to be stored. Should have the following keys:>
                            {
                                   'observations': <dict, required, sensor or state data signals>
                                   'action': <dict, required, action signal properties>
                                   'reward': <dict, required, reward signal properties>
                                   'terminal': <dict, required, properties of signal that
                                   indicates whether the experience was the last in the episode>
                                   'experience_meta_data': <dict, optional, meta data signals used for
                                   learning (such as TDE)>
                            }
                            observations and learn data are dicts where the keys are names of
                            signals and the values are dicts of the same form as action,
                            reward and terminal:
                            {
                                'shape': <tuple, required, dimensions of signal (e.g. (2,) )>
                                'dtype': <numpy dtype, required, numpy data type>
                                'ttype': <tensorflow dtype, required, tensorflow data type>
                            }
                        'load_replay_buffer': <string, optional, file path of a saved experience
                        buffer, from which experiences are loaded into this one>
                        'buffer_properties': <dict, required, describes the overwrite and sample
                            strategies, see overwrite and sample policy classes at the bottom of
                            this file for options. Default options:
                                'overwrite': {
                                    'type': 'FIFO'
                                },
                                'sample': {
                                    'type': 'uniform'
                                },
                        'scale_rewards': <float, optional, scale rewards by this factor when
                            replaying, to limit (increase) gradients while keeping original rewards
                            for bookkeeping.
                        'forgetting_factor': <float, required, gamma in [0,1)>
                    }
        """
        assert properties is not None
        self._properties = properties
        self._buffer = self._create_buffer()
        self._meta_data_change_listeners = self._create_meta_data_change_listeners()
        """_buffer contains the numpy data of the signals (experience tuples)"""
        self._buffer_metadata = self._create_buffer_metadata()
        """_buffer_metadata contains meta data about the buffer, such as the last write index, 
        the number of new experiences and the indices of unused experiences """
        self._experience_and_episode_metadata = self._create_experience_and_episode_metadata()
        """_experience_metadata contains automatically collected meta data about the 
        experiences in the buffer, such the episode they were from and the return from the 
        experience to the final experience in the episode. """
        self._placeholders = self._create_placeholders()

        self.overwrite_policy = self._create_overwrite_policy()
        self.sample_policy = self._create_sample_policy()

        self._optionally_load_buffer()

    def get_two_timestep_tensor_placeholders(self):
        """Get a dict with references to the placeholders by time-step (current, next) """
        timestep_tensors = {'current_timestep': {}, 'next_timestep': {}}
        ct = timestep_tensors['current_timestep']
        nt = timestep_tensors['next_timestep']
        for name in self._placeholders['observations']:
            ct[name] = self._placeholders['observations'][name]
            nt[name] = self._placeholders['observations_post'][name]
        ct['action'] = self._placeholders['action']
        nt['reward'] = self._placeholders['reward']
        nt['terminal'] = self._placeholders['terminal']
        return timestep_tensors

    def add_experience(self, observation, action, next_observation, reward, terminal,
                       experience_meta_data=None):
        """Add a new experience to the buffer.

        Args:
            observation: current time-step observation dict with numpy arrays for the sensor signals
            action: current time-step action numpy array, float or int
            next_observation: next time-step observation dict with numpy arrays for the sensor
                signals
            reward: float or int of the (next time-step) reward
            terminal: bool: True is the experience is the last of an episode. False otherwise
            experience_meta_data: optional dict with (part of)
        """
        write_index = self.overwrite_policy.next_index()
        if self._experience_and_episode_metadata['current_episode_finished']:
            self._start_episode()
        self._experience_and_episode_metadata['last_episode_rewards']['rewards'].append(
            self.Seq_ep_rew(buffer_index=write_index, reward=reward))

        if write_index is not None:
            for modality in self._buffer['observations']:
                self._buffer['observations'][modality][write_index] = observation[modality]
            for modality in self._buffer['observations_post']:
                self._buffer['observations_post'][modality][write_index] = next_observation[
                    modality]
            self._buffer['action'][write_index] = action
            self._buffer['reward'][write_index] = reward
            self._buffer['terminal'][write_index] = terminal
            for cat in self._buffer['experience_meta_data']:
                self._call_meta_data_change_listeners(indices=write_index, category=cat, pre=True)
                if experience_meta_data is not None and cat in experience_meta_data:
                    self._buffer['experience_meta_data'][cat][write_index] = experience_meta_data[cat]
                else:
                    self._buffer['experience_meta_data'][cat][write_index] = \
                        self._properties['experience_properties']['experience_meta_data'][cat]['default']
                self._call_meta_data_change_listeners(indices=write_index, category=cat)
            self._buffer_metadata['unused_experience_idcs'].add(write_index)
            self._buffer_metadata['fresh_experience_count'] += 1
        if terminal:
            self._finish_episode()

    def nr_available_batch_updates(self, batch_size=None):
        """Number of batch updates available given batch_size, fresh experiences, reuse

        Args:
            batch_size: int, optional, use this batch size instead of the one given during
                initialization

        Returns: int, number of batch updates available. Note that get_batch gives no warning
        when more batches are requested.

        """
        batch_size = batch_size or self._properties['batch_size']
        reuse = self._properties['reuse']
        fresh = min(self._buffer_metadata['fresh_experience_count'], len(self))
        return math.floor(fresh * reuse / batch_size)

    def get_batch_baselines(self, batch_size):
        indcs = self.sample_policy.sample_indices(batch_size, only_new=False)
        return {
            'obs0': array_min2d(self._buffer['observations']['state'][indcs]),
            'obs1': array_min2d(self._buffer['observations_post']['state'][indcs]),
            'rewards': array_min2d(self._buffer['reward'][indcs]),
            'actions': array_min2d(self._buffer['action'][indcs]),
            'terminals1': array_min2d(self._buffer['terminal'][indcs]),
            'indices': indcs
        }


    def get_batch(self, batch_size=None, **kwargs):
        """Get a tuple: (training batch feed_dict, the buffer indices of the experiences)

        Args:
            batch_size: int, optional: use a different batch size than given in the init properties
            **kwargs: give additional named arguments, options include:
                only_new_experiences: boolean, only return experiences that have not been
                    returned before
                dont_count_as_use: boolean, do not count the returned experiences as used
                indices: list, return the experiences with the given indices
        Returns: a tuple: (feed_dict (placeholders and the numpy contents), list: indices of the
            returned experiences.
        """
        only_new = kwargs.get('only_new_experiences', False)
        dont_count_as_use = kwargs.get('dont_count_as_use', False)
        batch_size = batch_size or self._properties['batch_size']
        if batch_size < self._buffer_metadata['last_write_index'] + 1:
            if 'indices' in kwargs:
                indcs = kwargs['indices']
            else:
                indcs = self.sample_policy.sample_indices(batch_size, only_new)
            if indcs is None:
                return None, None
            if not dont_count_as_use:
                self._buffer_metadata['unused_experience_idcs'].difference_update(indcs)
                self._buffer_metadata['fresh_experience_count'] -= (batch_size / float(
                    self._properties['reuse']))
                self._buffer_metadata['fresh_experience_count'] = max(
                    self._buffer_metadata['fresh_experience_count'], 0)
            feed_dict = {}
            for exp_comp in 'observations observations_post action reward terminal'.split():
                self._feed_data(feed_dict=feed_dict, exp_cmp=exp_comp,
                                indcs=indcs, place_holders=self._placeholders,
                                buffer=self._buffer, properties=self._properties[
                        'experience_properties'])
            if self._properties.get('scale_rewards'):
                feed_dict[self._placeholders['reward']] = feed_dict[self._placeholders[
                    'reward']] * self._properties.get('scale_rewards')
            return feed_dict, indcs
        else:
            return None, None

    def get_indices_for_n_batches(self, number_of_batches, batch_size=None):
        """Predetermine the buffer indices for sampling a number of batches.

        The buffer indices are returned and can be given to get_batch() to get those specific
            experience
        Args:
            number_of_batches: int, required, number of batches to return indices for
            batch_size: int, optional, the number of experiences per batch. If not specified the
                given during initialization is used.

        Returns: numpy array of batches * batch_size with the indices

        """
        batch_size = batch_size or self._properties['batch_size']
        if number_of_batches > 0:
            indices = np.empty((number_of_batches, batch_size), dtype=np.int32)
            indices.fill(np.nan)
            for bi in range(number_of_batches):
                idcs = self.sample_policy.sample_indices(batch_size)
                if idcs is not None:
                    indices[bi] = idcs
            return indices

    def update_experience_meta_data(self, indices, data):
        """Update the metadata (learn data) for the experiences of the given indices.

        Args:
            indices: list, buffer indices of the experiences for which the data is provided. Note
            that get_batch gives the indices of the experiences in the batch
            data: dict, containing (some of) the fields specified in learn data during init and
            the values of those fields corresponding to the experiences with the provided indices.
        """
        for cat in data:
            self._call_meta_data_change_listeners(category=cat, indices=indices, pre=True)
            self._buffer['experience_meta_data'][cat][indices] = data[cat]
            self._call_meta_data_change_listeners(category=cat, indices=indices)

    def feed_dict_from_observation(self, observation):
        """Return a feed dict with the internal placeholders and the given observation

        Args:
            observation: observation dict with numpy observation (no batch dimension)

        Returns: the feed dict, observations are expanded to batch dimension 1

        """
        feed_dict = {}
        meta_data = self._properties['experience_properties']['observations']
        for mod in observation:
            mod_meta_data = meta_data[mod]
            data = np.expand_dims(observation[mod], axis=0)
            feed_dict[
                self._placeholders['observations'][mod]] = \
                ExperienceBuffer.optionally_normalize(data, mod_meta_data)
        return feed_dict

    @staticmethod
    def optionally_normalize(data, meta_data):
        if 'norm_dev' in meta_data:
            data = data.astype(np.float32)
            data /= meta_data['norm_dev']
        if 'norm_add' in meta_data:
            data += meta_data['norm_add']
        return data

    def save_to_disk(self, file_path):
        """Saves the contents of the buffer (experiences only) to a specified directory.

        Args:
            file_path: directory path, file name buffer.npz is appended by the function.
        """
        file_path = path.expanduser(file_path)
        makedirs(file_path, exist_ok=True)
        filename = file_path + 'buffer.npz'
        flat_buffer = self._flatten_dict(self._buffer)
        for key, npar in flat_buffer.items():
            flat_buffer[key] = npar[0:self._buffer_metadata['last_write_index']]
        np.savez_compressed(filename, **flat_buffer)

    def load_buffer_from_disk(self, file_path):
        """Loads the experiences from a previously saved buffer into this one.

        Caution: this function assumes the current buffer is empty and overwrites it. Only
        experiences and learn data are loaded, no metadata.
        Args:
            file_path: directory in which a file 'buffer.npz' is saved.
        """
        bufferfile_name = path.expanduser(file_path) + 'buffer.npz'
        try:
            with np.load(bufferfile_name) as external_flat_buffer:
                added_experiences = self._process_flat_buffer_file(external_flat_buffer)
                self._buffer_metadata['last_write_index'] = added_experiences - 1
            print("Loaded {:d} experiences from {:s}".format(added_experiences, bufferfile_name))
        except IOError:
            print('Could not load: {:s}'.format(bufferfile_name))

    def all_fresh(self):
        """Mark all experiences in the buffer as unused for training. """
        self._buffer_metadata['fresh_experience_count'] = self._buffer_metadata['last_write_index']

    def discard_memory(self):
        """Discard all experiences to start with an empty buffer"""
        self._buffer_metadata['last_write_index'] = -1
        self._buffer_metadata['unused_experience_idcs'] = set()

    def add_experience_meta_data_update_listener(self, experience_meta_data_category, listener):
        """Add an event listener that is called with indices for which the metadata has changed."""
        assert experience_meta_data_category in self._buffer['experience_meta_data'], 'no metadata for {:s}'.format(
            experience_meta_data_category)
        self._meta_data_change_listeners[experience_meta_data_category].append(listener)

    def get_report(self):
        """Get a report of the buffer data for a tb summary"""
        report = {'experiences': self._buffer_metadata['last_write_index'] + 1}
        for exp_data in self._buffer['experience_meta_data']:
            x = self._buffer['experience_meta_data'][exp_data][
                0:self._buffer_metadata['last_write_index'], 0]
            x = x[~np.isnan(x)]
            x = x[~np.isinf(x)]
            report[exp_data] = x

        return report

    def __len__(self):
        return self._properties['buffer_size']

    def _create_meta_data_change_listeners(self):
        return {name: [] for name in self._buffer['experience_meta_data']}

    def _call_meta_data_change_listeners(self, category, indices, pre=False):
        for callback_function in self._meta_data_change_listeners[category]:
            callback_function(indices, pre)

    @property
    def fresh_experiences(self):
        """The number of experiences not yet trained with (keeping in mind batch size and reuse)"""
        return self._buffer_metadata['fresh_experience_count']

    @property
    def last_episode_mean_return(self):
        """Returns the mean return over the states visited in the last episode.

        This function can only be called between episodes; after an experience has been added
        with terminal = True, but before the first experience of the next episode is added.

        Returns: The mean return over the states visited in the last episode
        Throws: assertion error when an episode has not just finished
        """
        assert self._experience_and_episode_metadata['current_episode_finished'], \
            'last_episode_mean_return can only be called after an episode has just terminated; ' \
            'after ' \
            'an experience has been added with terminal = True and before the first experience' \
            ' of the next episode is added.'
        return self._experience_and_episode_metadata['last_episode_mean_return']

    @property
    def last_episode_initial_state_return(self):
        """Returns the return of the first state visited in the last episode.

        This function can only be called between episodes; after an experience has been added
        with terminal = True, but before the first experience of the next episode is added.

        Returns: The return of the first state visited in the last episode
        Throws: assertion error when an episode has not just finished
        """
        assert self._experience_and_episode_metadata['current_episode_finished'], \
            'last_episode_initial_state_return can only be called after an episode has just ' \
            'terminated; after ' \
            'an experience has been added with terminal = True and before the first experience' \
            ' of the next episode is added.'
        return self._experience_and_episode_metadata['last_episode_initial_return']

    def _create_buffer(self):
        """ Create the numpy nd-arrays for the experiences and their meta data.

        Returns:
            A dict of the same structure as 'experience_properties' with the initialized numpy
            tensors
        """
        exp_prop = self._properties['experience_properties']
        # here the s a s' r t experience is saved each time-step because of experience replay
        # research.
        # More memory efficient would be to save s a r t per timestep and ensure timesteps are not
        # orphaned (at least 2 subsequent)
        assert all(name in exp_prop for name in ['observations', 'action', 'reward'])
        exp_prop['observations_post'] = exp_prop['observations']
        return self._create_variable_buffer(exp_prop)

    def _create_variable_buffer(self, variable_description):
        """Recursively build parts of the experience buffer from the dict definition.

        Args:
            variable_description: either a signal description dict of the following structure:
                {
                    'shape': <tuple, required, dimensions of signal (e.g. (2,) )>
                    'dtype': <numpy dtype, required, numpy data type>
                    'ttype': <tensorflow dtype, required, tensorflow data type>
                }
            or a (multi level) dict containing signal descriptions as values.

        Returns:
            numpy nd-array for a signal description, (multi level) dict of numpy arrays for a
            (multi level) dict of descriptions

        """
        if 'shape' in variable_description and 'dtype' in variable_description:
            shape = [self._properties['buffer_size']]
            shape.extend(list(variable_description['shape']))
            return np.empty(shape=shape, dtype=variable_description['dtype'])
        else:
            returndict = {}
            for var_props in variable_description:
                assert isinstance(variable_description[var_props], dict), 'bad experience replay ' \
                                                                          'settings'
                returndict[var_props] = self._create_variable_buffer(
                    variable_description[var_props])
            return returndict

    @staticmethod
    def _create_buffer_metadata():
        """Create a dict with metadata specific to the operation of the buffer.
        Returns: the metadatadict
        """
        metadata_dict = {
            'last_write_index': -1,
            'fresh_experience_count': 0,
            'unused_experience_idcs': set(),
        }
        return metadata_dict

    def _create_experience_and_episode_metadata(self):
        """Create a dict with metadata specific to experiences and episodes.
        Returns: the metadatadict
        """
        self.Seq_ep_rew = namedtuple('rewardseq', ['reward', 'buffer_index'])
        metadata_dict = {
            'experience_episodes': np.zeros(self._properties['buffer_size'], dtype=np.int32),
            'experience_returns': np.zeros(self._properties['buffer_size'], dtype=np.float32),
            'last_episode_mean_return': None,
            'last_episode_initial_return': None,
            'last_episode_rewards': {'episode': 0, 'rewards': []},
            'current_episode_index': 0,
            'current_episode_finished': False
        }
        return metadata_dict

    def _create_placeholders(self):
        """Create the internal set of tensorflow placeholders to feed experiences to."""
        prop = self._properties['experience_properties']
        with tf.variable_scope('placeholders'):
            return {
                'observations': self._create_placeholder_set(prop['observations'], timestep=0),
                'observations_post': self._create_placeholder_set(
                    prop['observations'], timestep=1),
                'action': self._create_placeholder_set(prop['action'], timestep=0, name='action'),
                'reward': self._create_placeholder_set(prop['reward'], timestep=1, name='reward'),
                'terminal': self._create_placeholder_set(prop['terminal'], timestep=1,
                                                         name='terminal')
            }

    def _create_placeholder_set(self, param, **kwargs):
        """Recursively create a (dict of) tf placeholders from a (dict of) signal description(s).

        Args:
            param: a (dict of) signal description(s) (see init)

        Returns: a (dict of) placeholders with the specified type and shape (+ -1 batch dimension)

        """
        if 'shape' in param:
            shape = [None]
            shape.extend(list(param['shape']))
            full_name = '{:s}_time_{:d}'.format(kwargs['name'], kwargs['timestep'])
            return tf.placeholder(shape=shape, dtype=param['ttype'], name=full_name)
        else:
            return {name: self._create_placeholder_set(param[name], name=name, **kwargs) for name in
                    param}

    def _create_overwrite_policy(self):
        """Init the overwrite policy which determines the next buffer index to be (over)written to.
        Returns: The overwrite policy object

        """
        policy_prop = self._properties['buffer_properties']['overwrite policy']
        if policy_prop['type'] == 'FIFO':
            return FifoOverwritePolicy(self)
        elif policy_prop['type'] == 'rank based stochastic':
            return StochasticRankBasedOverwritePolicy(
                experience_buffer=self,
                metric=policy_prop['metric'],
                highest_values_highest_priority=policy_prop['proportional'],
                alpha=policy_prop['alpha']
            )
        elif policy_prop['type'] == 'Reservoir':
            return ReservoirOverwritePolicy(self)
        else:
            assert False, 'unknown overwrite policy'

    def _create_sample_policy(self):
        """Create the sample policy instance based on the settings dict provided to init.

        Returns: the sample policy instance, which determines how to sample from the buffer."""
        policy_prop = self._properties['buffer_properties']['sample policy']
        if policy_prop['type'] == 'uniform':
            return UniformSamplePolicy(self)
        elif policy_prop['type'] == 'rank based stochastic':
            return RankBasedPrioritizedSamplePolicy(
                self, metric=policy_prop['metric'],
                highest_values_highest_priority=policy_prop['proportional'],
                alpha=policy_prop['alpha'])
        else:
            assert False, 'unknown sample policy'

    def _feed_data(self, feed_dict, exp_cmp, indcs, place_holders, buffer, properties):
        """Internal recursive function to fill part of a feed_dict with placeholders and data.

        Args:
            feed_dict: the (partially filled) feed_dict
            exp_cmp: key of the dict to be filled (the value of which is
                either another dict with signals or a signal)
            indcs: the experience indices to be used for the batch
            place_holders: dict with (dict of) placeholders containing at least exp_cmp
            buffer: buffer dict containing at least exp_cmp as key
        """
        if isinstance(buffer[exp_cmp], dict):
            for sub_cmp in buffer[exp_cmp]:
                self._feed_data(feed_dict, sub_cmp, indcs, buffer=buffer[exp_cmp],
                                place_holders=place_holders[exp_cmp],
                                properties=properties[exp_cmp])
        else:
            result_data = buffer[exp_cmp][indcs]

            feed_dict[place_holders[exp_cmp]] = \
                ExperienceBuffer.optionally_normalize(result_data, properties[exp_cmp])

    def _start_episode(self):
        """Start experience metadata administration for a new episode.

        This function is called when a new experience is added after the last episode finished.
        """
        self._experience_and_episode_metadata['last_episode_mean_return'] = None
        self._experience_and_episode_metadata['last_episode_initial_return'] = None
        self._experience_and_episode_metadata['current_episode_finished'] = False
        self._experience_and_episode_metadata['current_episode_index'] += 1
        self._experience_and_episode_metadata['last_episode_rewards']['episode'] = \
            self._experience_and_episode_metadata['current_episode_index']
        self._experience_and_episode_metadata['last_episode_rewards']['rewards'] = []

    def _finish_episode(self):
        """Update experience metdadata with the knowledge that the current episode just finished.

        This function is called by add_experience when terminal is True
        """
        self._experience_and_episode_metadata['current_episode_finished'] = True
        episode = self._experience_and_episode_metadata['current_episode_index']
        count, rollout_sum, ret = 0, 0, 0
        for seq_rew in reversed(
                self._experience_and_episode_metadata['last_episode_rewards']['rewards']):
            ret = seq_rew.reward + self._properties['forgetting_factor'] * ret
            count, rollout_sum = count + 1, rollout_sum + ret
            idx = seq_rew.buffer_index
            if idx is not None and self._experience_and_episode_metadata['experience_episodes'][ \
                    idx] == episode:
                self._experience_and_episode_metadata['experience_returns'][idx] = ret
        self._experience_and_episode_metadata['last_episode_initial_return'] = ret
        self._experience_and_episode_metadata['last_episode_mean_return'] = rollout_sum / float(
            count)
        self._experience_and_episode_metadata['last_episode_rewards']['rewards'] = []

    def _optionally_load_buffer(self):
        """Load the contents of a saved buffer iff 'load_replay_buffer' is set in settings."""
        filepath = self._properties.get('load_replay_buffer')
        if filepath:
            self.load_buffer_from_disk(filepath)

    def _flatten_dict(self, dictionary, basename=''):
        """Recursive helper function that produces a one level dict from a dict.

        Args:
            dictionary: the dict to be flattened
            basename: concatenated name of the higher level keys, used to recreate the original
            structure.

        Returns: a one level dictionary in which the keys of different levels are joined by '/'

        """
        result = {}
        for key, val in dictionary.items():
            if isinstance(val, np.ndarray):
                result[basename + key] = val
            elif isinstance(val, dict):
                branch = self._flatten_dict(val, basename=basename + key + '/')
                result.update(branch)
            else:
                assert False, 'unexpected type: {:s}'.format(str(type(val)))
        return result

    def _process_flat_buffer_file(self, flat_external):
        """Load the contents from an external flat buffer file into the buffer.

        Args:
            flat_external: the flat buffer dict to be loaded

        Returns: the number of experiences that were loaded from the external buffer into the
        local buffer. This function does not mark the loaded experiences as new experiences,
        see all_fresh() to do so.

        """
        added_experiences = min(len(self), len(flat_external.items()[0][1]))
        flat_self = self._flatten_dict(self._buffer)
        for key, val in flat_self.items():
            if key in flat_external:
                val[0:added_experiences - 1] = flat_external[key][0: added_experiences - 1]
            else:
                print("MISSING FROM EXTERNAL DATABASE BUFFER: {:s}".format(key))
        return added_experiences


# noinspection PyProtectedMember
class OverwritePolicy(object):
    """Abstract base class for determining buffer index to write new experience to.

     This class is only defines general methods and is subclassed by the actual overwrite policy
     classes. """

    def __init__(self, experience_buffer):
        """Initialize the overwrite policy.

        Args:
            experience_buffer: ExperienceBuffer instance which the policy acts upon.
        """
        self.experience_buffer = experience_buffer
        self.index = -1

    def next_index(self):
        """Return the buffer index that the next new experience should be written to.

        Returns: int, buffer index

        """
        if self.experience_buffer._buffer_metadata['last_write_index'] < len(
                self.experience_buffer) - 1:
            self.experience_buffer._buffer_metadata['last_write_index'] += 1
            self.index = self.experience_buffer._buffer_metadata['last_write_index']
        else:
            self.experience_buffer._buffer_metadata['last_write_index'] = len(
                self.experience_buffer) - 1
            self._next_index()  # overwrite in a smart way when full
        if self.index is not None and self.index > len(self.experience_buffer) - 1:
            self.index = 0
        return self.index

    def _next_index(self):
        """Called by next_index when the buffer is full to implement more advanced overwriting
        logic.

        Do not call this method directly, always call next_index() instead.
        """
        raise NotImplementedError


class FifoOverwritePolicy(OverwritePolicy):
    """Basic overwrite policy that always overwrites the oldest experience."""

    def __init__(self, experience_buffer):
        """Initialize the policy to overwrite the given ExperienceBuffer instance in a FIFO
        manner."""
        super().__init__(experience_buffer)

    def _next_index(self):
        """Called by next_index when the buffer is full to implement more advanced overwriting
        logic.

        Do not call this method directly, always call next_index() instead."""
        self.index += 1


class ReservoirOverwritePolicy(OverwritePolicy):
    """Overwrite policy that ensures each time-step ever experienced has an equal chance of
    being in the buffer at any given time."""

    def __init__(self, experience_buffer):
        """Initialize the policy to overwrite the given ExperienceBuffer instance using
        Reservoir sampling."""
        super().__init__(experience_buffer)
        self.idx_count = len(experience_buffer)

    def _next_index(self):
        """Called by next_index when the buffer is full to implement more advanced overwriting
        logic.

        Do not call this method directly, always call next_index() instead."""
        self.idx_count += 1
        retention_chance = len(self.experience_buffer) / self.idx_count
        if random.random() < retention_chance:
            self.index = random.randint(0, len(self.experience_buffer) - 1)
        else:
            self.index = None


class StochasticRankBasedOverwritePolicy(OverwritePolicy):
    """Overwrite policy that overwrites stochastically based on some (experience_meta_data) metric when
    full."""

    def __init__(self, experience_buffer, metric, highest_values_highest_priority=True, alpha=1.2):
        super().__init__(experience_buffer)
        self.sampler = OrderedDatabaseIndicesSampler(
            experience_buffer=experience_buffer,
            metric=metric,
            bins=max(3, int(len(self.experience_buffer) / 500)),
            alpha=alpha,
            lowest_value_lowest_index=not highest_values_highest_priority
        )

    def _next_index(self):
        """Called by next_index when the buffer is full to implement more advanced overwriting
                logic.

                Do not call this method directly, always call next_index() instead."""
        self.index = self.sampler.sample_one()


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class SamplePolicy(object):
    """Abstract base class for determining buffer indices to sample experience batches from.

     This class is only defines general methods and is subclassed by the actual sample policy
     classes. """

    def __init__(self, experience_buffer):
        """Initialize the sample policy for the given ExperienceBuffer instance.

        Args:
            experience_buffer: the ExperienceBuffer to be sampled from
        """
        self.experience_buffer = experience_buffer

    def sample_indices(self, batch_size, only_new):
        """Get the buffer indices for a training batch of experiences.

        Args:
            batch_size: int, required, number of experiences in the batch
            only_new: boolean, only sample from previously unsampled experiences.
        """
        raise NotImplementedError

    def _default_sample(self, batch_size, only_new):
        if only_new:
            idcs = np.array(list(
                self.experience_buffer._buffer_metadata['unused_experience_idcs']))
            if len(idcs) == 0:
                return None
            if len(idcs) < batch_size:
                return np.random.choice(idcs, batch_size, replace=True)
            else:
                return np.random.choice(idcs, batch_size, replace=False)
        else:
            if self.experience_buffer._buffer_metadata['last_write_index'] - 1 < batch_size:
                return np.random.choice(self.experience_buffer._buffer_metadata['last_write_index'],
                                        batch_size, replace=True)
            else:
                return np.random.choice(self.experience_buffer._buffer_metadata['last_write_index'],
                                        batch_size, replace=False)


# noinspection PyProtectedMember
class UniformSamplePolicy(SamplePolicy):
    """Sample policy that samples uniformly at random from the buffer."""

    def __init__(self, experience_buffer):
        """Initialize the sample policy that samples from experience_buffer uniformly at random"""
        super().__init__(experience_buffer)

    def sample_indices(self, batch_size, only_new=False):
        """Return a batch of buffer indices.

        Args:
            batch_size: int, required, number of buffer indices to return
            only_new: bool, only return previously unsampled experiences.

        Returns:
            list, experience buffer indices.
        """
        return self._default_sample(batch_size, only_new)


# noinspection PyProtectedMember
class RankBasedPrioritizedSamplePolicy(SamplePolicy):
    def __init__(self, experience_buffer, metric, highest_values_highest_priority=True, alpha=0.7):
        super().__init__(experience_buffer)
        self.sampler = OrderedDatabaseIndicesSampler(
            experience_buffer=experience_buffer,
            metric=metric,
            bins=self.experience_buffer._properties['batch_size'],
            alpha=alpha,
            lowest_value_lowest_index=not highest_values_highest_priority
        )

    def sample_indices(self, batch_size, only_new=False):
        if batch_size:
            assert batch_size == self.experience_buffer._properties['batch_size']
        if only_new or self.experience_buffer._buffer_metadata[
            'last_write_index'] - 1 <= batch_size:
            return self._default_sample(batch_size, only_new)
        else:
            return self.sampler.sample_all()


class OrderedDatabaseIndicesSampler(object):
    def __init__(self, experience_buffer, metric, bins, alpha, lowest_value_lowest_index=True):
        order_multiplier = 1 if lowest_value_lowest_index else -1
        self.experience_buffer = experience_buffer
        self.bins = bins
        self.bin_indices = [0, 0]
        self.alpha = alpha
        start_list = []
        self.ordered_indices = sortedcontainers.SortedListWithKey(start_list, key=lambda
            x: float(order_multiplier * self.experience_buffer._buffer['experience_meta_data'][metric][x][0]))
        experience_buffer.add_experience_meta_data_update_listener(metric, self.update)

    def update(self, indices, pre=False):
        """Since the ordered list is indexed based on the sorting,
        entries should be removed with their old keys, otherwise duplicate entries arise."""

        if type(indices) == int:
            if pre:
                self.ordered_indices.discard(indices)
            else:
                self.ordered_indices.add(indices)
        else:
            for idx in indices:
                i = int(idx)
                if pre:
                    self.ordered_indices.discard(i)
                else:
                    self.ordered_indices.add(i)

    def __getitem__(self, item):
        return self.ordered_indices[item]

    def __len__(self):
        return len(self.ordered_indices)

    def sample_one(self):
        self._possibly_rebuild_bins()
        return self._sample_bin(np.random.randint(0, len(self.bin_indices) - 1))

    def sample_all(self):
        self._possibly_rebuild_bins()
        return [self._sample_bin(i) for i in range(len(self.bin_indices) - 1)]

    def _possibly_rebuild_bins(self):
        size = self.experience_buffer._buffer_metadata['last_write_index']
        if len(self.bin_indices) - 1 != self.bins or self.bin_indices[-1] != \
                size:
            sample_probabilities = (1 / np.arange(1, size + 1)) ** self.alpha
            sample_probabilities = sample_probabilities / sample_probabilities.sum()
            cum_prob = sample_probabilities.cumsum()
            self.bin_indices = [0]
            bins = min(self.bins, size)
            for i in range(bins - 1):
                self.bin_indices.append(max(
                    self.bin_indices[i] + 1,
                    np.argmax(cum_prob >= (i + 1) / (bins))))
            self.bin_indices.append(size)

    def _sample_bin(self, bin):
        ordered_index = np.random.randint(self.bin_indices[bin], self.bin_indices[
            bin + 1])
        return self[ordered_index]
