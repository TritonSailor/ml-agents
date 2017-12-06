import numpy as np

class BufferException(Exception):
    """
    Related to errors with the Buffer.
    """
    pass

class Buffer(object):
	class AgentBuffer(object):
		class AgentBufferField(object):
			def __init__(self):
				self._list = []
			def __len__(self):
				return len(self._list)
			def __str__(self):
				return str(np.array(self._list).shape)
			def __getitem__(self, index):
				return self._list[index]
			def reorder(self, order):
				self._list = [self._list[i] for i in order]
				#Find better names
			def append_element(self, data):
				#need to handle the case the data is not the right size
				self._list += [np.array(data)]
			def append_list(self, data):
				#need to handle the case the data is not the right size
				self._list += list(np.array(data))
			def set(self, data):
				self._list = list(np.array(data))
			def get_batch(self, batch_size = None, training_length = None):
				# Is there enough points to retrieve ?
				# This should be going backwards
				if training_length == None:
					if batch_size == None:
						#return all of them
						return np.array(self._list)
					else:
						# return the batch_size first elements
						if batch_size > len(self._list):
							raise BufferException("Batch size requested is too large")
						return np.array(self._list[-batch_size:])
				else:
					if batch_size == None:
						# retrieve the maximum number of elements
						batch_size = len(self._list) - training_length + 1
					if (len(self._list) - training_length + 1) < batch_size :
						raise BufferException("The batch size and training length requested for get_batch where" 
							" too large given the current number of data points.")
						return 
					tmp_list = []
					for end in range(len(self._list)-batch_size+1, len(self._list)+1):
						tmp_list += [np.array(self._list[end-training_length:end])]
					return np.array(tmp_list)
					# do we need to reset the local buffer now ?	
			def reset_field(self):
				del self._list
				self._list = []
		def __init__(self):
			self._data = {}
		def __str__(self):
			return ", ".join(["'{0}' : {1}".format(k, str(self._data[k])) for k in self._data.keys()])
		def reset_agent(self):
			# for key in self._data:
			# 	self._data[key] = self.AgentBufferField()
			del self._data
			self._data = {}
			#Careful with garbage collection
		def __len__(self):
			return len(self._data)
		def __getitem__(self, key):
			if key not in self._data:
				self._data[key] = self.AgentBufferField()
			return self._data[key]
		def __setitem__(self, key, value):
			self._data[key] = value    
		def keys(self):
			return self._data.keys()
		def __contains__(self, key):
			return key in self.keys()
		def __iter__(self):
			for key in self.keys():
				yield key
		def check_length(self, key_list):
			if len(key_list) < 2:
				return True
			l = None
			for key in key_list:
				if key not in self._data:
					return False
				if ((l != None) and (l!=len(self._data[key]))):
					return False
				l = len(self._data[key])
			return True
		def shuffle(self, key_list = None):
			if key_list is None:
				key_list = self.keys()
			if not self.check_length(key_list):
				raise BufferException("Unable to shuffle if the fields are not of same length")
				return
			s = np.arange(len(self._data[key_list[0]]))
			np.random.shuffle(s)
			for key in key_list:
				self._data[key].reorder(s)
	def __init__(self):
		self.global_buffer = self.AgentBuffer() 
		#Should we have a global buffer? what if the system is distributed ?
		self.local_buffers = {}
	def __str__(self):
		return "global buffer :\n\t{0}\nlocal_buffers :\n{1}".format(str(self.global_buffer), 
			'\n'.join(['\tagent {0} :{1}'.format(k, str(self.local_buffers[k])) for k in self.local_buffers.keys()]))
	def __getitem__(self, key):
		if key not in self.local_buffers:
			self.local_buffers[key] = self.AgentBuffer()
		return self.local_buffers[key]
	def __setitem__(self, key, value):
		self.local_buffers[key] = value
	def keys(self):
		return self.local_buffers.keys()
	def __contains__(self, key):
		return key in self.keys()
	def __iter__(self):
		for key in self.keys():
			yield key
	# need to figure out how useful this is...
	def append_BrainInfo(self, info, extra_dictionary = None):
		# extra_dictionary : key to np.array 
		# is extra_dictionary needed ?
		# Not up to date
		extra_dictionary = {} if extra_dictionary is None else extra_dictionary
		for index , agent_id in enumerate(info.agents):
			if agent_id not in self.local_buffers:
				self.local_buffers[agent_id] = self.AgentBuffer()
			for key in extra_dictionary:
				self.local_buffers[agent_id].append({key:np.array(extra_dictionary[key])[index]})
			self.local_buffers[agent_id].append({'states':info.states[index]})
			for obs_num in range(len(info.observations)):
				self.local_buffers[agent_id].append({'observations_'+str(obs_num):info.observations[obs_num][index]})
			# self.local_buffers[agent_id].append({'observations':info.observations[index]})
			self.local_buffers[agent_id].append({'memories':info.memories[index]})
			self.local_buffers[agent_id].append({'rewards':info.rewards[index]})
			self.local_buffers[agent_id].append({'local_done':info.local_done[index]})
			self.local_buffers[agent_id].append({'previous_actions':info.previous_actions[index]})
	def reset_global(self):
		del self.global_buffer
		self.global_buffer = self.AgentBuffer() #Is it efficient for garbage collection ?
	def reset_all(self):
		del self.global_buffer
		self.global_buffer = self.AgentBuffer() 
		del self.local_buffers
		self.local_buffers = {}
	def append_global(self, agent_id ,key_list = None,  batch_size = None, training_length = None):
		
		if key_list is None:
			key_list = self.local_buffers[agent_id].keys()
		if not self.local_buffers[agent_id].check_length(key_list):
			raise BufferException("The length of the fields {0} for agent {1} where not of comparable length"
				.format(key_list, agent_id))
		for field_key in key_list:
			self.global_buffer[field_key].append_list(
				self.local_buffers[agent_id][field_key].get_batch(batch_size, training_length)
			)
	def append_all_agent_batch_to_global(self, key_list = None,  batch_size = None, training_length = None):
		#Maybe no batch_size, only training length and a flag corresponding to "get only last of training_length"
		#Or have a "max_training_length" and a "real_training_length" ?
		for agent_id in self.local_buffers.keys():
			self.append_global(agent_id ,key_list,  batch_size, training_length)



def discount_rewards(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def get_gae(rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
    """
    Computes generalized advantage estimate for use in updating policy.
    :param rewards: list of rewards for time-steps t to T.
    :param value_next: Value estimate for time-step T+1.
    :param value_estimates: list of value estimates for time-steps t to T.
    :param gamma: Discount factor.
    :param lambd: GAE weighing factor.
    :return: list of advantage estimates for time-steps t to T.
    """
    value_estimates = np.asarray(value_estimates.tolist() + [value_next])
    delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
    advantage = discount_rewards(r=delta_t, gamma=gamma*lambd)
    return advantage

