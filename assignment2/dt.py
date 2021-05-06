import sys
import math
from collections import deque
import pandas as pd

train_file, test_file = sys.argv[1], sys.argv[2]
output_file = sys.argv[3]



def bfs(root):
	q = deque()
	q.append(root)

	while q :
		for i in range(len(q)) :
			t = q.popleft()
			if isinstance(t, LeafNode):
				print(t.label,"|"+'\t', end= " ")
			else :
				print(t.attribute, t.child_nodes.keys(), "|"+'\t' , end = " ")
				for child in t.child_nodes :
					q.append(t.child_nodes[child])

		print()


class InternalNode():
	def __init__(self, attribute):
		self.attribute = attribute
		self.child_nodes = {}

	def __str__(self):
		ret = str(self.__dict__)
		return ret


class LeafNode():
	def __init__(self, label) :
		self.label = label

	def __str__(self):
		ret = str(self.label)
		return ret



class Builder():

	def __init__(self, train_file, output_file):
		self.df = []
		self.attributes = {}

		data = None
		with open(train_file, 'r') as f :
			data = [ datum.split('\t') for datum in f.read().splitlines()  ]
		data = pd.DataFrame(data)
		self.df, header = data[1:], data.iloc[0]
		self.df.columns = header

		with open(output_file, 'a') as f :
			cols = ""
			for col in list(self.df.columns):
				cols += col+'\t'
			f.write(cols[:-1]+'\n')
		
		for col_name in self.df.columns:
			self.attributes[col_name] = list(self.df[col_name].unique())


	def make_decision_tree(self, data):
		if len(data.columns) == 2:
			return LeafNode(data.iloc[:,-1].value_counts().idxmax())

		if len(data.iloc[:,-1].unique()) == 1 :
			return LeafNode(str(data.iloc[:,-1].unique()[0]))

		test_att = get_test_att(data)
		node = InternalNode(test_att)

		for att in self.attributes[test_att]:
			child = data.loc[ data[test_att] == att  ]

			if child.empty :
				node.child_nodes[att] = LeafNode(data.iloc[:,-1].value_counts().idxmax())

			else:
				tmp_node = self.make_decision_tree(child.drop(test_att ,1))
				if tmp_node is not None :
					node.child_nodes[att] = tmp_node

		return node


	def ret_decision_tree(self):
		dt = self.make_decision_tree(self.df)
		
		return dt


def info(val):
	val_sum, info = val.sum(), 0

	for num in val :
		if num :
			info -= (num/val_sum) * math.log(num/val_sum)
		else :
			return 0
	return info


def gain_ratio(table):
	all_val_sum = table.values.sum()
	wei_avg, spl_info = 0, 0

	for val in table.values :
		val_div_sum = val.sum()/ all_val_sum 
		wei_avg += val_div_sum * info(val)
		spl_info -= val_div_sum * math.log(val_div_sum) / math.log(2)

	ret = wei_avg / spl_info
	
	return ret


def get_test_att(data):
	cand_atts = dict()
	
	att_names = data.columns[:-1]
	label = data[data.columns[-1]]

	for att_name in att_names :
		cross_table = pd.crosstab(data[att_name], label)
		cand_atts[att_name] = gain_ratio(cross_table)

	ret = min(cand_atts.keys(), key = lambda x : cand_atts[x])

	return ret


def classification(node, row):
	if isinstance(node, LeafNode):
		return node.label
	else :
		return classification( node.child_nodes[row[node.attribute]], row   )

def classification_using_dt(dt, test_file, output_file):
	test_df = None
	
	data = None
	with open(test_file, 'r') as f :
		data = [ datum.split('\t') for datum in f.read().splitlines()  ]
	data = pd.DataFrame(data)
	test_df, header = data[1:], data.iloc[0]
	test_df.columns = header

	with open(output_file, 'a') as f :
		for i, row in test_df.iterrows():
			result = list(row.values)
			result.append( classification(dt, row) )

			sentence = ""
			for word in result :
				sentence += word+'\t'
			f.write(sentence[:-1]+'\n')
	

builder = Builder(train_file, output_file)
dt = builder.ret_decision_tree()
# bfs(dt)
classification_using_dt(dt, test_file, output_file)