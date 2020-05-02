from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
from pomegranate import Node
from pomegranate import BayesianNetwork

holiday = DiscreteDistribution({True: 0.25, False: 0.75})
work = DiscreteDistribution({True: 0.7, False: 0.3})

got_fat = ConditionalProbabilityTable(
        [[True,  True,  True,  0.25],
         [True,  True,  False, 0.75],
         [True,  False, True,  0.1],
         [True,  False, False, 0.9],
         [False, True,  True,  0.9],
         [False, True,  False, 0.1],
         [False, False, True,  0.2],
         [False, False, False, 0.8]], [holiday, work])

free = ConditionalProbabilityTable(
        [[True,  True,  True, 	True,  	0.55],
         [True,  True,  True, 	False, 	0.45],
         [True,  True,	False,  True,  	0.9],
         [True,  True,  False,  False, 	0.1],
         [True,	 False, True,   True, 	0.2],
         [True,	 False, True,   False,	0.8],
         [True,  False, False,  True, 	0.7],
         [True,  False, False,  False,	0.3],
         [False, True,  True,   True,  	0.55],
         [False, True,  True,   False, 	0.45],
         [False, True,  False,  True,  	0.9],
         [False, True,  False,  False, 	0.1],
         [False, False, True,   True, 	0.2],
         [False, False, True,   False,	0.8],
         [False, False, False,  True,	0.7],
         [False, False, False,  False,	0.3]], [holiday, work, got_fat])

s1 = Node(holiday, name="holiday")
s2 = Node(work, name="work")
s3 = Node(free, name="free")
s4 = Node(got_fat, name="got_fat")


model = BayesianNetwork("Free time")
model.add_states(s1, s2, s3, s4)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.add_edge(s4, s3)
model.add_edge(s1, s4)
model.add_edge(s2, s4)
model.bake()


print('Holiday, work and free', model.probability([True, True, True, True]))
print('Holiday, not work and not free:', model.probability([True, False,True, False]))
print('Not holiday, not work and free:', model.probability([False, False,True, True]))
print('Not holiday, work and not free:', model.probability([False, True,True, False]))

print('Predict with no constraints')
print(model.predict_proba({}))

print('Predict whether is free, knowing that someone is working')
print(model.predict_proba({'work': True}))
print('Predict whether is free, knowing that the persone got fat')
print(model.predict_proba({'got_fat': True}))
''' sus ii spun eu ce probabilitate are al 3-lea true in cazul in care primele 2
sunt true
in al doilea il intreb care este probabilitatea ca toate 3 sa fie true'''
