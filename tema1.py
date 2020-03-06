from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
from pomegranate import Node
from pomegranate import BayesianNetwork

holiday = DiscreteDistribution({True: 0.25, False: 0.75})
work = DiscreteDistribution({True: 0.7, False: 0.3})
free = ConditionalProbabilityTable(
        [[True,  True,  True,  0.55],
         [True,  True,  False, 0.45],
         [True,  False, True,  0.9],
         [True,  False, False, 0.1],
         [False, True,  True,  0.2],
         [False, True,  False, 0.8],
         [False, False, True,  0.7],
         [False, False, False, 0.3]], [holiday, work])


s1 = Node(holiday, name="holiday")
s2 = Node(work, name="work")
s3 = Node(free, name="free")


model = BayesianNetwork("Free time")
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()


print('Holiday, work and free', model.probability([True, True, True]))
print('Holiday, not work and not free:', model.probability([True, False, False]))
print('Not holiday, not work and free:', model.probability([False, False, True]))
print('Not holiday, work and not free:', model.probability([False, True, False]))

print('Predict with no constraints')
print(model.predict_proba({}))

print('Predict whether is free, knowing that someone is working')
print(model.predict_proba({'work': True}))
