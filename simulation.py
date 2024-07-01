# ==============================
# import libs
# ==============================
import copy
import math
import numpy as np
import pandas as pd

mu = 20
sigma = 2
rho_1 = 0
alpha=0
L = 3
p = 5
h = 1
b = 9
u = 4
w = 6
m = 1.5


# ==============================
# define classes
# ==============================
class echelon:
    def __init__(self, id, inventory, lead_time):
        self.id = id
        self.customer = self.id - 1
        self.supplier = self.id + 1
        self.inventory = inventory
        self.demand = 0
        self.backorder = 0
        self.depletion = []
        self.lead_time = lead_time
        self.customer_order = []
        self.safety_factor = 3
        self.pred_error = []
        self.past_pred = []
        self.pred_len = p
        self.pred = moving_avg

    def deplete(self):
        demand = self.customer_order[-1]
        if demand + self.backorder <= self.inventory:
            deplete = demand + self.backorder
            self.inventory -= deplete
            self.backorder = 0
        else:
            deplete = self.inventory
            self.inventory = 0
            self.backorder = demand + self.backorder - deplete
        self.depletion.append([deplete, 0])

    def demand_forecast(self, WIP):

        # get pred error
        if len(self.customer_order) == 1:  # when no pred record, do not consider safety inventory
            sigma = 0
        elif len(self.customer_order) < self.pred_len + 1:  # past records are not long enough
            sigma = np.sqrt(np.square(np.array(self.past_pred) - np.array(self.customer_order[1:])).mean(axis=None))
        else:
            sigma = np.sqrt(
                np.square(np.array(self.past_pred[-p:]) - np.array(self.customer_order[-p:])).mean(axis=None))

        self.pred_error.append(sigma)

        # pred
        if len(self.customer_order) < self.pred_len:  # past records are not long enough
            pred = self.customer_order[-1]
        else:
            pred = self.pred(np.array(self.customer_order[-p:]))
        self.past_pred.append(pred)


        # # old version
        # # lead time
        # D_hat = pred * self.lead_time
        # # safety inventory
        # y = D_hat + self.safety_factor * sigma

        # new version: representing safety inventory by L+1
        # lead time
        y = pred * (self.lead_time * 1.2)
        # negative order allowed

        q = y - self.inventory - WIP  # if not allowed, q should be max(y - self.inventory, 0)
        # negative order not allowed
        # q = max(y - self.inventory - WIP, 0)  # if not allowed, q should be max(y - self.inventory, 0)

        # bullwhip absorption
        if self.id == target_index - 1:
            q = q - k * (q - mu / (1 - rho_1))

        self.demand = q

        return q


# ==============================
# define prediction functions
# ==============================
def LS(y):
    x = np.linspace(0, len(y) - 1, len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    y = y[:, np.newaxis]
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), y)
    pred = (len(y) * alpha[0] + alpha[1])[0]
    return pred


def moving_avg(y):
    pred = y.mean()
    return pred


def w_moving_avg(y):
    weights = np.arange(1, 8)
    pred = np.dot(y, weights) / weights.sum()
    return pred


def e_moving_avg(y):
    alpha = 2 / (len(y) + 1)
    pred = y[0]
    for i in range(1, len(y)):
        pred = alpha * y[i] + (1 - alpha) * pred
    return pred


# ==============================
# define demand generation functions
# ==============================
def sin_wave(t):
    # pred = 1000 + 200 * math.sin(math.pi / 20 * t) + np.random.normal(0, 100) # sin20
    pred = 1000 + 200 * math.sin(math.pi / 10 * t) + np.random.normal(0, 100)  # sin10
    return pred


def corr_demand(D_t_1):
    pred = mu + rho_1 * D_t_1 + np.random.normal(0, 20)
    return pred


def AR_2(D_t_1, D_t_2):
    pred = 1000 + 0.5 * D_t_1 - 0.25 * D_t_2 + np.random.normal(0, 100)
    return pred, D_t_1


# ==============================
# Simulation
# ==============================

demand_type = 'AR1'

# generate echelons
eche_num = 2
eche_list = []
for i in range(eche_num):
    eche = echelon(i, mu / (1 - rho_1) * (L+1), L)
    eche_list.append(eche)

# define simulation parameters

sim_length = 10000
demand_delay = 0
delivery_delay = 0

demand = []
depletion = []
depletion_all = []

# stat
inventory = []
backorder = []

D_t = mu / (1 - rho_1)
D_t_1 = 900
D_t_2 = 1000
for t in range(sim_length):
    print(t)

    # determine the demand

    # corr
    D_t = round(corr_demand(D_t))
    demand.append([round(D_t), t])

    # # sin
    # demand.append([round(sin_wave(t)), t])

    # # linear
    # demand.append([round(linear_trend(t)), t])

    # # AR_2
    # D_t_1, D_t_2 = AR_2(D_t_1, D_t_2)
    # demand.append([round(D_t_1, t)])

    for eche in eche_list:
        if eche.id == 0:
            # first receive ordered amount
            WIP = 0
            for x in eche_list[eche.supplier].depletion:
                x[1] += 1

            if len(eche_list[eche.supplier].depletion) != 0:

                if eche_list[eche.supplier].depletion[0][1] == L:
                    eche.inventory += eche_list[eche.supplier].depletion[0][0]
                    del (eche_list[eche.supplier].depletion[0])
            if len(eche_list[eche.supplier].depletion) != 0:
                WIP = np.array(eche_list[eche.supplier].depletion)[:, 0].sum()


            # then deplete ordered amount
            eche.customer_order.append(D_t)
            eche.deplete()

            # determine the order

            eche_list[eche.supplier].customer_order.append(eche.demand_forecast(WIP))

        if eche.id in range(1, eche_num - 1):
            # first receive ordered amount
            WIP = 0
            for x in eche_list[eche.supplier].depletion:
                x[1] += 1
            if len(eche_list[eche.supplier].depletion) != 0:

                if eche_list[eche.supplier].depletion[0][1] == L:
                    eche.inventory += eche_list[eche.supplier].depletion[0][0]
                    del (eche_list[eche.supplier].depletion[0])
            if len(eche_list[eche.supplier].depletion) != 0:
                WIP = np.array(eche_list[eche.supplier].depletion)[:, 0].sum()

            # then deplete ordered amount
            # eche.customer_order.append(eche_list[eche.customer].customer_order)
            eche.deplete()

            # determine the order

            eche_list[eche.supplier].customer_order.append(eche.demand_forecast(WIP))

        if eche.id == eche_num - 1:
            # first receive ordered amount
            WIP = 0
            for x in depletion:
                x[1] += 1
            if len(depletion) != 0:

                if depletion[0][1] == L:
                    eche.inventory += depletion[0][0]
                    del (depletion[0])
            if len(depletion) != 0:
                WIP = np.array(depletion)[:, 0].sum()

            # then deplete ordered amount
            eche.deplete()

            # determine the order

            order_last = eche.demand_forecast(WIP)
            depletion.append([order_last, 0])
            depletion_all.append([order_last, 0])

    inventory.append([eche_list[0].inventory, eche_list[1].inventory])
    backorder.append([eche_list[0].backorder, eche_list[1].backorder])

# ==============================
# Saving data
# ==============================

stat_order = []

for i in range(len(eche_list)):
    stat_order.append(eche_list[i].customer_order)
stat_order.append(np.array(depletion_all)[:,0])
inventory = np.array(inventory)
stat_order = np.array(stat_order).T
backorder = np.array(backorder)
inventory = pd.DataFrame(inventory, columns=['Inventory 0', 'Inventory 1'])
stat_order = pd.DataFrame(stat_order, columns=['Order 0', 'Order 1', 'Order 2'])
backorder = pd.DataFrame(backorder, columns=['Back 0', 'Back 1'])

eche = 'eche_' + str(target_index)
inventory.to_csv('data/inventory_noLT_' + demand_type + '_' + eche + '_' + str(k) + '.csv', index=False)
stat_order.to_csv('data/order_noLT_' + demand_type + '_' + eche + '_' + str(k) + '.csv', index=False)
backorder.to_csv('data/back_noLT_' + demand_type + '_' + eche + '_' + str(k) + '.csv', index=False)
