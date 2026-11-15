import json
import math
import sys

INF = 10 ** 9
CAP = 3
P = {'robust': 0.3, 'maxw': 5, 'tb1': 5.5, 'tb2': 7.5, 'pre_every': 1, 'use_sv': 1, 'sv_floor': 0.1, 'opt_iters': 150, 'K': 6, 'k_sup': 0.9, 'k_raw': 1.0, 'k_cr': 1.0, 'hop_cost': 0.5, 'act_cost': 1.0}


def build_static(m, seat):
    R = m['routes']
    n = len(R)
    d = [[INF] * n for _ in range(n)]
    h = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
        h[i][i] = 0
        for j in range(n):
            if R[i][j] > 0:
                d[i][j] = R[i][j]
                h[i][j] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                nd = d[i][k] + d[k][j]
                nh = h[i][k] + h[k][j]
                if nd < d[i][j] or (nd == d[i][j] and nh < h[i][j]):
                    d[i][j] = nd
                    h[i][j] = nh
    nxt = [[-1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best = None
            for k in range(n):
                if R[i][k] > 0:
                    c = (R[i][k] + d[k][j], 1 + h[k][j])
                    if best is None or c < best:
                        best = c
                        nxt[i][j] = k
    tc = [[d[i][j] + P['hop_cost'] * h[i][j] for j in range(n)] for i in range(n)]
    return {'n': n, 'd': d, 'h': h, 'tc': tc, 'nxt': nxt, 'R': R, 'roles': m['roles'],
            'home': m['homes'][seat], 'ehome': m['homes'][1 - seat], 'C': m['contracts']}


def obs_state(obs, memory):
    own = obs['own']
    rnd = obs['round']
    roles = obs['map']['roles']
    stk = []
    for p, v in enumerate(obs['stocks']):
        if v is not None:
            memory['stk'][p] = (v, rnd)
        if roles[p] in (0, 1):
            s = memory['stk'].get(p)
            if s is None:
                e = min(6, 4 + rnd // 4)
            else:
                e = min(6, s[0] + rnd // 4 - s[1] // 4)
            stk.append(e)
        else:
            stk.append(0)
    return {
        'r': rnd,
        'boats': [[b['position'], b['destination'], b['eta']] + list(b['cargo']) for b in own['boats']],
        'wh': [list(w) for w in own['warehouses']],
        'ws': list(own['workshops']),
        'jobs': list(own['jobs']),
        'claims': list(obs['claims']),
        'stk': stk,
        'tasks': memory.get('tasks', []),
        'last': memory.setdefault('last', []),
    }


def copy_state(st):
    return {
        'r': st['r'],
        'boats': [list(b) for b in st['boats']],
        'wh': [list(w) for w in st['wh']],
        'ws': list(st['ws']),
        'jobs': list(st['jobs']),
        'claims': list(st['claims']),
        'stk': list(st['stk']),
        'tasks': [None if t is None else [t[0], t[1]] for t in st['tasks']],
        'last': list(st['last']),
    }


def task_contract(t):
    for s in t[0][t[1]:]:
        if s[0] == 'V':
            return s[2]
    return None


def policy(S, st, prm, chooser=None, arb=None, preempt_eval=None):
    r = st['r']
    C = S['C']
    d = S['d']
    tc = S['tc']
    roles = S['roles']
    boats = st['boats']
    wh = st['wh']
    W = prm['W']
    target = prm['target']
    home = S['home']
    tasks = st['tasks']
    mult = prm.get('mult') or [1.0] * len(C)
    sv = S.get('surv') if P['use_sv'] else None

    def sw(i, tt):
        if sv is None:
            return 1.0
        t2 = int(r + tt)
        if t2 > 80:
            t2 = 80
        return P['sv_floor'] + (1 - P['sv_floor']) * sv[i][t2]
    ksup = prm.get('ksup', P['k_sup'])
    while len(tasks) < len(boats):
        tasks.append(None)
    open_t = []
    for i in target:
        if st['claims'][i] == -1 and C[i]['deadline'] > r:
            open_t.append(i)
    for bi in range(len(boats)):
        t = tasks[bi]
        if t is None:
            continue
        if t[1] >= len(t[0]):
            tasks[bi] = None
            continue
        for s in t[0][t[1]:]:
            if s[0] == 'V' and (st['claims'][s[2]] != -1 or C[s[2]]['deadline'] <= r):
                tasks[bi] = None
                break
    if chooser is not None and P.get('preempt', 1) and preempt_eval is not None:
        for bi in range(len(boats)):
            t = tasks[bi]
            if t is None or boats[bi][2] > 0 or t[1] == 0:
                continue
            st2 = copy_state(st)
            st2['tasks'][bi] = None
            if preempt_eval(st2) > preempt_eval(st) + 0.05:
                tasks[bi] = None
    reserved = set()
    res_src = {}
    pick_res = 0
    sup_T = 0
    sup_F = 0
    for t in tasks:
        if t is None:
            continue
        for s in t[0][t[1]:]:
            if s[0] == 'V':
                reserved.add(s[2])
            elif s[0] == 'T':
                k = (s[1], s[2])
                res_src[k] = res_src.get(k, 0) + s[3]
            elif s[0] == 'P':
                pick_res += s[2]
            elif s[0] == 'D' and s[1] == W:
                if s[2] == 0:
                    sup_T += s[3]
                elif s[2] == 1:
                    sup_F += s[3]
    cr_open = []
    if W >= 0:
        for i in open_t:
            if C[i]['good'] == 2 and C[i]['deadline'] > r + 1 + d[W][C[i]['destination']]:
                cr_open.append(i)
    g_have = 0
    if W >= 0:
        g_have = wh[W][2] + (1 if st['jobs'][W] > 0 else 0)
        for b in boats:
            g_have += b[5]
    batches = 0
    late_cut = 80
    if W >= 0:
        for i in cr_open:
            pass
        batches = max(0, len(cr_open) - g_have)
        lim = 0
        for i in cr_open:
            if C[i]['deadline'] > r + 8 + d[W][C[i]['destination']]:
                lim += 1
        batches = min(batches, max(0, lim - g_have))
    needT = 0
    needF = 0
    if W >= 0 and batches > 0:
        needT = 2 * batches + (0 if st['ws'][W] else 2)
        needF = batches + (0 if st['ws'][W] else 1)
    shortT = max(0, needT - wh[W][0] - sup_T) if W >= 0 else 0
    shortF = max(0, needF - wh[W][1] - sup_F) if W >= 0 else 0
    for b in boats:
        pass

    def avail(s, g):
        if roles[s] == g:
            v = st['stk'][s]
        else:
            v = wh[s][g]
        return v - res_src.get((s, g), 0)

    for bi, b in enumerate(boats):
        if tasks[bi] is not None or b[2] > 0:
            continue
        p = b[0]
        cg = b[3:6]
        free = CAP - cg[0] - cg[1] - cg[2]
        best = None
        bsc = 0.0
        cands = []
        for i in open_t:
            if i in reserved:
                continue
            c = C[i]
            g = c['good']
            q = c['quantity']
            D = c['destination']
            if cg[g] >= q:
                tm = tc[p][D] + 1
                if r + d[p][D] < c['deadline']:
                    sc = c['points'] * 1.3 * mult[i] * sw(i, tm) / (tm + 1)
                    cands.append((sc, [('V', D, i)]))
                continue
            if g == 2:
                if W < 0 or free < 1:
                    continue
                ga = wh[W][2] - pick_res
                ready = 0
                if ga < 1:
                    if st['jobs'][W] > 0 and ga == 0:
                        ready = st['jobs'][W]
                    else:
                        continue
                arr = max(d[p][W], ready)
                if r + arr + 1 + d[W][D] >= c['deadline']:
                    continue
                tm = max(tc[p][W], ready) + 1 + tc[W][D] + 1
                sc = c['points'] * P['k_cr'] * mult[i] * sw(i, tm) / (tm + 1)
                cands.append((sc, [('P', W, 1), ('V', D, i)]))
            else:
                need = q - cg[g]
                if need > free:
                    continue
                for s in range(S['n']):
                    if s == W and cr_open:
                        continue
                    if avail(s, g) >= need:
                        if r + d[p][s] + 1 + d[s][D] >= c['deadline']:
                            continue
                        tm = tc[p][s] + 1 + tc[s][D] + 1
                        sc = c['points'] * P['k_raw'] * mult[i] * sw(i, tm) / (tm + 1)
                        cands.append((sc, [('T', s, g, need), ('V', D, i)]))
        if (shortT > 0 or shortF > 0) and free > 0:
            for s in range(S['n']):
                ta = 0
                fa = 0
                if roles[s] == 0:
                    ta = min(avail(s, 0), free)
                elif roles[s] == 1:
                    fa = min(avail(s, 1), free)
                elif s != W:
                    at = max(0, min(avail(s, 0), free, shortT))
                    af = max(0, min(avail(s, 1), free, shortF))
                    for ta2 in range(at + 1):
                        fa2 = min(af, free - ta2)
                        if ta2 + fa2 <= 0:
                            continue
                        nact = (1 if ta2 else 0) + (1 if fa2 else 0)
                        tm = tc[p][s] + tc[s][W] + 2 * nact
                        sc = 4.0 * ksup * (ta2 + fa2) / (tm + 1)
                        tl = []
                        if ta2:
                            tl.append(('T', s, 0, ta2))
                        if fa2:
                            tl.append(('T', s, 1, fa2))
                        if ta2:
                            tl.append(('D', W, 0, ta2))
                        if fa2:
                            tl.append(('D', W, 1, fa2))
                        cands.append((sc, tl))
                    continue
                if roles[s] == 0:
                    ta = max(0, min(ta, shortT))
                if roles[s] == 1:
                    fa = max(0, min(fa, shortF))
                if ta + fa <= 0:
                    continue
                useful = min(ta, shortT) + min(fa, shortF)
                if useful <= 0:
                    continue
                nact = (1 if ta else 0) + (1 if fa else 0)
                tm = tc[p][s] + tc[s][W] + 2 * nact
                sc = 4.0 * ksup * useful / (tm + 1)
                if True:
                    tl = []
                    if ta:
                        tl.append(('T', s, 0, ta))
                    if fa:
                        tl.append(('T', s, 1, fa))
                    if ta:
                        tl.append(('D', W, 0, ta))
                    if fa:
                        tl.append(('D', W, 1, fa))
                    cands.append((sc, tl))
        if cands:
            cands.sort(key=lambda x: -x[0])
            if chooser is not None and len(cands) > 1:
                best = chooser(st, bi, cands[:P['K']])
            else:
                best = cands[0][1]
        if best is None and W >= 0 and (cg[0] > 0 or cg[1] > 0) and batches > 0:
            tl = []
            if cg[0]:
                tl.append(('D', W, 0, cg[0]))
            if cg[1]:
                tl.append(('D', W, 1, cg[1]))
            best = tl
        if best is not None:
            tasks[bi] = [best, 0]
            for s in best:
                if s[0] == 'V':
                    reserved.add(s[2])
                elif s[0] == 'T':
                    k = (s[1], s[2])
                    res_src[k] = res_src.get(k, 0) + s[3]
                elif s[0] == 'P':
                    pick_res += s[2]
                elif s[0] == 'D' and s[1] == W:
                    if s[2] == 0:
                        shortT -= s[3]
                    else:
                        shortF -= s[3]

    nxt = S['nxt']
    R = S['R']
    best = None
    bestp = -1
    acts = []
    for bi, b in enumerate(boats):
        if b[2] > 0:
            continue
        p = b[0]
        act = None
        pr = 0
        if p == W:
            w = wh[W]
            if not st['ws'][W]:
                if w[0] >= 2 and w[1] >= 1 and (batches > 0 or g_have < len(cr_open)):
                    act = ('build', bi)
                    pr = 80
            elif st['jobs'][W] == 0 and w[0] >= 2 and w[1] >= 1 and batches > 0:
                act = ('craft', bi)
                pr = 90
        t = tasks[bi]
        while act is None and t is not None and t[1] < len(t[0]):
            s = t[0][t[1]]
            port = s[1]
            if p != port:
                nh = nxt[p][port]
                act = ('sail', bi, nh)
                pr = 50 + R[p][nh]
                break
            k = s[0]
            if k == 'T':
                g = s[2]
                free = CAP - b[3] - b[4] - b[5]
                if roles[p] == g:
                    qq = min(s[3], st['stk'][p], free)
                    if qq > 0:
                        act = ('gather', bi, g, qq)
                else:
                    qq = min(s[3], wh[p][g], free)
                    if qq > 0:
                        act = ('load', bi, g, qq)
                if act is not None:
                    pr = 10
                    break
                tasks[bi] = None
                t = None
                break
            if k == 'D':
                qq = min(s[3], b[3 + s[2]])
                if qq > 0:
                    act = ('unload', bi, s[2], qq)
                    pr = 12
                    break
                t[1] += 1
                continue
            if k == 'V':
                c = C[s[2]]
                if st['claims'][s[2]] == -1 and b[3 + c['good']] >= c['quantity'] and r < c['deadline']:
                    act = ('deliver', bi, s[2])
                    pr = 100
                    break
                tasks[bi] = None
                t = None
                break
            if k == 'P':
                free = CAP - b[3] - b[4] - b[5]
                qq = min(s[2], wh[p][2], free)
                if qq > 0:
                    act = ('load', bi, 2, qq)
                    pr = 11
                    break
                if st['jobs'][p] > 0:
                    break
                tasks[bi] = None
                t = None
                break
        if act is not None:
            if pr < 50:
                k = 1
                L = 0
                if t is not None:
                    pp = p
                    for s2 in t[0][t[1] + 1:]:
                        if s2[1] != pp:
                            L = R[pp][nxt[pp][s2[1]]]
                            break
                        k += 1
                pr += L / k + 0.3 * (r - st['last'][bi] if bi < len(st['last']) else 0)
            acts.append((pr, act))
            if pr > bestp:
                best = act
                bestp = pr
    if arb is not None and len(acts) > 1 and bestp < 100:
        bv = None
        for pr, act in acts:
            st2 = copy_state(st)
            commit(st2, act)
            v = arb(st2, act) + 0.0001 * pr
            if bv is None or v > bv:
                bv = v
                best = act
    commit(st, best)
    return best


def commit(st, best):
    r = st['r']
    boats = st['boats']
    tasks = st['tasks']
    if best is not None:
        while len(st['last']) < len(boats):
            st['last'].append(r)
        if best[0] != 'build' or True:
            if len(best) > 1:
                st['last'][best[1]] = r
    if best is not None and best[0] in ('gather', 'load', 'unload', 'deliver'):
        t = tasks[best[1]]
        if t is not None:
            t[1] += 1
            if t[1] >= len(t[0]):
                tasks[best[1]] = None



STYLES = [(3.5, 0.85, 8), (1.8, 1.15, 0), (1.8, 0.85, 0), (3.5, 0.0, 0), (0.0, 0.85, 8)]
M64 = (1 << 64) - 1


def style_of(seed, game):
    perm = [0, 1, 2, 3, 4]
    x = (seed ^ 0x5354594c45534554) & M64
    n = 5
    while n != 1:
        x = (x + 0x9e3779b97f4a7c15) & M64
        z = x
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & M64
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & M64
        z = z ^ (z >> 31)
        j = z % n
        perm[n - 1], perm[j] = perm[j], perm[n - 1]
        n -= 1
    return perm[game]


class PS(object):
    __slots__ = ('boats', 'wh', 'ws', 'jobs', 'stk', 'claims', 'score', 't', 'valid', 'acts')

    def copy(self):
        s = PS()
        s.boats = [list(b) for b in self.boats]
        s.wh = [list(w) for w in self.wh]
        s.ws = list(self.ws)
        s.jobs = list(self.jobs)
        s.stk = list(self.stk)
        s.claims = list(self.claims)
        s.score = self.score
        s.t = self.t
        s.valid = self.valid
        s.acts = self.acts
        return s


class Prof(object):
    def __init__(self, m, seat, style):
        self.R = m['routes']
        self.n = len(self.R)
        self.roles = m['roles']
        self.C = [(c['good'], c['quantity'], c['destination'], c['points'], c['deadline']) for c in m['contracts']]
        self.seat = seat
        self.home = m['homes'][seat]
        self.wmat, self.alpha, self.bonus = STYLES[style]
        n = self.n
        INF = 10 ** 6
        d = [[0 if i == j else (self.R[i][j] if self.R[i][j] else INF) for j in range(n)] for i in range(n)]
        nx = [[j if self.R[i][j] else -1 for j in range(n)] for i in range(n)]
        for i in range(n):
            nx[i][i] = -1
        for k in range(n):
            dk = d[k]
            for i in range(n):
                di = d[i]
                dik = di[k]
                if dik >= INF:
                    continue
                for j in range(n):
                    v = dik + dk[j]
                    if v < di[j]:
                        di[j] = v
                        nx[i][j] = nx[i][k]
        self.D = d
        self.NH = nx
        self.est = [4] * n
        self.cache = {}
        self.last = -1
        self.queue = []
        self.planned = []

    def legal(self, s, a):
        op = a[0]
        if op == 'wait':
            return True
        b = s.boats[a[1]]
        if b[2] != 0:
            return False
        p = b[0]
        if op == 'sail':
            return self.R[p][a[2]] > 0
        if op == 'deliver':
            g, q, dst, pts, dl = self.C[a[2]]
            return s.claims[a[2]] == -1 and dst == p and b[3 + g] >= q and dl > s.t
        if op == 'load':
            g = a[2]
            q = a[3]
            return 1 <= q and q <= 3 - b[3] - b[4] - b[5] and q <= s.wh[p][g]
        if op == 'unload':
            return 1 <= a[3] <= b[3 + a[2]]
        if op == 'gather':
            g = a[2]
            q = a[3]
            return self.roles[p] == g and 1 <= q and q <= 3 - b[3] - b[4] - b[5] and q <= s.stk[p]
        if op == 'build':
            w = s.wh[p]
            return self.roles[p] == 2 and not s.ws[p] and w[0] >= 2 and w[1] >= 1
        if op == 'craft':
            w = s.wh[p]
            return s.ws[p] and s.jobs[p] == 0 and w[0] >= 2 and w[1] >= 1
        return False

    def apply(self, s, a, mode=0):
        if not s.valid or s.t > 159:
            s.valid = False
            return
        op = a[0]
        if mode and op == 'deliver':
            b = s.boats[a[1]]
            g, q, dst, pts, dl = self.C[a[2]]
            if b[2] != 0 or b[0] != dst or b[3 + g] < q or s.claims[a[2]] != -1:
                s.valid = False
                return
        elif not self.legal(s, a):
            s.valid = False
            return
        if op != 'wait':
            b = s.boats[a[1]]
            if op == 'sail':
                b[1] = a[2]
                b[2] = self.R[b[0]][a[2]]
            elif op == 'gather':
                s.stk[b[0]] -= a[3]
                b[3 + a[2]] += a[3]
            elif op == 'load':
                s.wh[b[0]][a[2]] -= a[3]
                b[3 + a[2]] += a[3]
            elif op == 'unload':
                s.wh[b[0]][a[2]] += a[3]
                b[3 + a[2]] -= a[3]
            elif op == 'build':
                w = s.wh[b[0]]
                w[0] -= 2
                w[1] -= 1
                s.ws[b[0]] = True
            elif op == 'craft':
                w = s.wh[b[0]]
                w[0] -= 2
                w[1] -= 1
                s.jobs[b[0]] = 3
            elif op == 'deliver':
                g, q, dst, pts, dl = self.C[a[2]]
                b[3 + g] -= q
                s.score += pts
                s.claims[a[2]] = self.seat
        for b in s.boats:
            if b[2] != 0:
                b[2] -= 1
                if b[2] == 0:
                    b[0] = b[1]
                    b[1] = -1
        jobs = s.jobs
        for p in range(self.n):
            if jobs[p] != 0:
                jobs[p] -= 1
                if jobs[p] == 0:
                    s.wh[p][2] += 1
        s.t += 1
        if (s.t & 3) == 0:
            roles = self.roles
            stk = s.stk
            for p in range(self.n):
                if 0 <= roles[p] < 2:
                    v = stk[p] + 1
                    stk[p] = 6 if v > 6 else v
        s.acts = (s.acts, a)

    def ff(self, s, k):
        if k <= 0 or not s.valid:
            return
        if s.t + k > 160:
            s.valid = False
            return
        for b in s.boats:
            if b[2] != 0:
                if b[2] <= k:
                    b[0] = b[1]
                    b[1] = -1
                    b[2] = 0
                else:
                    b[2] -= k
        jobs = s.jobs
        for p in range(self.n):
            if jobs[p] != 0:
                if jobs[p] <= k:
                    jobs[p] = 0
                    s.wh[p][2] += 1
                else:
                    jobs[p] -= k
        cnt = (s.t + k) // 4 - s.t // 4
        s.t += k
        if cnt:
            roles = self.roles
            stk = s.stk
            for p in range(self.n):
                if 0 <= roles[p] < 2:
                    v = stk[p] + cnt
                    stk[p] = 6 if v > 6 else v

    def move(self, s, bi, port):
        if not s.valid:
            return
        b = s.boats[bi]
        e = b[2]
        if e != 0:
            self.ff(s, e)
        NH = self.NH
        R = self.R
        while s.valid:
            p = b[0]
            if p == port:
                return
            nh = NH[p][port]
            if nh < 0 or s.t > 159:
                s.valid = False
                return
            s.acts = (s.acts, ('sail', bi, nh))
            L = R[p][nh]
            b[1] = nh
            b[2] = L
            self.ff(s, L)

    def collect(self, st, bi, need, dest, mode):
        s = st.copy()
        b = s.boats[bi]
        if max(need[0], b[3]) + max(need[1], b[4]) + max(need[2], b[5]) > 3:
            tgt = b[1] if b[2] != 0 else b[0]
            self.move(s, bi, tgt)
            for g in range(3):
                c = s.boats[bi][3 + g]
                if need[g] < c:
                    self.apply(s, ('unload', bi, g, c - need[g]))
        if not s.valid:
            return None
        b = s.boats[bi]
        short = [need[g] - b[3 + g] if need[g] - b[3 + g] > 0 else 0 for g in range(3)]
        goods = [g for g in range(3) if short[g] > 0]
        best = [None]
        D = self.D
        roles = self.roles

        def rec(s, k):
            if not s.valid:
                return
            if best[0] is not None and not (s.t < best[0].t):
                return
            if k == len(goods):
                s2 = s.copy()
                self.move(s2, bi, dest)
                if mode:
                    for g in range(3):
                        if need[g] != 0:
                            self.apply(s2, ('unload', bi, g, need[g]))
                if s2.valid and (best[0] is None or s2.t < best[0].t):
                    best[0] = s2
                return
            g = goods[k]
            q = short[g]
            cands = []
            for i in range(self.n):
                if mode and dest == i:
                    continue
                if s.wh[i][g] < q:
                    continue
                cands.append((i, True))
            for i in range(self.n):
                if (not mode or dest != i) and roles[i] == g and q <= s.stk[i]:
                    cands.append((i, False))
            bb = s.boats[bi]
            loc = bb[1] if bb[2] != 0 else bb[0]
            Dl = D[loc]
            cands.sort(key=lambda x: Dl[x[0]] + D[x[0]][dest])
            for i, isload in cands[:5]:
                s2 = s.copy()
                self.move(s2, bi, i)
                self.apply(s2, ('load' if isload else 'gather', bi, g, q))
                rec(s2, k + 1)

        perm = list(goods)
        while True:
            goods = perm
            rec(s, 0)
            if len(perm) <= 1 or not next_perm(perm):
                break
        return best[0]

    def matval_pen(self, s, p, w):
        wh = s.wh[p]
        f = wh[1] - 1
        t = wh[0] - 2
        return s.t - ((f if f > 0 else 0) * 1.5 + (t if t > 0 else 0)) * w

    def prep(self, st, p):
        needT = 2 - st.wh[p][0]
        needF = 1 - st.wh[p][1]
        if needT < 0:
            needT = 0
        if needF < 0:
            needF = 0
        best = [None, 1e30, -1]
        nb = len(st.boats)

        def consider(s, bi):
            if s is None or not s.valid:
                return
            v = self.matval_pen(s, p, self.wmat)
            if v < best[1]:
                best[0] = s
                best[1] = v
                best[2] = bi

        if needT + needF == 0:
            for bi in range(nb):
                s = st.copy()
                self.move(s, bi, p)
                consider(s, bi)
            return best[0], best[2]
        for bi in range(nb):
            consider(self.collect(st, bi, [needT, needF, 0], p, 1), bi)
            if needT != 0 and needF == 0:
                consider(self.collect(st, bi, [3, 0, 0], p, 1), bi)
            if needF != 0 and needT == 0:
                consider(self.collect(st, bi, [0, 3, 0], p, 1), bi)
        if needT != 0 and needF != 0 and st.t < 50:
            for first, second in (([3, 0, 0], [0, 3, 0]), ([0, 3, 0], [3, 0, 0])):
                for i in range(nb):
                    s1 = self.collect(st, i, first, p, 1)
                    if s1 is None:
                        continue
                    for bi in range(nb):
                        consider(self.collect(s1, bi, second, p, 1), bi)
        return best[0], best[2]

    def matval(self, s):
        v = 0.0
        for p in range(self.n):
            if s.ws[p]:
                w = s.wh[p]
                v += (w[1] if w[1] < 2 else 2) * 1.5 + (w[0] if w[0] < 4 else 4)
        return v

    def crafted(self, st, dest):
        best = None
        bv = 1e30
        bb = -1
        nb = len(st.boats)
        for bi in range(nb):
            if st.boats[bi][5] != 0:
                s = st.copy()
                self.move(s, bi, dest)
                if s.valid:
                    v = s.t - self.matval(s) * 0.5
                    if v < bv:
                        bv = v
                        best = s
                        bb = bi
        for p in range(self.n):
            if st.wh[p][2] != 0 or st.jobs[p] != 0:
                for bi in range(nb):
                    s = st.copy()
                    self.move(s, bi, p)
                    if s.valid and s.jobs[p] != 0:
                        self.ff(s, s.jobs[p])
                    if not s.valid:
                        continue
                    b = s.boats[bi]
                    if b[3] + b[4] + b[5] < 3:
                        self.apply(s, ('load', bi, 2, 1))
                        self.move(s, bi, dest)
                        if s.valid:
                            v = s.t - self.matval(s) * 0.5
                            if v < bv:
                                bv = v
                                best = s
                                bb = bi
        for p in range(self.n):
            if self.roles[p] != 2:
                continue
            s0 = st.copy()
            if not s0.ws[p]:
                r, rb = self.prep(s0, p)
                if r is None:
                    continue
                s0 = r
                self.apply(s0, ('build', rb))
            if not s0.valid or s0.jobs[p] != 0:
                continue
            r, rb = self.prep(s0, p)
            if r is None:
                continue
            s0 = r
            self.apply(s0, ('craft', rb))
            for bi in range(nb):
                s = s0.copy()
                self.move(s, bi, p)
                if s.valid and s.jobs[p] != 0:
                    self.ff(s, s.jobs[p])
                if s.valid:
                    b = s.boats[bi]
                    if b[3] + b[4] + b[5] < 3:
                        self.apply(s, ('load', bi, 2, 1))
                        self.move(s, bi, dest)
                        if s.valid:
                            v = s.t - self.matval(s) * 0.5
                            if v < bv:
                                bv = v
                                best = s
                                bb = bi
        return best, bb

    def contract_plan(self, st, c):
        if st.claims[c] != -1:
            return None
        g, q, dst, pts, dl = self.C[c]
        if g == 2:
            s, bi = self.crafted(st, dst)
            if s is None:
                return None
        else:
            need = [0, 0, 0]
            need[g] = q
            s = None
            bi = -1
            for b in range(len(st.boats)):
                r = self.collect(st, b, need, dst, 0)
                if r is not None and (s is None or r.t < s.t):
                    s = r
                    bi = b
            if s is None:
                return None
        s = s.copy()
        self.apply(s, ('deliver', bi, c), 1)
        if not s.valid:
            return None
        return s

    def merge(self, base, acts, variant):
        acts = flat(acts)
        s = base.copy()
        s.acts = None
        nb = len(s.boats)
        qs = [[] for _ in range(3)]
        for a in acts:
            if a[0] == 'wait':
                continue
            b = a[1] if a[1] >= 0 else 0
            qs[b].append(a)
        ptr = [0, 0, 0]
        while s.t < 80:
            if all(ptr[k] >= len(qs[k]) for k in range(3)):
                return s
            chosen = -1
            for j in range(variant, variant + nb):
                b = j % nb
                if ptr[b] < len(qs[b]):
                    a = qs[b][ptr[b]]
                    if self.legal(s, a):
                        if chosen < 0 or a[0] == 'deliver':
                            chosen = b
                        if a[0] == 'deliver':
                            break
            if chosen < 0:
                busy = any(b[2] > 0 for b in s.boats) or any(j > 0 for j in s.jobs)
                if not busy:
                    return None
                self.apply(s, ('wait',))
            else:
                a = qs[chosen][ptr[chosen]]
                ptr[chosen] += 1
                self.apply(s, a)
            if not s.valid:
                return None
        if all(ptr[k] >= len(qs[k]) for k in range(3)):
            return s
        return None

    def node_score(self, base, s):
        mv = self.matval(s)
        bonus = 0
        if not any(base.ws) and any(s.ws):
            bonus = self.bonus
        return (mv * 0.1 + (s.score - base.score + bonus)) / ((s.t - base.t + 3) ** self.alpha)

    def replan(self, base):
        beam = [(base, [])]
        best = None
        bscore = None
        for depth in range(4):
            children = []
            for node, cl in beam:
                for c in range(len(self.C)):
                    if node.claims[c] != -1 or not (base.t < self.C[c][4]):
                        continue
                    cp = self.contract_plan(node, c)
                    if cp is None:
                        continue
                    for variant in (0, 1):
                        mg = self.merge(base, cp.acts, variant)
                        if mg is None:
                            continue
                        sc = self.node_score(base, mg)
                        if best is None or bscore < sc:
                            best = (mg, cl + [c])
                            bscore = sc
                        children.append((sc, len(children), mg, cl + [c]))
            if not children:
                break
            children.sort(key=lambda x: (-x[0], x[2].t, x[1]))
            beam = [(x[2], x[3]) for x in children[:8]]
        return best


    def legal_side(self, side, stk, claims, t, a):
        op = a[0]
        if op == 'wait':
            return True
        bs = side['boats']
        if a[1] >= len(bs):
            return False
        b = bs[a[1]]
        if b[2] != 0:
            return False
        p = b[0]
        if op == 'sail':
            return self.R[p][a[2]] > 0
        if op == 'deliver':
            g, q, dst, pts, dl = self.C[a[2]]
            return claims[a[2]] == -1 and dst == p and b[3 + g] >= q and dl > t
        if op == 'load':
            q = a[3]
            return 1 <= q and q <= 3 - b[3] - b[4] - b[5] and q <= side['wh'][p][a[2]]
        if op == 'unload':
            return 1 <= a[3] <= b[3 + a[2]]
        if op == 'gather':
            q = a[3]
            return self.roles[p] == a[2] and 1 <= q and q <= 3 - b[3] - b[4] - b[5] and q <= stk[p]
        if op == 'build':
            w = side['wh'][p]
            return self.roles[p] == 2 and not side['ws'][p] and w[0] >= 2 and w[1] >= 1
        if op == 'craft':
            w = side['wh'][p]
            return side['ws'][p] and side['jobs'][p] == 0 and w[0] >= 2 and w[1] >= 1
        return False

    def step_fast(self, side, stk, claims, t, vis):
        if t == 0:
            self.last = -1
            self.est = [4] * self.n
            self.queue = []
            self.planned = []
        inc = t // 4 - (self.last // 4 if self.last >= 0 else 0)
        est = self.est
        for p in range(self.n):
            v = est[p] + inc
            est[p] = 6 if v > 6 else v
            if vis[p] is not None:
                est[p] = vis[p]
        self.last = t
        for c in self.planned:
            if claims[c] != -1 and claims[c] != self.seat:
                self.queue = []
        nC = len(self.C)
        for bi, b in enumerate(side['boats']):
            if b[2] != 0:
                continue
            for c in range(nC):
                g, q, dst, pts, dl = self.C[c]
                if claims[c] == -1 and dst == b[0] and b[3 + g] >= q and t < dl:
                    self.queue = []
                    self.planned = []
                    return ('deliver', bi, c)
        if self.queue and self.legal_side(side, stk, claims, t, self.queue[0]):
            return self.queue.pop(0)
        self.queue = []
        base = PS()
        base.boats = [list(b) for b in side['boats']]
        base.wh = [list(w) for w in side['wh']]
        base.ws = list(side['ws'])
        base.jobs = list(side['jobs'])
        base.stk = list(est)
        base.claims = list(claims)
        base.score = side['score']
        base.t = t
        base.valid = True
        base.acts = None
        key = (base.t, tuple(tuple(b) for b in base.boats), tuple(tuple(x) for x in base.wh), tuple(base.ws),
               tuple(base.jobs), tuple(base.stk), tuple(base.claims), base.score)
        res = self.cache.get(key)
        if res is None:
            res = self.replan(base)
            if res is None:
                res = (None, None)
            self.cache[key] = res
        if res[0] is not None and res[0].acts:
            self.planned = res[1]
            self.queue = flat(res[0].acts)
            if self.legal_side(side, stk, claims, t, self.queue[0]):
                return self.queue.pop(0)
        self.queue = []
        return ('wait',)

    def step(self, obs_round, boats, wh, ws, jobs, score, claims, visible_stocks, legal):
        if obs_round == 0:
            self.last = -1
            self.est = [4] * self.n
            self.queue = []
            self.planned = []
        inc = obs_round // 4 - (self.last // 4 if self.last >= 0 else 0)
        for p in range(self.n):
            v = self.est[p] + inc
            self.est[p] = 6 if v > 6 else v
            if visible_stocks[p] is not None:
                self.est[p] = visible_stocks[p]
        self.last = obs_round
        for c in self.planned:
            if claims[c] != -1 and claims[c] != self.seat:
                self.queue = []
        for a in legal:
            if a[0] == 'deliver':
                self.queue = []
                self.planned = []
                return a
        if self.queue and self.queue[0] in legal:
            return self.queue.pop(0)
        self.queue = []
        base = PS()
        base.boats = [list(b) for b in boats]
        base.wh = [list(w) for w in wh]
        base.ws = list(ws)
        base.jobs = list(jobs)
        base.stk = list(self.est)
        base.claims = list(claims)
        base.score = score
        base.t = obs_round
        base.valid = True
        base.acts = None
        key = (base.t, tuple(tuple(b) for b in base.boats), tuple(tuple(x) for x in base.wh), tuple(base.ws),
               tuple(base.jobs), tuple(base.stk), tuple(base.claims), base.score)
        res = self.cache.get(key)
        if res is None:
            res = self.replan(base)
            if res is None:
                res = (None, None)
            self.cache[key] = res
        if res[0] is None:
            res = None
        if res is not None and res[0].acts:
            self.planned = res[1]
            self.queue = flat(res[0].acts)
            if self.queue[0] in legal:
                return self.queue.pop(0)
        self.queue = []
        return legal[0]


def flat(acts):
    out = []
    while acts is not None:
        out.append(acts[1])
        acts = acts[0]
    out.reverse()
    return out


def next_perm(a):
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        a.reverse()
        return False
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return True




def init_side(m, seat):
    home = m['homes'][seat]
    n = len(m['routes'])
    s = {'boats': [[home, -1, 0, 0, 0, 0], [home, -1, 0, 0, 0, 0]], 'wh': [[0, 0, 0] for _ in range(n)],
         'ws': [False] * n, 'jobs': [0] * n, 'score': 0}
    s['wh'][home] = [5, 3, 0]
    return s


def legal_list(m, side, stk, claims, t):
    R = m['routes']
    roles = m['roles']
    C = m['contracts']
    n = len(R)
    out = [('wait',)]
    for bi, b in enumerate(side['boats']):
        if b[2] != 0:
            continue
        p = b[0]
        free = 3 - b[3] - b[4] - b[5]
        for j in range(n):
            if R[p][j]:
                out.append(('sail', bi, j))
        if roles[p] in (0, 1):
            for q in range(1, min(free, stk[p]) + 1):
                out.append(('gather', bi, roles[p], q))
        for g in range(3):
            for q in range(1, min(free, side['wh'][p][g]) + 1):
                out.append(('load', bi, g, q))
        for g in range(3):
            for q in range(1, b[3 + g] + 1):
                out.append(('unload', bi, g, q))
        w = side['wh'][p]
        if roles[p] == 2 and not side['ws'][p] and w[0] >= 2 and w[1] >= 1:
            out.append(('build', bi))
        if side['ws'][p] and side['jobs'][p] == 0 and w[0] >= 2 and w[1] >= 1:
            out.append(('craft', bi))
        for c, k in enumerate(C):
            if claims[c] == -1 and k['destination'] == p and b[3 + k['good']] >= k['quantity'] and t < k['deadline']:
                out.append(('deliver', bi, c))
    return out


class World(object):
    def __init__(self, m, seat, style):
        self.m = m
        self.seat = seat
        self.n = len(m['routes'])
        self.sides = [init_side(m, 0), init_side(m, 1)]
        self.stk = [4 if m['roles'][p] in (0, 1) else 0 for p in range(self.n)]
        self.claims = [-1] * len(m['contracts'])
        self.t = 0
        self.prof = Prof(m, 1 - seat, style)
        self.claim_time = [None] * len(m['contracts'])

    def prof_action(self):
        ps = 1 - self.seat
        side = self.sides[ps]
        home = self.m['homes'][ps]
        vis = [None] * self.n
        vis[home] = self.stk[home]
        for b in side['boats']:
            if b[2] == 0:
                vis[b[0]] = self.stk[b[0]]
        return self.prof.step_fast(side, self.stk, self.claims, self.t, vis)

    def apply_one(self, si, a, locks):
        if a[0] == 'wait':
            return True
        side = self.sides[si]
        b = side['boats'][a[1]]
        op = a[0]
        if op == 'sail':
            key = (min(b[0], a[2]), max(b[0], a[2]))
            if key in locks:
                return False
            locks.add(key)
            b[1] = a[2]
            b[2] = self.m['routes'][b[0]][a[2]]
        elif op == 'gather':
            if self.stk[b[0]] < a[3]:
                return False
            self.stk[b[0]] -= a[3]
            b[3 + a[2]] += a[3]
        elif op == 'load':
            side['wh'][b[0]][a[2]] -= a[3]
            b[3 + a[2]] += a[3]
        elif op == 'unload':
            side['wh'][b[0]][a[2]] += a[3]
            b[3 + a[2]] -= a[3]
        elif op == 'build':
            w = side['wh'][b[0]]
            w[0] -= 2
            w[1] -= 1
            side['ws'][b[0]] = True
        elif op == 'craft':
            w = side['wh'][b[0]]
            w[0] -= 2
            w[1] -= 1
            side['jobs'][b[0]] = 3
        elif op == 'deliver':
            c = a[2]
            if self.claims[c] != -1:
                return False
            k = self.m['contracts'][c]
            b[3 + k['good']] -= k['quantity']
            side['score'] += k['points']
            self.claims[c] = si
            self.claim_time[c] = self.t
        elif op == 'build_boat':
            home = self.m['homes'][si]
            w = side['wh'][home]
            w[0] -= 4
            w[1] -= 2
            side['boats'].append([home, -1, 0, 0, 0, 0])
        return True

    def step(self, a_us, a_prof):
        acts = [None, None]
        acts[self.seat] = a_us
        acts[1 - self.seat] = a_prof
        first = self.t % 2
        locks = set()
        res = [True, True]
        for si in (first, 1 - first):
            res[si] = self.apply_one(si, acts[si], locks)
        for side in self.sides:
            for b in side['boats']:
                if b[2] != 0:
                    b[2] -= 1
                    if b[2] == 0:
                        b[0] = b[1]
                        b[1] = -1
            jobs = side['jobs']
            for p in range(self.n):
                if jobs[p] != 0:
                    jobs[p] -= 1
                    if jobs[p] == 0:
                        side['wh'][p][2] += 1
        self.t += 1
        if self.t % 4 == 0:
            roles = self.m['roles']
            for p in range(self.n):
                if roles[p] in (0, 1) and self.stk[p] < 6:
                    self.stk[p] += 1
        return res


def choose_W(S):
    n = S['n']
    d = S['d']
    home = S['home']
    C = S['C']
    best = None
    for p in range(n):
        if S['roles'][p] != 2:
            continue
        c = 3 * d[home][p]
        for k in C:
            if k['good'] == 2:
                c += d[p][k['destination']]
        for g, wgt in ((0, 2), (1, 1)):
            c += wgt * min(d[p][q] for q in range(n) if S['roles'][q] == g)
        if best is None or c < best[0]:
            best = (c, p)
    return best[1]




def prof_clone(pf):
    c = Prof.__new__(Prof)
    c.__dict__.update(pf.__dict__)
    c.est = list(pf.est)
    c.queue = list(pf.queue)
    c.planned = list(pf.planned)
    return c


def world_copy(w):
    c = World.__new__(World)
    c.m = w.m
    c.seat = w.seat
    c.n = w.n
    c.sides = [{'boats': [list(b) for b in s['boats']], 'wh': [list(x) for x in s['wh']], 'ws': list(s['ws']),
                'jobs': list(s['jobs']), 'score': s['score']} for s in w.sides]
    c.stk = list(w.stk)
    c.claims = list(w.claims)
    c.t = w.t
    c.prof = prof_clone(w.prof)
    c.claim_time = list(w.claim_time)
    return c


def to_tuple(a):
    op = a['op']
    if op == 'wait':
        return ('wait',)
    if op == 'sail':
        return ('sail', a['boat'], a['destination'])
    if op in ('load', 'gather', 'unload'):
        return (op, a['boat'], a['good'], a['quantity'])
    if op == 'deliver':
        return ('deliver', a['boat'], a['contract'])
    if op == 'build_workshop':
        return ('build', a['boat'])
    if op == 'craft':
        return ('craft', a['boat'])
    return (op,)


def to_engine(a):
    if a is None:
        return {'op': 'wait'}
    op = a[0]
    if op == 'sail':
        return {'boat': a[1], 'destination': a[2], 'op': 'sail'}
    if op in ('gather', 'load', 'unload'):
        return {'boat': a[1], 'good': a[2], 'op': op, 'quantity': a[3]}
    if op == 'deliver':
        return {'boat': a[1], 'contract': a[2], 'op': 'deliver'}
    if op == 'build':
        return {'boat': a[1], 'op': 'build_workshop'}
    if op == 'craft':
        return {'boat': a[1], 'op': 'craft'}
    if op == 'build_boat':
        return {'op': 'build_boat'}
    return {'op': 'wait'}


def consistent(w, obs):
    seat = w.seat
    mine = w.sides[seat]
    own = obs['own']
    if own['score'] != mine['score'] or obs['claims'] != w.claims or obs['opponent_score'] != w.sides[1 - seat]['score']:
        return False
    if [[b['position'], b['destination'], b['eta']] + list(b['cargo']) for b in own['boats']] != mine['boats']:
        return False
    if own['warehouses'] != mine['wh'] or own['jobs'] != mine['jobs'] or own['workshops'] != mine['ws']:
        return False
    for p, v in enumerate(obs['stocks']):
        if v is not None and v != w.stk[p]:
            return False
    vis = obs['visible']
    exp = sorted((bi, b[0], b[1], b[2]) for bi, b in enumerate(w.sides[1 - seat]['boats'])
                 if vis[b[0]] or (b[2] > 0 and vis[b[1]]))
    got = sorted((e['boat'], e['position'], e['destination'], e['eta']) for e in obs['enemy_boats'])
    return exp == got


def policy_view(w, tasks, last):
    side = w.sides[w.seat]
    return {'r': w.t, 'boats': side['boats'], 'wh': side['wh'], 'ws': side['ws'], 'jobs': side['jobs'],
            'claims': w.claims, 'stk': w.stk, 'tasks': tasks, 'last': last}


def jroll(S, prm, w0, a_prof0, st_pol, forced):
    w = world_copy(w0)
    tasks = [None if t is None else [t[0], t[1]] for t in st_pol['tasks']]
    last = list(st_pol['last'])
    first = True
    while w.t < 80:
        if first:
            ap = a_prof0
        else:
            ap = w.prof_action()
        if first and forced is not None:
            au = forced
        else:
            st = policy_view(w, tasks, last)
            au = policy(S, st, prm)
            tasks = st['tasks']
            if au is None:
                au = ('wait',)
        first = False
        w.step(au, ap)
    us = w.sides[w.seat]['score']
    them = w.sides[1 - w.seat]['score']
    return us - them, w


def agg(vals):
    mn = min(vals)
    return sum(vals) / len(vals) + P['robust'] * mn


def predicted_surv(S, worlds_final, seat):
    nC = len(S['C'])
    out = []
    for c in range(nC):
        row = [0.0] * 81
        for t in range(81):
            alive = 0.0
            for wf in worlds_final:
                ct = wf.claim_time[c]
                if wf.claims[c] == seat or ct is None or t <= ct:
                    alive += 1.0
            row[t] = alive / len(worlds_final)
        out.append(row)
    return out


def decide(obs, memory):
    m = obs['map']
    seat = obs['seat']
    if obs['round'] == 0 or 'worlds' not in memory:
        memory.clear()
        memory['S'] = build_static(m, seat)
        memory['worlds'] = [World(m, seat, st) for st in range(5)]
        memory['tasks'] = []
        memory['last'] = []
        S = memory['S']
        memory['prm'] = {'W': choose_W(S), 'target': list(range(len(S['C'])))}
        memory['fallback'] = False
        memory['stk'] = {}
    S = memory['S']
    prm = memory['prm']
    if not memory['fallback']:
        worlds = [w for w in memory['worlds'] if consistent(w, obs)]
        if not worlds:
            memory['fallback'] = True
        memory['worlds'] = worlds
    if memory['fallback']:
        st = obs_state(obs, memory)
        a = policy(S, st, prm)
        memory['tasks'] = st['tasks']
        return to_engine(a)
    worlds = memory['worlds']
    used = memory.get('used', 0.0)
    level = 0 if used < P['tb1'] else (1 if used < P['tb2'] else 2)
    mw = P['maxw'] if level == 0 else 1
    evw = worlds[:mw]
    aps = [w.prof_action() for w in worlds]
    if level == 2:
        st = policy_view(worlds[0], memory['tasks'], memory['last'])
        st = copy_state(st)
        a = policy(S, st, prm)
        memory['tasks'] = st['tasks']
        memory['last'] = st['last']
        au = a if a is not None else ('wait',)
        for w, ap in zip(worlds, aps):
            w.step(au, ap)
        return to_engine(a)
    apmap = dict(zip([id(w) for w in worlds], aps))
    base_st = policy_view(worlds[0], memory['tasks'], memory['last'])
    finals = []
    for w in evw:
        v, wf = jroll(S, prm, w, apmap[id(w)], base_st, None)
        finals.append(wf)
    S['surv'] = predicted_surv(S, finals, seat)

    def evalf(st2, forced):
        vals = []
        for w in evw:
            v, wf = jroll(S, prm, w, apmap[id(w)], st2, forced)
            vals.append(v + 0.01 * wf.sides[seat]['score'])
        return agg(vals)

    def chooser(st, bi, cands):
        best = None
        for sc, task in cands:
            st2 = copy_state(st)
            st2['tasks'][bi] = [task, 0]
            v = evalf(st2, None) + 0.0001 * sc
            if best is None or v > best[0]:
                best = (v, task)
        return best[1]

    def arb(st2, act):
        return evalf(st2, act)

    def preempt_eval(st2):
        return evalf(st2, None)

    st = policy_view(worlds[0], memory['tasks'], memory['last'])
    st = copy_state(st)
    pe = preempt_eval if (level == 0 and obs['round'] % P['pre_every'] == 0) else None
    a = policy(S, st, prm, chooser, arb if level == 0 else None, pe)
    memory['tasks'] = st['tasks']
    memory['last'] = st['last']
    au = a if a is not None else ('wait',)
    for w, ap in zip(worlds, aps):
        w.step(au, ap)
    return to_engine(a)


def agent(obs, memory):
    import time as _t
    t0 = _t.time()
    legal = obs['legal_actions']
    try:
        if obs['round'] == 0:
            memory.clear()
        a = decide(obs, memory)
        memory['used'] = memory.get('used', 0.0) + (_t.time() - t0)
        if a in legal:
            return a
        print('not legal', a, file=sys.stderr)
        memory['fallback'] = True
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        memory['fallback'] = True
    for x in legal:
        if x['op'] == 'wait':
            return x
    return legal[0]


def serve():
    protocol = sys.stdout
    sys.stdout = sys.stderr
    memory = {}
    for line in sys.stdin:
        observation = json.loads(line)['observation']
        if observation['round'] == 0:
            memory = {}
        protocol.write(json.dumps(agent(observation, memory)) + '\n')
        protocol.flush()


if __name__ == '__main__':
    serve()
