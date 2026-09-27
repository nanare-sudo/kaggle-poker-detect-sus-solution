"""Neural action-reference features, network and fold trainer.

Loaded by the training and inference runners with their explicit runtime state.
The legacy feature builder retains final board length; RCLEAN replaces column 10
with visible board length before fitting and prediction.
"""
def _table_feats(t):
    import action_context as Q
    from engine import A
    ht, hp, st = np.asarray(A("hand_table")), np.asarray(A("hand_phase")), np.asarray(A("hand_started_at"))
    off = np.asarray(A("act_offsets")); nbd = np.asarray(A("hand_n_board")).astype(np.float32)
    a_act, a_str, a_seat, a_amt = A("act_action"), A("act_street"), A("act_seat"), A("act_amount")
    a_hidx, a_ply, bb = A("act_hidx"), A("act_player"), np.asarray(A("hand_bb"))
    h1, h2, board = np.asarray(A("seat_hole1")), np.asarray(A("seat_hole2")), np.asarray(A("hand_board"))
    out = []
    for ph in (0, 1):
        sel = np.flatnonzero((ht == t) & (hp == ph)); sel = sel[np.argsort(st[sel], kind="stable")]
        X, cls, idx = Q.features(sel)
        idx = np.asarray(idx).astype(np.int64)
        assert len(idx) == int((off[sel + 1] - off[sel]).sum())
        hid = np.asarray(a_hidx[idx]).astype(np.int64)
        X[:, 10] = nbd[hid]                                       # Legacy reference uses final board length; RCLEAN replaces this column.
        seat = np.asarray(a_seat[idx]).astype(np.int64); strt = np.asarray(a_str[idx]).astype(np.int64)
        pos = idx - off[hid]                                      # Position der Aktion in ihrer Hand
        hc = np.full((len(idx), H, 3), -1, np.int8); ha = np.zeros((len(idx), H), np.float16)
        for k in range(1, H + 1):
            ok = pos >= k; i = np.flatnonzero(ok); j = idx[ok] - k
            hc[i, H - k, 0] = a_act[j]; hc[i, H - k, 1] = a_str[j]
            hc[i, H - k, 2] = ((np.asarray(a_seat[j]).astype(np.int64) - seat[ok]) % 6).astype(np.int8)
            ha[i, H - k] = np.log1p(np.asarray(a_amt[j], np.float32) / np.maximum(bb[hid[ok]].astype(np.float32), 1)).astype(np.float16)
        c7 = np.full((len(idx), 7), -1, np.int8)                 # Hole Cards + sichtbares Board (Code rank*4+suit)
        c7[:, 0] = h1[hid, seat]; c7[:, 1] = h2[hid, seat]
        vis = np.array([0, 3, 4, 5])[strt]; B = board[hid]
        for jj in range(5):
            c7[:, 2 + jj] = np.where(vis > jj, B[:, jj], -1)
        out.append((t, ph, dict(idx=idx, X=np.clip(X, -6e4, 6e4).astype(np.float16), cls=cls.astype(np.int8),
                                ply=np.asarray(a_ply[idx]).astype(np.int16), hc=hc, ha=ha, c7=c7, hid=hid.astype(np.int32),
                                tab=np.full(len(idx), t, np.int16), ph=np.full(len(idx), ph, np.int8))))
    return out

def build_features(tables):
    """Schreibt die Merkmale tischweise in vorab angelegte .npy-Memmaps (kein Concatenate-Spitzenwert im RAM)."""
    if os.path.exists(f"{WORK}/feat_{NAMES[-1]}.npy"):
        log("Merkmale vorhanden, lade"); return
    from numpy.lib.format import open_memmap
    from multiprocessing import get_context
    n_all = int(np.load(os.path.join(os.environ["POKER_CACHE"], "act_hidx.npy"), mmap_mode="r").shape[0])
    spec = dict(idx=((n_all,), np.int64), X=((n_all, 44), np.float16), cls=((n_all,), np.int8), ply=((n_all,), np.int16),
                hc=((n_all, H, 3), np.int8), ha=((n_all, H), np.float16), c7=((n_all, 7), np.int8), hid=((n_all,), np.int32),
                tab=((n_all,), np.int16), ph=((n_all,), np.int8))
    M = {nm: open_memmap(f"{WORK}/feat_{nm}.part.npy", mode="w+", dtype=dt, shape=sh) for nm, (sh, dt) in spec.items()}
    nproc = 2 if SMOKE else (4 if ON_KAGGLE else max(1, min(os.cpu_count() or 1, 8)))
    cur = 0
    with get_context("fork").Pool(nproc) as pool:
        it = pool.imap(_table_feats, tables, chunksize=1)              # geordnet -> Zeilenfolge deterministisch
        for i in range(len(tables)):
            for _t, _ph, d in it.next(timeout=1800):
                n = len(d["idx"])
                for nm in NAMES:
                    M[nm][cur:cur + n] = d[nm]
                cur += n
            if (i + 1) % 50 == 0:
                log(f"  Merkmale {i+1}/{len(tables)} Tische, {cur:,} Aktionen")
    for nm in NAMES:
        M[nm].flush()
    del M
    for nm in NAMES:
        part, final = f"{WORK}/feat_{nm}.part.npy", f"{WORK}/feat_{nm}.npy"
        if cur == n_all:
            os.rename(part, final)
        else:                                                           # SMOKE: nur ein Teil der Tische
            np.save(final, np.ascontiguousarray(np.load(part, mmap_mode="r")[:cur])); os.remove(part)
    log(f"Merkmale gespeichert: {cur:,} von {n_all:,} Aktionen")

def card_block(torch, F, c7):
    """Hole Cards (Spalten 0-1) + sichtbares Board (2-6, -1 = nicht sichtbar) -> 109 Merkmale."""
    c = c7.long(); valid = c >= 0
    cz = torch.where(valid, c, torch.zeros_like(c)); vf = valid.float().unsqueeze(-1)
    R = F.one_hot(cz // 4, 13).float() * vf
    S = F.one_hot(cz % 4, 4).float() * vf
    Rh, Rb = R[:, :2].sum(1), R[:, 2:].sum(1); Ra = Rh + Rb
    Sh, Sb = S[:, :2].sum(1), S[:, 2:].sum(1); Sa = Sh + Sb

    def windows(Rc):                                              # 5er-Fenster ueber Raenge, Ass auch unten: A-5 ... T-A
        P = (Rc > 0).float()
        return torch.cat([P[:, 12:13], P], 1).unfold(1, 5, 1).sum(-1)

    def category(Rc, Sc, W):                                      # 0 high .. 7 quads (Straight Flush zaehlt als Flush)
        srt = Rc.sort(1, descending=True).values; m1, m2 = srt[:, 0], srt[:, 1]
        npair = (Rc >= 2).float().sum(1)
        cat = torch.zeros(Rc.shape[0], dtype=torch.long, device=Rc.device)
        for cond, v in ((npair >= 1, 1), (npair >= 2, 2), (m1 >= 3, 3), (W.max(1).values >= 5, 4),
                        (Sc.max(1).values >= 5, 5), ((m1 >= 3) & (m2 >= 2), 6), (m1 >= 4, 7)):
            cat = torch.where(cond, torch.full_like(cat, v), cat)
        return cat

    Wa, Wb = windows(Ra), windows(Rb)
    ca, cb = category(Ra, Sa, Wa), category(Rb, Sb, Wb)
    ar = torch.arange(13, device=c.device, dtype=torch.float32)
    nvis = Rb.sum(1)
    topb = torch.where(nvis > 0, ((Rb > 0).float() * (ar + 1)).max(1).values - 1, torch.full_like(nvis, -1.0))
    hr = (cz[:, :2] // 4).float()
    over = (hr > topb.unsqueeze(1)).float().sum(1)
    toppair = ((hr == topb.unsqueeze(1)).float().sum(1) > 0).float()
    pocket = (Rh.max(1).values >= 2).float()
    overpair = pocket * (hr[:, 0] > topb).float()
    flushinv = (Sa * (Sh > 0).float()).max(1).values
    return torch.cat([Rh, Rb, Ra, Rh * Rb, Sh, Sb, Sa, Wa / 5, Wb / 5, F.one_hot(ca, 9).float(), F.one_hot(cb, 9).float(),
                      torch.stack([nvis / 5, topb / 12, over / 2, toppair, pocket, overpair, flushinv / 5], 1)], 1)

def selftest_cards():
    import torch, torch.nn.functional as F
    RK, SU = "23456789TJQKA", "cdhs"
    cases = [("Ah Ad", "Ac 7s 2d Kc Kh", 6, 1), ("5h 6h", "7h 8h 9c 2s 2d", 4, 1), ("Ah 2c", "3d 4s 5h Kc Qd", 4, 0),
             ("Kh Qh", "2h 7h 9h Jc 3s", 5, 0), ("7c 7d", "", 1, 0), ("9c 9d", "9h 9s 2c", 7, 1), ("Ac Kd", "2c 5h 9s", 0, 0),
             ("Ac Kd", "Ah Kc 5s", 2, 0), ("5c 5d", "5h Kc 2s", 3, 0), ("2c 3d", "9h Ts Jc Qd Kh", 4, 4)]
    rows = []
    for hole, brd, _, _ in cases:
        cc = [RK.index(x[0]) * 4 + SU.index(x[1]) for x in (hole + " " + brd).split()]
        rows.append(cc + [-1] * (7 - len(cc)))
    f = card_block(torch, F, torch.tensor(rows, dtype=torch.int8))
    assert f.shape[1] == 109, f.shape
    o = 52 + 12 + 20
    got_a, got_b = f[:, o:o + 9].argmax(1).tolist(), f[:, o + 9:o + 18].argmax(1).tolist()
    exp_a, exp_b = [x[2] for x in cases], [x[3] for x in cases]
    assert got_a == exp_a and got_b == exp_b, (got_a, exp_a, got_b, exp_b)
    log("Karten-Selbsttest bestanden (10 Haende, Kategorien Hand/Board)")

def make_model(torch, nn, F, var, mu, sd):
    cap = 2 if var.endswith("x2") else 1
    base = var[:-2] if var.endswith("x2") else var
    use_card, use_ply = base in ("full", "noply"), base in ("full", "nocard")
    W1, WC1, WC2, WG, WH = 192 * cap, 128 * cap, 64 * cap, 96 * cap, 384 * cap

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mu", mu.clone()); self.register_buffer("sd", sd.clone())
            self.xin = nn.Sequential(nn.Linear(45, W1), nn.GELU()); d = W1
            if use_ply:
                self.ply = nn.Embedding(N_PLAYERS, 32); d += 32
            if use_card:
                self.card = nn.Sequential(nn.Linear(109, WC1), nn.GELU(), nn.Linear(WC1, WC2), nn.GELU()); d += WC2
            self.e_act = nn.Embedding(9, 8); self.e_str = nn.Embedding(5, 4); self.e_rel = nn.Embedding(7, 6)
            self.hin = nn.Linear(19, 64); self.gru = nn.GRU(64, WG, batch_first=True); d += WG
            self.head = nn.Sequential(nn.Linear(d, WH), nn.GELU(), nn.Dropout(0.1), nn.Linear(WH, WH), nn.GELU(),
                                      nn.Linear(WH, 4))

        def forward(self, b):
            x = (b["X"].float() - self.mu) / self.sd
            parts = [self.xin(torch.cat([x, b["ph"].float().unsqueeze(1)], 1))]
            if use_ply:
                parts.append(self.ply(b["ply"].long()))
            if use_card:
                parts.append(self.card(card_block(torch, F, b["c7"])))
            hc = b["hc"].long() + 1                                # -1 (leer) -> 0
            h = torch.cat([self.e_act(hc[..., 0]), self.e_str(hc[..., 1]), self.e_rel(hc[..., 2]),
                           b["ha"].float().unsqueeze(-1)], -1)
            _, hn = self.gru(torch.relu(self.hin(h)))
            parts.append(hn[-1])
            return self.head(torch.cat(parts, 1))
    return PolicyNet()

def train_device(dev_name, fold_list, row_fold, row_isval, results):
    import torch, torch.nn as nn, torch.nn.functional as F
    try:
        dev = torch.device(dev_name); cuda = dev.type == "cuda"
        if cuda:
            torch.cuda.set_device(dev); torch.backends.cudnn.benchmark = True
            a = torch.randn(512, 512, device=dev); torch.nn.functional.silu(a @ a); torch.cuda.synchronize(dev)
        L = {nm: np.load(f"{WORK}/feat_{nm}.npy", mmap_mode="r") for nm in ("X", "ph", "ply", "hc", "ha", "c7", "cls")}

        def to_dev(rows):
            return {nm: torch.from_numpy(np.ascontiguousarray(L[nm][rows])).to(dev) for nm in L}

        def run_eval(model, D, probs=False):
            model.eval(); n = D["cls"].shape[0]; tot = 0.0; P = np.zeros((n, 4), np.float32) if probs else None
            with torch.no_grad():
                for b0 in range(0, n, 65536):
                    bt = {k: v[b0:b0 + 65536] for k, v in D.items()}
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cuda):
                        lg = model(bt).float()
                    tot += float(F.cross_entropy(lg, bt["cls"].long(), reduction="sum"))
                    if probs:
                        P[b0:b0 + len(lg)] = torch.softmax(lg, -1).cpu().numpy()
            model.train()
            return tot / max(n, 1), P

        for fold in fold_list:
            te = np.flatnonzero(row_fold == fold); trv = np.flatnonzero(row_fold != fold)
            va, tr = trv[row_isval[trv]], trv[~row_isval[trv]]
            np.save(f"{WORK}/te_f{fold}.npy", te)
            Dtr, Dva, Dte = to_dev(tr), to_dev(va), to_dev(te)
            s1 = torch.zeros(44, dtype=torch.float64, device=dev); s2 = torch.zeros(44, dtype=torch.float64, device=dev)
            for b0 in range(0, len(tr), 1 << 20):
                xb = Dtr["X"][b0:b0 + (1 << 20)].double(); s1 += xb.sum(0); s2 += (xb * xb).sum(0)
            mu = (s1 / len(tr)).float(); sd = (torch.sqrt(torch.clamp(s2 / len(tr) - (s1 / len(tr)) ** 2, min=0)) + 1e-3).float()
            mem = f", GPU-Speicher {torch.cuda.memory_allocated(dev)/2**30:.1f} GB" if cuda else ""
            log(f"{dev_name} Fold {fold}: Train {len(tr):,} · Val {len(va):,} · Vorhersage {len(te):,}{mem}")
            for var in VARIANTS:
                tv = time.time(); torch.manual_seed(1000 + fold)
                model = make_model(torch, nn, F, var, mu, sd).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
                spe = (len(tr) + BATCH - 1) // BATCH
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * spe, pct_start=0.15)
                scaler = torch.amp.GradScaler("cuda", enabled=cuda)
                g = torch.Generator().manual_seed(2000 + fold)
                best_v, best_state, best_ep, hist = float("inf"), None, -1, []
                for ep in range(EPOCHS):
                    model.train(); perm = torch.randperm(len(tr), generator=g).to(dev)
                    tl = torch.zeros((), dtype=torch.float64, device=dev)
                    for b in range(spe):
                        ix = perm[b * BATCH:(b + 1) * BATCH]
                        bt = {k: v[ix] for k, v in Dtr.items()}
                        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cuda):
                            lg = model(bt)
                        loss = F.cross_entropy(lg.float(), bt["cls"].long())
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward(); scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update(); sched.step()
                        tl += loss.detach().double() * len(ix)
                    vll, _ = run_eval(model, Dva)
                    hist.append([round(float(tl) / len(tr), 5), round(vll, 5)])
                    log(f"  {var:6s} Fold {fold} Epoche {ep+1:2d}/{EPOCHS}: Train {hist[-1][0]:.4f} · Val {vll:.4f} [{time.time()-tv:.0f}s]")
                    if vll < best_v:
                        best_v, best_ep = vll, ep
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_state)
                tell, P = run_eval(model, Dte, probs=True)
                np.save(f"{WORK}/P_{var}_f{fold}.npy", P)
                torch.save(best_state, f"{OUTD}/model_{var}_f{fold}.pt")
                results[(var, fold)] = dict(best_val=best_v, best_epoch=best_ep + 1, test_ll=tell, hist=hist,
                                            sec=round(time.time() - tv), device=dev_name)
                log(f"  {var:6s} Fold {fold} fertig: beste Epoche {best_ep+1}, Val {best_v:.4f}, Vorhersagehaelfte {tell:.4f}")
                del model, opt, best_state
            del Dtr, Dva, Dte
            if cuda:
                torch.cuda.empty_cache()
    except BaseException:
        results[("FEHLER", dev_name)] = traceback.format_exc()
        log(f"FEHLER auf {dev_name}:\n{results[('FEHLER', dev_name)]}")
