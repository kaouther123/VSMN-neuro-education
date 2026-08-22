// ==========================================
// VSMN - Variable Structure Model of Neurons
// ==========================================

// Fonction f(t) = exp(-t²/2)
function f(t) {
    return Math.exp(-(t * t) / 2);
}


// Équations différentielles du VSMN
function derivatives(u, v, P) {

    const du =
        -(u + P.p) / P.tau
        + (u + P.p)
        * v
        * f(P.beta)
        * f(P.lambda * (u + P.p));

    const dv =
        -P.alpha * v
        + P.k
        * Math.pow(u + P.q, P.n)
        * P.alpha
        * Math.pow(
            f(P.lambda * (u + P.p)),
            2
        );

    return { du, dv };
}


// Méthode Runge-Kutta 4
function rk4(u, v, dt, P) {

    const k1 = derivatives(u, v, P);

    const k2 = derivatives(
        u + dt * k1.du / 2,
        v + dt * k1.dv / 2,
        P
    );

    const k3 = derivatives(
        u + dt * k2.du / 2,
        v + dt * k2.dv / 2,
        P
    );

    const k4 = derivatives(
        u + dt * k3.du,
        v + dt * k3.dv,
        P
    );

    const newU =
        u + dt *
        (k1.du + 2*k2.du + 2*k3.du + k4.du) / 6;

    const newV =
        v + dt *
        (k1.dv + 2*k2.dv + 2*k3.dv + k4.dv) / 6;

    return {
        u: newU,
        v: newV
    };
}


// Simulation complète
function simulateVSMN(P) {

    let u = P.u0;
    let v = P.v0;

    const time = [];
    const U = [];
    const V = [];

    for (let i = 0; i <= P.steps; i++) {

        const t = i * P.dt;

        time.push(t);
        U.push(u);
        V.push(v);

        const result = rk4(u, v, P.dt, P);

        u = result.u;
        v = result.v;
    }

    return {
        time: time,
        U: U,
        V: V
    };
}
