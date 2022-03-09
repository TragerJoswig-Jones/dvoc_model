from dataclasses import dataclass
from enum import Enum
from dvoc_model.constants import *
from numpy import sin, cos, arctan2, pi, sqrt

class RefFrames(Enum):
    ALPHA_BETA = 1
    POLAR = 2
    DQ0 = 3

@dataclass
class SinCos:
    sin_val: float
    cos_val: float

    @classmethod
    def from_theta(cls, theta: float):
        sin_val = sin(theta)
        cos_val = cos(theta)
        return SinCos(sin_val, cos_val)

    def rotate_right_120(self):
        _sin_val = -ONE_HALF * self.sin_val + SQRT_3_OVER_2 * self.cos_val
        _cos_val = -ONE_HALF * self.cos_val - SQRT_3_OVER_2 * self.sin_val
        return SinCos(_sin_val, _cos_val)

    def rotate_left_120(self):
        _sin_val = -ONE_HALF * self.sin_val - SQRT_3_OVER_2 * self.cos_val
        _cos_val = -ONE_HALF * self.cos_val + SQRT_3_OVER_2 * self.sin_val
        return SinCos(_sin_val, _cos_val)



@dataclass
class AlphaBeta:
    alpha: float
    beta: float
    gamma: float

    @classmethod
    def from_polar(cls, v: float, theta: float):
        alpha = SQRT_2 * v * cos(theta)
        beta = SQRT_2 * v * sin(theta)
        gamma = 0
        return AlphaBeta(alpha, beta, gamma)

    def to_dq0(self, sin_cos: SinCos):
        d = sin_cos.cos_val * self.alpha + sin_cos.sin_val * self.beta
        q = -sin_cos.sin_val * self.alpha + sin_cos.cos_val * self.beta
        z = self.gamma
        return Dq0(d, q, z)

    def to_polar(self):
        v = sqrt(self.alpha**2 + self.beta**2) / sqrt(2)
        theta = arctan2(self.beta, self.alpha)
        return Polar(v, theta)

    def to_abc(self):
        a = self.alpha + self.gamma
        b = -ONE_HALF * self.alpha + SQRT_3_OVER_2 * self.beta + self.gamma
        c = -ONE_HALF * self.alpha - SQRT_3_OVER_2 * self.beta + self.gamma
        return Abc(a, b, c)

    def __sub__(self, other):
        alpha = self.alpha - other.alpha
        beta = self.beta - other.beta
        gamma = self.gamma - other.gamma
        return AlphaBeta(alpha, beta, gamma)

    def __add__(self, other):
        alpha = self.alpha + other.alpha
        beta = self.beta + other.beta
        gamma = self.gamma + other.gamma
        return AlphaBeta(alpha, beta, gamma)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            alpha = self.alpha * other
            beta = self.beta * other
            gamma = self.gamma * other
            return AlphaBeta(alpha, beta, gamma)
        elif isinstance(other, AlphaBeta):
            throw(NotImplemented)

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            alpha = self.alpha / other
            beta = self.beta / other
            gamma = self.gamma / other
            return AlphaBeta(alpha, beta, gamma)
        elif isinstance(other, AlphaBeta):
            throw(NotImplemented)

@dataclass
class Dq0:
    d: float
    q: float
    z: float

    @classmethod
    def from_polar(cls, v: float, theta: float):
        d = v * cos(theta)
        q = v * sin(theta)
        z = 0
        return Dq0(d, q, z)

    def to_abc(self, sin_cos: SinCos):
        # calculate sin/cos of 120 degree shifted values
        left = sin_cos.rotate_left_120()
        right = sin_cos.rotate_right_120()

        a = sin_cos.sin_val * self.d + sin_cos.cos_val * self.q + self.z
        b = left.sin_val * self.d + left.cos_val * self.q + self.z 
        c = right.sin_val * self.d + right.cos_val * self.q + self.z

        return Abc(a, b, c)

    def to_alpha_beta(self, sin_cos: SinCos):
        alpha = sin_cos.cos_val * self.d - sin_cos.sin_val * self.q
        beta = sin_cos.sin_val * self.d + sin_cos.cos_val * self.q
        gamma = self.z 
        return AlphaBeta(alpha, beta, gamma)

@dataclass
class Abc:
    a: float
    b: float
    c: float

    @classmethod
    def from_polar(cls, v: float, theta: float):
        a = v * sin(theta)
        b = v * sin(theta - TWO_PI_OVER_3)
        c = v * sin(theta + TWO_PI_OVER_3)
        return Abc(a, b, c)

    def to_alpha_beta(self) -> AlphaBeta:
        alpha = TWO_THIRDS * self.a - ONE_THIRD * self.b - ONE_THIRD * self.c
        beta = SQRT_3_OVER_3 * (self.b - self.c)
        gamma = ONE_THIRD * (self.a + self.b + self.c)
        return AlphaBeta(alpha, beta, gamma)

    def to_dq0(self, sin_cos: SinCos):
        # calculate sin/cos of 120 degree shifted values
        left = sin_cos.rotate_left_120()
        right = sin_cos.rotate_right_120()

        d = TWO_THIRDS * (sin_cos.sin_val * self.a + left.sin_val * self.b + right.sin_val * self.c)
        q = TWO_THIRDS * (sin_cos.cos_val * self.a + left.cos_val * self.b + right.cos_val * self.c)
        z = ONE_THIRD * (self.a + self.b + self.c)

        return Dq0(d, q, z)


@dataclass
class Polar:
    r: float
    theta: float

    @classmethod
    def from_alpha_beta(cls, alpha: float, beta: float):
        r = sqrt(alpha**2 + beta**2) / sqrt(2)
        theta = arctan2(beta, alpha)
        return Polar(r, theta)

    def to_alpha_beta(self) -> AlphaBeta:
        return AlphaBeta.from_polar(mag, theta)

    def deg(self):
        return self.theta * 180 / pi


def convert_state_ref(x, fr_ref, to_ref):
    if fr_ref is to_ref:
        return x
    if fr_ref is RefFrames.POLAR:
        x_ab = AlphaBeta.from_polar(x[0], x[1])
        x_ref = (x_ab.alpha, x_ab.beta)
    elif fr_ref is RefFrames.ALPHA_BETA:
        x_ref = AlphaBeta(x[0], x[1], 0).to_polar()
    return x_ref


if __name__ == "__main__":
    sin_cos = SinCos.from_theta(0)
    abc = Abc.from_polar(480, pi/3)
    dq0 = abc.to_dq0(sin_cos)
    abc_back = dq0.to_abc(sin_cos)
    print(abc)
    print(dq0)
    print(abc_back)