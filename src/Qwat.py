import math
 
class Vector3f:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0):
        self.x = x
        self.y = y
        self.z = z
 
    def __str__(self):
     return "Vector = " + str(self.x) + " " + str(self.y) + " " + str(self.z);
 
class Quaternion:
    def __init__(self, w = 1.0, x = 0.0, y = 0.0, z = 0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
 
    def __str__(self):
     return "Quaternion = " + str(self.w) + " " + str(self.x) + " " + str(self.y) + " " + str(self.z);
 
 
def normal(v):
    normal = Vector3f()
    length = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5    
    normal.x = v.x / length
    normal.y = v.y / length
    normal.z = v.z / length
    return normal
 
def create_quat(rotate_vector, rotate_angle):
    quat = Quaternion()
    rotate_vector = normal(rotate_vector)
    quat.w = math.cos(rotate_angle / 2)
    quat.x = rotate_vector.x * math.sin(rotate_angle / 2)
    quat.y = rotate_vector.y * math.sin(rotate_angle / 2)
    quat.z = rotate_vector.z * math.sin(rotate_angle / 2)
    return quat
 
def quat_scale(q, val):
    q.w = q.w * val
    q.x = q.x * val
    q.y = q.y * val
    q.z = q.z * val
    return q
 
def quat_length(q):
    quat_length = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) ** 0.5
    return quat_length
 
def quat_normalize(q):
    n = quat_length(q)
    return quat_scale(q, 1 / n)
 
def quat_invert(q):
    res = Quaternion()
    res.w = q.w
    res.x = -q.x
    res.y = -q.y
    res.z = -q.z
    return quat_normalize(res)
 
def quat_mul_quat(a, b):
    res = Quaternion()
    res.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    res.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    res.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    res.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    return res
 
def quat_mul_vector(a, b):
    res = Quaternion()
    res.w = -a.x * b.x - a.y * b.y - a.z * b.z
    res.x = a.w * b.x + a.y * b.z - a.z * b.y
    res.y = a.w * b.y - a.x * b.z + a.z * b.x
    res.z = a.w * b.z + a.x * b.y - a.y * b.x
    return res
 
def quat_transform_vector(q, v):
    t = Quaternion()
    t = quat_mul_vector(q, v)
    t = quat_mul_quat(t, quat_invert(q))
    ret = Vector3f()
    ret.x = t.x
    ret.y = t.y
    ret.z = t.z
    return ret
 
qua = create_quat(Vector3f(0,0,1), math.pi/2)
print(qua)
vec = Vector3f(0,0,1)
print(quat_transform_vector(qua, vec))