import math
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
import glm
import ctypes
import glfw

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Camera:
    def __init__(self, h, w):
        self.znear = 0.01
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = np.pi / 2
        self.position = np.array([0.0, 0.0, 3.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, -1.0, 0.0])
        self.yaw = -np.pi / 2
        self.pitch = 0
        
        self.is_pose_dirty = True
        self.is_intrin_dirty = True
        
        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True
        
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        
        self.rot_sensitivity = 0.002
        self.trans_sensitivity = 0.02
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
    
    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_project_matrix(self):
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_leftmouse_pressed:
            up, right, front = self.get_dir_vecs()
            
            yaw = -xoffset * self.rot_sensitivity
            pitch = yoffset * self.rot_sensitivity

            front = rotation_matrix(up,yaw) @ rotation_matrix(right, pitch) @ front
            self.up = rotation_matrix(up,yaw) @ rotation_matrix(right, pitch) @ self.up
            self.target[:] = self.position + front
            
            self.is_pose_dirty = True
        
        if self.is_rightmouse_pressed:
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            right = np.cross(self.up, front)
            self.position += right * xoffset * self.trans_sensitivity
            self.target += right * xoffset * self.trans_sensitivity
            cam_up = np.cross(right, front)
            self.position += cam_up * yoffset * self.trans_sensitivity
            self.target += cam_up * yoffset * self.trans_sensitivity
            
            self.is_pose_dirty = True
        
    def process_wheel(self, dx, dy):
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        self.position += front * dy * self.zoom_sensitivity
        self.target += front * dy * self.zoom_sensitivity
        self.is_pose_dirty = True
        
    def process_roll_key(self, d):
        front = self.target - self.position
        right = np.cross(front, self.up)
        new_up = self.up + right * (d * self.roll_sensitivity / np.linalg.norm(right))
        self.up = new_up / np.linalg.norm(new_up)
        self.is_pose_dirty = True
        
    def get_dir_vecs(self):
        front = self.target - self.position
        right = np.cross(front, self.up)
        self.up /= np.linalg.norm(self.up)
        right /= np.linalg.norm(right)
        front /= np.linalg.norm(front)
        return self.up,right,front
    
    def process_trans_keys(self, current_keys, delta_t):
        key_trans_sens = 6 * delta_t
        
        up,right,front = self.get_dir_vecs()
        
        if glfw.KEY_W in current_keys:
            self.position += front * key_trans_sens
        if glfw.KEY_A in current_keys:
            self.position -= right * key_trans_sens
        if glfw.KEY_S in current_keys:
            self.position -= front * key_trans_sens
        if glfw.KEY_D in current_keys:
            self.position += right * key_trans_sens
        if glfw.KEY_LEFT_SHIFT in current_keys:
            self.position -= up * key_trans_sens
        if glfw.KEY_SPACE in current_keys:
            self.position += up * key_trans_sens
        self.target = self.position + front
        
        self.is_pose_dirty = True
    
        
    
    def update_resolution(self, height, width):
        self.h = height
        self.w = width
        self.is_intrin_dirty = True
        
    def set_camera_view(self, camera):
        #front = self.target - self.position
        self.position = np.array(camera["position"])
        front = np.array(camera["rotation"]) @ np.array([0.0,0.0,1])
        self.up = np.array(camera["rotation"]) @ np.array([0.0,-1,0])
        self.target = front + self.position
        self.fovy = np.abs(focal2fov(camera["fy"],camera["height"]))
        self.is_intrin_dirty = True


def load_shaders(vs, fs):
    vertex_shader = open(vs, 'r').read()        
    fragment_shader = open(fs, 'r').read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader

def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)
    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = glGenBuffers(1)
            buffer_ids[i] = b
        glBindBuffer(GL_ARRAY_BUFFER, b)
        glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
        length = value.shape[-1]
        pos = glGetAttribLocation(program, key)
        glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)
    
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_ids

def set_attribute(program, key, value, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_attribute_instanced(program, key, value, instance_stride=1, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glVertexAttribDivisor(pos, instance_stride)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_storage_buffer_data(program, key, value: np.ndarray, bind_idx, vao=None, buffer_id=None):
    glUseProgram(program)
    # if vao is None:  # TODO: if this is really unnecessary?
    #     vao = glGenVertexArrays(1)
    if vao is not None:
        glBindVertexArray(vao)
    
    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    # pos = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, key)  # TODO: ???
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    # glShaderStorageBlockBinding(program, pos, pos)  # TODO: ???
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

def set_faces_tovao(vao, faces: np.ndarray):
    # faces
    glBindVertexArray(vao)
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    return element_buffer

def set_gl_bindings(vertices, faces):
    # vertices
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    # vertex_buffer = glGenVertexArrays(1)
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(0)

    # faces
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    # glVertexAttribPointer(1, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(2, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(2)

def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name), 
        1,
        GL_FALSE,
        content.astype(np.float32)
    )

def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(
        glGetUniformLocation(shader, name), 
        content,
    )

def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(
        glGetUniformLocation(shader, name), 
        content
    )

def set_uniform_v3f(shader, contents, name):
    glUseProgram(shader)
    glUniform3fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )

def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )

def set_uniform_v1f(shader, contents, name):
    glUseProgram(shader)
    glUniform1fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )
    
def set_uniform_v2(shader, contents, name):
    glUseProgram(shader)
    glUniform2f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1]
    )

def set_texture2d(img, texid=None):
    h, w, c = img.shape
    assert img.dtype == np.uint8
    if texid is None:
        texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,   
        GL_RGB, GL_UNSIGNED_BYTE, img
    )
    glActiveTexture(GL_TEXTURE0)  # can be removed
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    return texid

def update_texture2d(img, texid, offset):
    x1, y1 = offset
    h, w = img.shape[:2]
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, x1, y1, w, h,
        GL_RGB, GL_UNSIGNED_BYTE, img
    )


