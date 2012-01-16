
import sys
import time

import numpy as np
import numpy.linalg as la
import transformations as tr

from OpenGL.GL import *
from OpenGL.GLU import *

import math as m

if sys.platform.startswith('win'):
    timer = time.clock
else:
    timer = time.time

from camera import Camera
from mesh import Mesh

class Scene :
	def __init__( self , fovy , ratio , near , far ) :
		self.fovy = fovy
		self.near = near 
		self.far = far
		self.ratio = ratio

		self.camera = None
		self.mesh = Mesh('plane.mesh')

		self.x = 0.0

		self.last_time = timer()

		self.plane_alpha = 65.0 / 180.0 * m.pi

		self.lpos = [ 1 ,-1 , 0 ]

	def gfx_init( self ) :
		self.camera = Camera( ( 0 , 0 , 5 ) , ( 0 , 0 , 0 ) , ( 0 , 1 , 0 ) )

		self._update_proj()

		glEnable( GL_DEPTH_TEST )
		glEnable( GL_NORMALIZE )
#        glEnable( GL_CULL_FACE )
		glEnable( GL_COLOR_MATERIAL )
		glColorMaterial( GL_FRONT , GL_AMBIENT_AND_DIFFUSE )

	def draw( self ) :
		self._update_proj()

		self.time = timer()

		dt = self.time - self.last_time

		self._step( dt )

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		self.camera.look()

		self.lpos = [ m.sin(self.x/100)*2 , -1 , m.cos(self.x/100)*2 ]

		self._set_lights()

		self._draw_scene()

		self.x+=dt*.3

		self.last_time = self.time

	def _step( self , dt ) :
		pass

	def _draw_scene( self ) :
		self.mesh.draw()

	def _update_proj( self ) :
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective( self.fovy , self.ratio , self.near , self.far )
		glMatrixMode(GL_MODELVIEW)

	def _set_lights( self ) :
		glEnable(GL_LIGHTING);
		glLightfv(GL_LIGHT0, GL_AMBIENT, [ 0.2 , 0.2 , 0.2 ] );
		glLightfv(GL_LIGHT0, GL_DIFFUSE, [ 0.9 , 0.9 , 0.9 ] );
		glLightfv(GL_LIGHT0, GL_SPECULAR,[ 0.3 , 0.3 , 0.3 ] );
		glLightfv(GL_LIGHT0, GL_POSITION, self.lpos );
		glEnable(GL_LIGHT0); 

	def set_fov( self , fov ) :
		self.fov = fov
		self._update_proj()

	def set_near( self , near ) :
		self.near = near
		self._update_proj()

	def set_ratio( self , ratio ) :
		self.ratio = ratio
		self._update_proj()

	def set_screen_size( self , w , h ) :
		self.width  = w 
		self.height = h
		self.set_ratio( float(w)/float(h) )

	def mouse_move( self , df , buts ) :
		if 3 in buts and buts[3] :
			self.camera.rot( *map( lambda x : -x*.2 , df ) )

	def key_pressed( self , mv ) :
		self.camera.move( *map( lambda x : x*.25 , mv ) )

