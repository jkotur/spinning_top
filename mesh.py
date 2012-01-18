import sys

import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *

from drawable import Drawable

class Mesh( Drawable ) :
	def __init__( self , filename = None , buffers = None ) :
		Drawable.__init__( self )

		self.empty()

		if filename :
			self.fromFile( filename )
		elif buffers :
			self.v , self.n , self.t = buffers

	def empty( self ) :
		self.verts , self.v , self.n , self.t , self.ev , self.et , self.tn = [[]] * 7

		self.volume_size = 0
		self.volume = np.zeros( 0 , np.float32 )
		self.normal = np.zeros( 0 , np.float32 )

		self.lpos = [0]*3
		self.prog = None


	def draw( self ) :
		glEnableClientState(GL_VERTEX_ARRAY)
		if self.n != None : glEnableClientState(GL_NORMAL_ARRAY)

		glVertexPointer( self.v.shape[-1] , GL_DOUBLE , 0 , self.v )
		if self.n != None : glNormalPointer(     GL_DOUBLE , 0 , self.n )

		glDrawElements( GL_TRIANGLES , self.t.size , GL_UNSIGNED_INT , self.t )

		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_NORMAL_ARRAY)

	def gfx_init( self ) :
		pass

	def fromFile( self , path ) :
		f = open(path,'r')

		try :
			self.verts      = self._readVertices ( f )
			self.v , self.n , self.vidx = self._readPoints   ( f , self.verts )
			self.t          = self._readTriangels( f )
			self.ev,self.et = self._readEdges    ( f )
		except IOError as (errno, strerror):
			print path , " I/O error({0}): {1}".format(errno, strerror)
		except ValueError as e :
			print "Could not convert data: " , e


		self.verts = np.array( self.verts , np.float64 )
		self.v     = np.array( self.v     , np.float64 )
		if len(self.v) == len(self.n) :
			self.n = np.array( self.n     , np.float64 )
		else :
			self.n = None
		self.t     = np.array( self.t     , np.uint64  )
		self.ev    = np.array( self.ev    , np.uint64  )
		self.et    = np.array( self.et    , np.uint64  )

		self.volume.resize( len(self.t)*3*2 )
		self.normal.resize( len(self.t)*3*2 )

	def _readVertices( self , f ) :
		v = []
		for i in xrange(int(f.readline())) :
			v += map( float , f.readline().split(' ') )
		return v

	def _readPoints( self , f , verts ) :
		v , n = [] , []
		vi = []
		for i in xrange(int(f.readline())) :
			l = f.readline().split(' ')
			vi.append( int(l[0]) )
			ind = int(l[0])*3
			v += map( float , verts[ind:ind+3] )
			n += map( float , l[1:] )
		return v , n , vi

	def _readTriangels( self , f ) :
		t = []
		for i in xrange(int(f.readline())) :
			t += map( int , f.readline().split(' ') )
		return t

	def _readEdges( self , f ) :
		ev , et = [] , []
		for i in xrange(int(f.readline())) :
			l = map( int , f.readline().split(' ') )
			ev += l[:2]
			et += l[2:]
		return ev , et

