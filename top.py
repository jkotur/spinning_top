
from OpenGL.GL import *
from OpenGL.GLU import *

import math as m
import numpy as np
import transformations as tr

from scipy.integrate import ode

from mesh import Mesh

MESH     = 1
DIAGONAL = 2
TRACE    = 4
GRAVITY  = 8

class Top( Mesh ) :
	def __init__( self ) :
		self.drawstate = MESH | GRAVITY

		self.set_block( (1,1,1) , 10 )

		self.reset()

	def set_block( self , s , d ) :
		aoy = m.atan2( s[2] , s[0] )
		aoz = m.atan2( s[1] , m.sqrt(s[0]**2+s[2]**2) )

		rot = tr.rotation_matrix( aoy , (0,1,0) )
		rot = np.dot( tr.rotation_matrix( -aoz , (0,0,1) ) , rot )
		rot = np.dot( tr.rotation_matrix( m.pi/2.0 , (0,0,1) ) , rot )

		v , n , t = self.gen_v( 1 , 1 )
		for x in range(v.shape[0]) :
			for y in range(v.shape[1]) :
				for z in range(v.shape[2]) :
					v[x,y,z] = np.dot(rot,v[x,y,z])
					n[x,y,z] = np.resize(np.dot(rot,np.resize(n[x,y,z],4)),3)
		Mesh.__init__( self , buffers = (v,n,t) )

		self.x = np.array( ((0,0,0,1),(s[0],0,0,1),(0,0,s[2],1),(s[0],0,s[2],1),(0,s[1],0,1),(s[0],s[1],0,1),(0,s[1],s[2],1),(s[0],s[1],s[2],1)) , np.float64 )
		for i in range(len(self.x)) : self.x[i] = np.dot(rot,self.x[i])
		self.r = np.resize( np.dot( rot , np.array((s[0],s[1],s[2],0) , np.float64 )/2.0 ) , 3 )
		self.m = np.array( [ d*s[0]*s[1]*s[2] / 8.0 ] * len(self.x) , np.float64 )
		self.M = self.calc_m( self.x , self.m )
		self.Mi = np.linalg.inv( self.M )
		self.g = np.array((0,-10,0,0) , np.float64 )
		self.G = np.resize( self.g , 3 )
		self.Q = ( 0 , 0 , 0 , 1 , 0 , 0 , 0 )

	def ode( self , t , Q ) :
		w = Q[:3]
		q = Q[3:]
		q = q / np.linalg.norm( q )

		qm = tr.inverse_matrix( tr.quaternion_matrix(q) )
		self.G = np.resize( np.dot( qm , self.g ) , 3 )
		N = np.cross( self.r , self.G )

#        print self.G , N , np.linalg.norm(self.G) , np.linalg.norm(w)

		QP = np.empty(7,np.float64)
		QP[:3] = np.dot( self.Mi , ( N + np.cross( np.dot(self.M,w) , w ) ) )
		qw = np.empty(4,np.float64)
		qw[0]  = 0
		qw[1:] = w
		QP[3:] = tr.quaternion_multiply( q , qw ) / 2.0

		return QP

	def reset( self ) :
		t0 = 0.0
		self.R = ode(self.ode).set_integrator('dopri5')
		self.R.set_initial_value(self.Q,t0)

	def step( self , dt ) :
		if not self.R.successful() : return
		self.Q = self.R.integrate(self.R.t+dt)

	def calc_m( self , v , m ) :
		M = np.zeros((3,3))
		for i in range(len(v)) :
			M[0,0] += m[i] * ( v[i,1]**2 + v[i,2]**2 )
			M[1,1] += m[i] * ( v[i,2]**2 + v[i,0]**2 )
			M[2,2] += m[i] * ( v[i,0]**2 + v[i,1]**2 )
			M[0,1] -= m[i] * v[i,0] * v[i,1]
			M[0,2] -= m[i] * v[i,0] * v[i,2]
			M[1,2] -= m[i] * v[i,1] * v[i,2]

		M[1,0] = M[0,1]
		M[2,0] = M[0,2]
		M[2,1] = M[1,2]

		return M

	def gen_v( self , nx , ny ) :
		nx += 1
		ny += 1

		v = np.zeros( (6,nx,ny,4)   , np.float64 )
		n = np.zeros( (6,nx,ny,3)   , np.float64 )
		t = np.zeros( (6,2,nx-1,ny-1,3) , np.uint32  )

		for x in range(nx) :
			for y in range(ny) :
				v[0,x,y] = np.array(( 0 , x/float(nx-1) , y/float(ny-1) , 1 ))
				v[1,x,y] = np.array(( 1 , x/float(nx-1) , y/float(ny-1) , 1 ))
				v[2,x,y] = np.array(( x/float(nx-1) , 1 , y/float(ny-1) , 1 ))
				v[3,x,y] = np.array(( x/float(nx-1) , 0 , y/float(ny-1) , 1 ))
				v[4,x,y] = np.array(( x/float(nx-1) , y/float(ny-1) , 0 , 1 ))
				v[5,x,y] = np.array(( x/float(nx-1) , y/float(ny-1) , 1 , 1 ))

				n[0,x,y] = np.array((-1 , 0 , 0 ))
				n[1,x,y] = np.array(( 1 , 0 , 0 ))
				n[2,x,y] = np.array(( 0 , 1 , 0 ))
				n[3,x,y] = np.array(( 0 ,-1 , 0 ))
				n[4,x,y] = np.array(( 0 , 0 ,-1 ))
				n[5,x,y] = np.array(( 0 , 0 , 1 ))

		for y in range(ny-1) :
			for x in range(nx-1) :
				for i in range(0,6,2) :
					t[i,0,x,y] = np.array(( 0, 1, nx))+ x + y*nx + i*nx*ny
					t[i,1,x,y] = np.array((1,nx+1,nx))+ x + y*nx + i*nx*ny
				for i in range(1,6,2) :
					t[i,0,x,y] = np.array(( 0, nx, 1))+ x + y*nx + i*nx*ny
					t[i,1,x,y] = np.array((1,nx,nx+1))+ x + y*nx + i*nx*ny

		return v , n , t

	def draw( self ) :
		qm = tr.quaternion_matrix( self.Q[3:] )
		if self.drawstate & MESH :
			glPushMatrix()
			glMultTransposeMatrixf( qm )
			Mesh.draw( self )
			glPopMatrix()

		if self.drawstate & DIAGONAL :
			pass
#            glPushMatrix()
#            glBegin(GL_LINES)
#            glVertex3f(0,0,0)
#            glVertex3f( np.resize( np.dot(qm

		if self.drawstate & TRACE :
			pass

		if self.drawstate & GRAVITY :
			glPushMatrix()
			glDisable(GL_LIGHTING)
			glTranslatef( 2 , 2 , 0 )
			glScalef(.1,.1,.1)
			glMultTransposeMatrixf( qm )
			glColor3f(1,.5,0)
			glBegin(GL_LINES)
			glVertex3f( 0 , 0 , 0 )
			glVertex3f( *self.G )
			glEnd()
			glEnable(GL_LIGHTING)
			glPopMatrix()
