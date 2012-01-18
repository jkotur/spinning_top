
from OpenGL.GL import *
from OpenGL.GLU import *

import math as m
import numpy as np
import transformations as tr

from scipy.integrate import ode

from mesh import Mesh

MESH      = 1
WIREFRAME = 2
TRACE     = 4
GRAVITY   = 8

class Top( Mesh ) :
	def __init__( self ) :
		self.drawstate = MESH | GRAVITY | TRACE

		self.size = [1,1,1]
		self.dens = 10
		self.set_block( self.size , self.dens )

		self.g = np.array((0,-10,0,0) , np.float64 )
		self.G = np.resize( self.g , 3 )
		self.Q = [ 0 , 0 , 0 , 1 , 0 , 0 , 0 ]

		self.a = 0.0
		self.w = 0.0

		self.trace = []
		self.trace_len = 0

		self.reset()

	def toggle_wireframe( self ) :
		self.drawstate ^= WIREFRAME

	def toggle_solid( self ):
		self.drawstate ^= MESH
	
	def toggle_gravity( self ):
		self.drawstate ^= GRAVITY

	def set_trace_len( self , tlen ) :
		self.trace_len = tlen

	def set_dens( self , dens ) :
		self.dens = dens
		self.set_block( self.size , self.dens )
	def set_x( self , x ):
		self.size[0] = x
		self.set_block( self.size , self.dens )
	def set_y( self , y ):
		self.size[1] = y
		self.set_block( self.size , self.dens )
	def set_z( self , z ):
		self.size[2] = z
		self.set_block( self.size , self.dens )
	def set_a( self , a ):
		self.a = a * m.pi / 180.0
	def set_w( self , w ):
		self.w = w

	def set_block( self , s , d ) :
		aoy = m.atan2( s[2] , s[0] )
		aoz = m.atan2( s[1] , m.sqrt(s[0]**2+s[2]**2) )

		rot = tr.rotation_matrix( aoy , (0,1,0) )
		rot = np.dot( tr.rotation_matrix( -aoz , (0,0,1) ) , rot )
		rot = np.dot( tr.rotation_matrix( m.pi/2.0 , (0,0,1) ) , rot )

		v , n , t = self.gen_v( 1 , 1 , s )
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

	def ode( self , t , Q ) :
		w = Q[:3]
		q = Q[3:]
		q = q / np.linalg.norm( q )

		qm = tr.inverse_matrix( tr.quaternion_matrix(q) )

		if self.drawstate & GRAVITY :
			self.G = np.resize( np.dot( qm , self.g ) , 3 )
			N = np.cross( self.r , self.G )
		else :
			N = np.zeros(3)

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
		self.trace = []
		self.Q[3:] = tr.quaternion_about_axis(self.a,(0,0,1))
		self.Q[:3] = (0,self.w,0)
		self.R = ode(self.ode).set_integrator('dopri5')
		self.R.set_initial_value(self.Q,t0)

	def step( self , dt ) :
		if not self.R.successful() : return
		self.Q = self.R.integrate(self.R.t+dt)
		if len(self.trace) > self.trace_len :
			self.trace.pop(0)
		if len(self.trace) < self.trace_len+1 :
			qm = tr.quaternion_matrix( self.Q[3:] )
			self.trace.append( np.dot( qm , self.x[-1] ) )

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

	def gen_v( self , nx , ny , size = (1,1,1) ) :
		nx += 1
		ny += 1

		s = np.resize(size,4)
		s[-1] = 1

		v = np.zeros( (6,nx,ny,4)   , np.float64 )
		n = np.zeros( (6,nx,ny,3)   , np.float64 )
		t = np.zeros( (6,2,nx-1,ny-1,3) , np.uint32  )

		for x in range(nx) :
			for y in range(ny) :
				v[0,x,y] = np.array(( 0 , x/float(nx-1) , y/float(ny-1) , 1 )) * s
				v[1,x,y] = np.array(( 1 , x/float(nx-1) , y/float(ny-1) , 1 )) * s
				v[2,x,y] = np.array(( x/float(nx-1) , 1 , y/float(ny-1) , 1 )) * s
				v[3,x,y] = np.array(( x/float(nx-1) , 0 , y/float(ny-1) , 1 )) * s
				v[4,x,y] = np.array(( x/float(nx-1) , y/float(ny-1) , 0 , 1 )) * s
				v[5,x,y] = np.array(( x/float(nx-1) , y/float(ny-1) , 1 , 1 )) * s

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

		if self.drawstate & WIREFRAME :
			glPushMatrix()
			glMultTransposeMatrixf( qm )
			glDisable(GL_LIGHTING)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
			glDisable(GL_CULL_FACE)
			Mesh.draw( self )
			glBegin(GL_LINES)
			glVertex3f(0,0,0)
			glVertex3f( self.x[-1,0] , self.x[-1,1] , self.x[-1,2] )
			glEnd()
			glEnable(GL_CULL_FACE)
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
			glEnable(GL_LIGHTING)
			glPopMatrix()

		if self.drawstate & TRACE :
			glDisable(GL_LIGHTING)
			glBegin(GL_POINTS)
			for p in self.trace : glVertex3f( *p[:3] )
			glEnd()
			glEnable(GL_LIGHTING)

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

