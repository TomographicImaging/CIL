#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:13:20 2018

@author: evangelos
"""

    Mx=spdiags([ones(N,1) -ones(N,1)],[0 1],N,N);
    Mx(N,:)=0;
    
    My = spdiags([-ones(M,1) ones(M,1)],[0 -1],M,M);
    My(1,:)=0; 
    
    DXY=kron(My,Mx);
    DYX=kron(Mx,My);
    div_XY=DXY';
    div_YX=DYX';