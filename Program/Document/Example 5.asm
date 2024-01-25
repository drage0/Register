6
4 4 2 2 2 1
MOV         4,   x0
UMUL        1,   x0, x0  
MOVSS       x1,  x2
MOVSS       x2,  x3 
IMUL        3,   x4, x0
MOV         2,   x5
SUB         x8,  x5, x8
ADD         x5,  x0, x5
MOV         x0,  x0     ; MOV 0, x0
IMUL        x0,  32, x0
MOV         x7,  x5
ADD         x6,  x0, x0
MOV         32,  x9
LEA         x10, x11
MOV         x0,  x5

; MOV         4, EAX
; UMUL        1, RAX, RAX  
; MOVSS       dword ptr [rsp+94h], xmm0
; MOVSS       xmm0, dword ptr [rsp+rax+160h] 
; IMUL        eax, dword ptr [i], eax
; MOV         2, ecx
; SUB         dword ptr [j], ecx  
; ADD         ecx, eax
; MOV         eax, eax                        ; MOV 0, EAX
; IMUL        rax 32, rax  
; MOV         qword ptr [model], rcx  
; ADD         qword ptr [rcx+8], rax  
; MOV         32, r8d
; LEA         [rsp+148h], rdx  
; MOV         rax, rcx
; 
; 00007FF62146A42F  mov         eax,4
; 00007FF62146A434  imul        rax,rax,1
; 00007FF62146A438  movss       xmm0,dword ptr [rsp+94h]
; 00007FF62146A441  movss       dword ptr [rsp+rax+160h],xmm0
;			memcpy(&(((struct r_vertex_prop *)model->data)[i*3+(2-j)]), &outgoing, sizeof(struct r_vertex_prop));
; 00007FF62146A44A  imul        eax,dword ptr [i],3
; 00007FF62146A452  mov         ecx,2
; 00007FF62146A457  sub         ecx,dword ptr [j]
; 00007FF62146A45E  add         eax,ecx
; 00007FF62146A460  mov         eax,eax
; 00007FF62146A462  imul        rax,rax,20h
; 00007FF62146A466  mov         rcx,qword ptr [model]
; 00007FF62146A46E  add         rax,qword ptr [rcx+8]
; 00007FF62146A472  mov         r8d,20h
; 00007FF62146A478  lea         rdx,[rsp+148h]
; 00007FF62146A480  mov         rcx,rax
;
