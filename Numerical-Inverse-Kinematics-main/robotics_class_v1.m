%%
clear;close all;clc
%%
dt=1/1000;
t=0:dt:12+dt;
scale = 2./(3 - cos(2*t));
x = scale.*cos(t);
y = scale.*sin(2*t)./ 2;
x=x*3;
y=y*3;
dx=diff(x)./dt;
dy=diff(y)./dt;
v=[dx;dy];
KX=[1 0;0 0];
%%
ath=0;
K=[cosd(ath) 0;sind(ath) 0];
kxx=K(1,1);
kxy=K(1,2);
kyx=K(2,1);
kyy=K(2,2);
q=[0;0;0];
for k=1:1:length(x)-1
    th1=q(1,k);th2=q(2,k);th3=q(3,k);
    J = jaco_3(th1,th2,th3);
    q(:,k+1)=q(:,k) + 1*pinv(J)*(v(:,k))*dt ;
end
% animate_3r(q,K,x,y)
%%
q=[pi/32;0;0];
for k=1:1:length(x)-1
    th1=q(1,k);th2=q(2,k);th3=q(3,k);
    J = jaco_3(th1,th2,th3);
    q(:,k+1)=q(:,k) + 1*pinv(J)*(v(:,k))*dt ;
end
animate_3r(q,K,x,y)

%%
q=[pi/32;1;0];
for k=1:1:length(x)-1
    th1=q(1,k);th2=q(2,k);th3=q(3,k);
    J = jaco_3(th1,th2,th3);
    [xo,yo] = fwd_kin3(q(:,k));
    E=[x(k);y(k)]-[xo;yo];
    q(:,k+1)=q(:,k) + 1*pinv(J)*(v(:,k) + 1E2*E)*dt ;
end
animate_3r(q,K,x,y)
%%
function [Xo,Yo,Xof,Yof]=ellipsoid(J,x0,y0)
[evt,evl]=eig(J*J');
if(evl(2,2)>evl(1,1))
    a=sqrt(evl(2,2));
    b=sqrt(evl(1,1));
    ax=1;
    ay=0;
    bx=evt(1,2);
    by=evt(2,2);
    th=atan2( ax*by-ay*bx, ax*bx+ay*by );
end
if(evl(2,2)<evl(1,1))
    a=sqrt(evl(1,1));
    b=sqrt(evl(2,2));
    ax=1;
    ay=0;
    bx=evt(1,2);
    by=evt(2,2);
    th=atan2( ax*by-ay*bx, ax*bx+ay*by );
end
thf= th + pi/2;
te=-pi:0.01:pi;
x56=a*cos(te);
y56=b*sin(te);
R5=[cos(th) -sin(th) ; sin(th) cos(th)];
P5= R5*[x56;y56];
Xo= P5(1,:)+x0; 
Yo= P5(2,:)+y0; 
R5=[ cos(thf) -sin(thf) ; sin(thf) cos(thf)];
P5= R5*[x56;y56];
Xof= P5(1,:)+x0; 
Yof= P5(2,:)+y0; 
end
function J = jaco_2(th1,th2)
l1=1;l2=1;
J=[-l1*sin(th1)-l2*sin(th2+th1),-l2*sin(th1+th2);l1*cos(th1)+l2*cos(th1+th2),l2*cos(th1+th2)];
end
function J = jaco_3(th1,th2,th3)
l1=1;l2=1;l3=1;
J=[-l1*sin(th1)-l2*sin(th2+th1)-l3*sin(th3+th2+th1),-l2*sin(th2+th1)-l3*sin(th3+th2+th1),-l3*sin(th1+th2+th3);l1*cos(th1)+l2*cos(th2+th1)+l3*cos(th3+th2+th1),l2*cos(th2+th1)+l3*cos(th3+th2+th1),l3*cos(th1+th2+th3)];
end
function [x,y] = fwd_kin(q)
l1=1;l2=1;
x=l1*cos(q(1,:)) + l2*cos(q(1,:)+q(2,:));
y=l1*sin(q(1,:)) + l2*sin(q(1,:)+q(2,:));
end
function [x,y] = fwd_kin3(q)
l1=1;l2=1;l3=1;
x=l1*cos(q(1,:)) + l2*cos(q(1,:)+q(2,:)) + l3*cos(q(1,:)+q(2,:)+q(3,:));
y=l1*sin(q(1,:)) + l2*sin(q(1,:)+q(2,:)) + l3*sin(q(1,:)+q(2,:)+q(3,:));
end
function NC = null_contribution(th1,th2,th3,kxx,kxy,kyx,kyy)
l1=1;l2=1;l3=1;
J=[-l1*sin(th1)-l2*sin(th2+th1)-l3*sin(th3+th2+th1),-l2*sin(th2+th1)-l3*sin(th3+th2+th1),-l3*sin(th1+th2+th3);l1*cos(th1)+l2*cos(th2+th1)+l3*cos(th3+th2+th1),l2*cos(th2+th1)+l3*cos(th3+th2+th1),l3*cos(th1+th2+th3)];
qof = qo(th1,th2,th3,kxx,kxy,kyx,kyy,l1,l2,l3);
NC=(eye(3) - pinv(J)*J)*qof ;
end
function qof = qo(th1,th2,th3,kxx,kxy,kyx,kyy,l1,l2,l3)
t2 = cos(th1);
t3 = sin(th1);
t4 = kxx+kyy;
t5 = th1+th2;
t6 = l3.^2;
t7 = l1.*t2;
t8 = cos(t5);
t9 = l1.*t3;
t10 = sin(t5);
t11 = t5+th3;
t16 = 1.0./t4;
t12 = cos(t11);
t13 = sin(t11);
t14 = l2.*t8;
t15 = l2.*t10;
t21 = kxx.*t16;
t22 = kxy.*t16;
t23 = kyx.*t16;
t24 = kyy.*t16;
t17 = t12.^2;
t18 = t13.^2;
t19 = l3.*t12;
t20 = l3.*t13;
t28 = t6.*t12.*t13;
t25 = t6.*t17;
t26 = t6.*t18;
t29 = t14+t19;
t30 = t15+t20;
t31 = t28.*2.0;
t27 = -t26;
t32 = t29.^2;
t33 = t30.^2;
t34 = t7+t29;
t35 = t9+t30;
t39 = t19.*t29;
t40 = t20.*t30;
t41 = t20.*t29.*2.0;
t42 = t19.*t30.*2.0;
t52 = t29.*t30;
t36 = -t33;
t37 = t34.^2;
t38 = t35.^2;
t44 = -t42;
t45 = -t40;
t46 = t19.*t34;
t47 = t20.*t35;
t48 = t20.*t34.*2.0;
t49 = t19.*t35.*2.0;
t53 = t52.*2.0;
t54 = t29.*t34;
t55 = t30.*t35;
t56 = t30.*t34.*2.0;
t57 = t29.*t35.*2.0;
t60 = t34.*t35;
t43 = -t38;
t50 = -t49;
t51 = -t47;
t58 = -t57;
t59 = -t55;
t61 = t60.*2.0;
t62 = t26+t33+t38;
t63 = t25+t32+t37;
t65 = t28+t52+t60;
t64 = t56+t58;
t66 = t31+t53+t61;
t67 = t62+t63;
t70 = t41+t44+t48+t50;
t71 = t27+t36+t43+t63;
t72 = t25+t27+t39+t45+t46+t51;
t73 = t25+t27+t32+t36+t54+t59;
t68 = 1.0./t67;
t69 = t68.^2;
t74 = t63.*t68;
t75 = t62.*t68;
t82 = t65.*t68;
t87 = t68.*t72;
t88 = t68.*t73;
t76 = -t74;
t77 = -t75;
t83 = t22+t82;
t84 = t23+t82;
t89 = t64.*t65.*t69;
t90 = t65.*t69.*t70;
t78 = t24+t76;
t79 = t21+t77;
t85 = t83.^2;
t86 = t84.^2;
t91 = t88+t89;
t92 = t87+t90;
t80 = t79.^2;
t81 = t78.^2;
t93 = t80+t81+t85+t86;
t94 = 1.0./sqrt(t93);
qof = [(t94.*(t66.*t68.*t78.*2.0-t66.*t68.*t79.*2.0+t68.*t71.*t83.*2.0+t68.*t71.*t84.*2.0))./2.0;(t94.*(t83.*t91.*2.0+t84.*t91.*2.0+t78.*(t68.*(t31+t53+t56)-t63.*t64.*t69).*2.0-t79.*(t68.*(t31+t53+t57)+t62.*t64.*t69).*2.0))./2.0;(t94.*(t83.*t92.*2.0+t84.*t92.*2.0+t78.*(t68.*(t31+t41+t48)-t63.*t69.*t70).*2.0-t79.*(t68.*(t31+t42+t49)+t62.*t69.*t70).*2.0))./2.0];
end
function []=animate_3r(q,K,xt,yt)
xi=q;
figure('WindowState','maximized')
%pause(7)
c=1;
for i=1:15:length(xi)
    theta1=xi(1,i);
    theta2=xi(2,i);
    theta3=xi(3,i);
    l1=1; %Input the l length
    l2=1; %Input the l length
    l3=1;
    % Homogeneus transformation matrix
    H01 = [cos(theta1) -sin(theta1) 0 l1*cos(theta1);sin(theta1) cos(theta1) 0 l1*sin(theta1);0 0 1 0;0 0 0 1]; %Frame 0 to 1 tranformation
    H12 = [cos(theta2) -sin(theta2) 0 l2*cos(theta2);sin(theta2) cos(theta2) 0 l2*sin(theta2);0 0 1 0;0 0 0 1]; %Frame 1 to 2 tranformation
    H23 = [cos(theta3) -sin(theta3) 0 l3*cos(theta3);sin(theta3) cos(theta3) 0 l3*sin(theta3);0 0 1 0;0 0 0 1]; %Frame 1 to 2 tranformation
    H02=H01*H12;      %Frame 0 to 2 tranformation
    H03=H01*H12*H23;

    P1=[H01(1,4) H01(2,4)];
    P2=[H02(1,4) H02(2,4)];
    P3=[H03(1,4) H03(2,4)];
    P3x(c,:)=P3(1,1);
    P3y(c,:)=P3(1,2);
    plot(xt,yt,'--y')
    hold on
    plot(P1(1),P1(2),'ok','LineWidth',1)

    plot(P3x,P3y,'k')
    plot(P2(1),P2(2),'ok','LineWidth',1)
    plot(P3(1),P3(2),'ok','LineWidth',1)

    J=[-l1*sin(theta1)-l2*sin(theta2+theta1)-l3*sin(theta3+theta2+theta1),-l2*sin(theta2+theta1)-l3*sin(theta3+theta2+theta1),-l3*sin(theta1+theta2+theta3);l1*cos(theta1)+l2*cos(theta2+theta1)+l3*cos(theta3+theta2+theta1),l2*cos(theta2+theta1)+l3*cos(theta3+theta2+theta1),l3*cos(theta1+theta2+theta3)];
    [~,~,Xof,Yof]=ellipsoid(J/5,P3(1),P3(2));
    [Xofk,Yofk,~,~]=ellipsoid(K/2,P3(1),P3(2));
    plot(0,0,'ok','LineWidth',3)
%     plot(Xof,Yof,'r')
%     plot(Xofk,Yofk,'m')
    xlim([-3.5 3.5])
    ylim([-3.5 3.5])
    axis square; 
    grid minor
    plot([0 P1(1)], [0 P1(2)],'g','LineWidth',2)
    plot([P1(1) P2(1)], [P1(2) P2(2)],'b','LineWidth',2)
    plot([P2(1) P3(1)], [P2(2) P3(2)],'g','LineWidth',2)
    xlabel('X axis (m)','Interpreter','latex')
    ylabel('Y axis (m)','Interpreter','latex')
    set(gca,'FontSize',18)
    drawnow
    hold off
    c=c+1;
end
end