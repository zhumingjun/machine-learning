function kMeans
clc
clear
K=4;
dataSet=load('testSet.txt');
[row,col]=size(dataSet);
%�洢���ľ���
centSet=zeros(K,col);
%�����ʼ������
for i=1:col
    minV=min(dataSet(:,i));
    rangV=max(dataSet(:,i))-minV;
    centSet(:,i)=repmat(minV,[K,1])+rangV*rand(K,1);
end
%���ڴ洢ÿ���㱻�����cluster�Լ������ĵľ���
clusterAssment=zeros(row,2)
clusterChange=true;
while clusterChange
    clusterChange=false;
    %����ÿ����Ӧ�ñ������cluster
    for i=1:row
        minDist=10000;
        minIndex=0;
        for j=1:K
        distCal=distEclud(dataSet(i,:),centSet(j,:));
        if(distCal<minDist)
            minDist=distCal;
            minIndex=j;
        end
        end
    if minIndex ~= clusterAssment(i,1) 
        clusterChange=true;
    end
    clusterAssment(i,1)=minIndex;
    clusterAssment(i,2)=minDist;
    end
% ����ÿ��cluster ������
 for j = 1:K
 simpleCluster = find(clusterAssment(:,1) == j);
 centSet(j,:) = mean(dataSet(simpleCluster',:));
 end
end
figure
%scatter(dataSet(:,1),dataSet(:,2),5)
for i = 1:K
 pointCluster = find(clusterAssment(:,1) == i);
 scatter(dataSet(pointCluster,1),dataSet(pointCluster,2),5)
 hold on
end
%hold on
scatter(centSet(:,1),centSet(:,2),300,'+')
hold off
end
function dist=distEclud(vecA,vecB)
 dist=sqrt(sum(power((vecA-vecB),2)));
end

            