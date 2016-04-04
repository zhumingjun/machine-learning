function bikMeans
%%
clc
clear
close all
%%
biK = 4;
biDataSet = load('testSet2.txt');
[row,col] = size(biDataSet);
% �洢���ľ���
biCentSet = zeros(biK,col);
% ��ʼ���趨cluster����Ϊ1
numCluster = 1;
%��һ�д洢ÿ���㱻��������ģ��ڶ��д洢�㵽���ĵľ���
biClusterAssume = zeros(row,2);
%��ʼ������
biCentSet(1,:) = mean(biDataSet)
for i = 1:row
 biClusterAssume(i,1) = numCluster;
 biClusterAssume(i,2) = distEclud(biDataSet(i,:),biCentSet(1,:));
end
while numCluster < biK
 minSSE = 10000;
 %Ѱ�Ҷ��ĸ�cluster���л�����ã�Ҳ����Ѱ��SSE��С���Ǹ�cluster
 for j = 1:numCluster
 curCluster = biDataSet(find(biClusterAssume(:,1) == j),:);
 [spiltCentSet,spiltClusterAssume] = kMeans(curCluster,2);
 spiltSSE = sum(spiltClusterAssume(:,2));
 noSpiltSSE = sum(biClusterAssume(find(biClusterAssume(:,1)~=j),2));
 curSSE = spiltSSE + noSpiltSSE;
 fprintf('��%d��cluster�����ֺ�����Ϊ��%f \n' , [j, curSSE])
 if (curSSE < minSSE)
 minSSE = curSSE;
 bestClusterToSpilt = j;
 bestClusterAssume = spiltClusterAssume;
 bestCentSet = spiltCentSet;
 end
 end
 bestClusterToSpilt
 bestCentSet
 %����cluster����Ŀ
 numCluster = numCluster + 1;
 bestClusterAssume(find(bestClusterAssume(:,1) == 1),1) = bestClusterToSpilt;
 bestClusterAssume(find(bestClusterAssume(:,1) == 2),1) = numCluster;
 % ���º������������
 biCentSet(bestClusterToSpilt,:) = bestCentSet(1,:);
 biCentSet(numCluster,:) = bestCentSet(2,:);
 biCentSet
 % ���±����ֵ�cluster��ÿ��������ķ����Լ����
 biClusterAssume(find(biClusterAssume(:,1) == bestClusterToSpilt),:) = bestClusterAssume;
end
figure
%scatter(dataSet(:,1),dataSet(:,2),5)
for i = 1:biK
 pointCluster = find(biClusterAssume(:,1) == i);
 scatter(biDataSet(pointCluster,1),biDataSet(pointCluster,2),5)
 hold on
end
%hold on
scatter(biCentSet(:,1),biCentSet(:,2),300,'+')
hold off
end
% ����ŷʽ����
function dist = distEclud(vecA,vecB)
 dist = sum(power((vecA-vecB),2));
end
% K-means�㷨
function [centSet,clusterAssment] = kMeans(dataSet,K)
[row,col] = size(dataSet);
% �洢���ľ���
centSet = zeros(K,col);
% �����ʼ������
for i= 1:col
 minV = min(dataSet(:,i));
 rangV = max(dataSet(:,i)) - minV;
 centSet(:,i) = repmat(minV,[K,1]) + rangV*rand(K,1);
end
% ���ڴ洢ÿ���㱻�����cluster�Լ������ĵľ���
clusterAssment = zeros(row,2);
clusterChange = true;
while clusterChange
 clusterChange = false;
 % ����ÿ����Ӧ�ñ������cluster
 for i = 1:row
 % �ⲿ�ֿ��ܿ����Ż�
 minDist = 10000;
 minIndex = 0;
 for j = 1:K
 distCal = distEclud(dataSet(i,:) , centSet(j,:));
 if (distCal < minDist)
 minDist = distCal;
 minIndex = j;
 end
 end
 if minIndex ~= clusterAssment(i,1) 
 clusterChange = true;
 end
 clusterAssment(i,1) = minIndex;
 clusterAssment(i,2) = minDist;
 end
% ����ÿ��cluster ������
 for j = 1:K
 simpleCluster = find(clusterAssment(:,1) == j);
 centSet(j,:) = mean(dataSet(simpleCluster',:));
 end
end
end