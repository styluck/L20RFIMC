function retVal=PerformanceMeasure(output,groundtruth,test_index)

%output: n*d 
%groundtruth: n*d
%test_index:1*test_num

% output = sign(output);
% output=(output+1)/2;
% groundtruth=(groundtruth+1)/2;


retVal=zeros(1,2);
%map for the whole matrix
retVal(1,1)=AveragePrecision(output(test_index,:),groundtruth(test_index,:));

%map for the test data only
retVal(1,2)=AveragePrecision(output,groundtruth);