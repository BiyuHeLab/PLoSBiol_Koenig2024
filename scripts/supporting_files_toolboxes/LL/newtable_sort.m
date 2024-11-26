%This function is similar to part of Qi's eprime_sorting, with a slight
%correction in the indexing
function interesting_trials = newtable_sort(newtable,categorysequence)

    p_no=length(categorysequence);
    v2s = newtable;
    
    %here is the main correction: previously no absolute indices were
    %provided, thus updating any index was only relative
    interesting_trials = [1:size(v2s,1)]'; 

    for i=1:p_no;
        funcname=categorysequence{i};
        [tableupdate,indupdate]=eval([funcname,'(','v2s',')']);  %this applies different functions to the output
        clear v2s
        v2s=tableupdate;
        interesting_trials = interesting_trials(indupdate);
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% subroutines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [table,ind]=keeptrial(v2s)
    w=cellfun(@length,v2s(:,1)); % finds NANs (catchtrials)
    ind=find(w>2);
    table=v2s(ind,:); %generate a new table with the same structure 
   
end 

function [table,ind]=keepcatch(v2s)
    w=cellfun(@length,v2s(:,1));
    ind=find(w<2);%all stimulus on index selected, catch trials not selected
    table=v2s(ind,:);
 
end 

function [table,ind]=keepyes(v2s)
   w=cellfun(@(x) find(x==1),v2s(:,2),'UniformOutput',false); 
   ind=find(~cellfun(@isempty,w)); %return the index
   table=v2s(ind,:);   
   
end
   
function [table,ind]=keepno(v2s)
   w=cellfun(@(x)find(x==2),v2s(:,2),'UniformOutput',false); 
   ind=find(~cellfun(@isempty,w)); %return the index
   table=v2s(ind,:);
end

function [table,ind]=keepcorrect(v2s)
    w=cellfun(@(x) find(x==1),v2s(:,6),'UniformOutput',false); 
   ind=find(~cellfun(@isempty,w)); %return the index
   table=v2s(ind,:);
end

function [table,ind]=keepwrong(v2s)
    w=cellfun(@(x) find(x==2),v2s(:,6),'UniformOutput',false); 
   ind=find(~cellfun(@isempty,w)); %return the index
   table=v2s(ind,:);
end

function [table,ind]=keep45(v2s)
    w=cellfun(@length,v2s(:,1)); % finds 45 deg presentations
    ind=find(w==32);
    table=v2s(ind,:); %generate a new table with the same structure 
   
end 

function [table,ind]=keep135(v2s)
    w=cellfun(@length,v2s(:,1)); % finds 45 deg presentations
    ind=find(w==33);
    table=v2s(ind,:); %generate a new table with the same structure 
   
end 