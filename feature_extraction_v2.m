
subject_signal_list = dir('incart_mat_segment');
subject_Rpeak_list = dir('incart_mat_Rpeak');
%fs = 360;
fs = 257;
%Rpeak = 101;
Rpeak = 77;
sig_len = 180;
mkdir('./incart_mat_TDfeature');
for t=3:100
    subject_name = subject_signal_list(35).name;
    load(strcat('./incart_mat_segment/',subject_name),'x');
    sig = x;
    subject_rpeak_name = subject_Rpeak_list(35).name;
    
    qrs_raw = load(strcat('./incart_mat_Rpeak/',subject_rpeak_name),'p');
    qrs_raw = cell2mat(struct2cell(qrs_raw));
    q_loc=[];
    s_loc=[];
    RR_int=diff(qrs_raw);
    RR=mean(RR_int);
    RR_mean = RR/fs;
    num = [ Rpeak - fix(RR/6)-10 : Rpeak-fix(RR/10) ];
    num2 = [ Rpeak + fix(RR/10) : sig_len];
    for i=1:size(sig,1) % q피크 s피크 찾기
            [~,q_loc_temp]=min(sig(i,Rpeak - round(0.1*fs):Rpeak));
            q_loc=[q_loc, q_loc_temp + Rpeak - round(0.1*fs)-1];
            [~,s_loc_temp]=min(sig(i,Rpeak : Rpeak + round(0.075*fs)));
            a = s_loc_temp+Rpeak-1;
            s_loc=[s_loc, a];
    end
    clear Y;
    clear X;    
    for i=1:size(sig,1)% p피크 찾기
        [Y(i),X(i)] = max(sig(i,num));
        num_p(i)=num(X(i));
    end  %%find p pe ak  % 전체 p peak  
    % hold on;scatter(num_p/fs,Y); 

    clear Y;
    clear X;

    for i=1:size(sig,1)
        [Y(i),X(i)] = max(sig(i,num2));
        num_t(i)=num2(X(i));
    end  %%find t peak
        %hold on;scatter(num_t/fs,Y); 
    clear Y;
    clear X;
    amp_feature = [];
    for i=1:size(sig,1)
        q_peak = q_loc(i);
        s_peak = s_loc(i);
        p_peak = num_p(i);
        t_peak = num_t(i);
% 
%         d_pq = (p_peak - q_peak).^2 + (sig(i,p_peak) - sig(i,q_peak)).^2;
%         d_qr = (q_peak - Rpeak).^2 + (sig(i,q_peak) - sig(i,Rpeak)).^2; 
%         d_pr = (p_peak - Rpeak).^2 + (sig(i,p_peak) - sig(i,Rpeak)).^2;
%         d_rs = (Rpeak - s_peak).^2 + (sig(i,Rpeak) - sig(i,s_peak)).^2;
%         d_qs = (q_peak - s_peak).^2 + (sig(i,q_peak) - sig(i,s_peak)).^2;
%         d_st = (s_peak - t_peak).^2 + (sig(i,s_peak) - sig(i,t_peak)).^2;
%         d_rt = (Rpeak - t_peak).^2 + (sig(i,Rpeak) - sig(i,t_peak)).^2;
% 
%         cos_q = (d_pq + d_qr - d_pr) / (2*sqrt(d_pq)*sqrt(d_qr));
% 
%         angle_q = acos(cos_q);
% 
%         cos_r = (d_qr + d_rs - d_qs) / (2*sqrt(d_qr)*sqrt(d_rs));
%         angle_r = acos(cos_r);
% 
%         cos_s = (d_rs + d_st - d_rt) / (2*sqrt(d_rs)*sqrt(d_st));
%         angle_s = acos(cos_s);

        PR_amp = sig(i,Rpeak)-sig(i,p_peak);
        QR_amp = sig(i,Rpeak)-sig(i,q_peak);
        RS_amp = sig(i,Rpeak)-sig(i,s_peak);
        RT_amp = sig(i,Rpeak)-sig(i,t_peak);
        PR_inter = abs(p_peak-Rpeak)/fs;
        QR_inter = (abs(Rpeak-q_peak))/fs;
        RS_inter = (abs(Rpeak-s_peak))/fs;
        RT_inter = abs(t_peak-Rpeak)/fs;
        ST_inter = abs(t_peak-s_peak)/fs;
        
        Pre_RR = double(RR_int(i))/fs;
        Post_RR = double(RR_int(i+1))/fs;
        R_amp = sig(i,Rpeak);
        PQ_inter = abs(p_peak - q_peak)/fs;
        PT_inter = abs(p_peak - t_peak)/fs;
        QT_inter = abs(q_peak - t_peak)/fs;
        QS_inter = abs(s_peak - q_peak)/fs;
        
        variance = var(sig(i,:));
        rms_ = rms(sig(i,:));
        Median_RR = 0.0 ;
        if (i<15)
            Median_RR = double(median(RR_int(1:i+15)))/fs;
        end
        if (i>15&& i<size(sig,1)-15)
            Median_RR = double(median(RR_int(i-15:i+15)))/fs;
        end
        if (i>size(sig,1)-15)
            Median_RR = double(median(RR_int(i-15:size(sig,1))))/fs;
        end
        
        f_max = max(sig(i,:));
        f_min = min(sig(i,:));
%         kurtosis_ = kurtosis(sig(i,:));
%         skewness_ = skewness(sig(i,:));
%         if(isnan(angle_q))
%             angle_q = 0;
%         end
%         if(isnan(angle_r))
%             angle_r = 0;
%         end
%         if(isnan(angle_s))
%             angle_s = 0;
%         end
        if (isnan(variance))
            variance = 0;
        end
        if (isnan(rms_))
            rms_ = 0;
        end        
%         if (isnan(kurtosis_))
%             kurtosis_ = 0;
%         end
%         if (isnan(skewness_))
%             skewness_ = 0;
%         end
        
        amp_feature = [amp_feature; R_amp PR_amp QR_amp RS_amp RT_amp PR_inter QR_inter RS_inter RT_inter ST_inter PQ_inter PT_inter QT_inter Pre_RR Post_RR RR_mean Median_RR f_max f_min variance rms_;];
        
    end
    
%     sig = sig.';
%     spectrum = fft(sig);
%     P2 = abs(spectrum/sig_len);
%     P1 = P2(1:sig_len/2+1,:);
%     P1(2:end-1) = 2*P1(2:end-1);
%     x2 = P1(2:36,:);
%     x2 = x2.';
    x = amp_feature;
    save(strcat('./incart_mat_TDfeature/',subject_name), 'x' );
    fprintf(strcat(subject_name,' end\n'));
end
