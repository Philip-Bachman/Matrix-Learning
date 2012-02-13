function [ fig ] = anim_c3iiia_basis(A, B, frac, pt, start_frame, class, fig)
% Draw each network encoded by the set of basis models in A, using the sequence
% of basis weights in B, and drawn in fig

if ~exist('fig','var')
    fig = figure();
end
if ~exist('frac','var')
    frac = 0.05;
end
if ~exist('pt','var')
    pt = 0.05;
end
if ~exist('start_frame','var')
    start_frame = 1;
end
if ~exist('class','var')
    class = 0;
end

frame_count = size(B,1);
basis_count = size(A,3);
if (basis_count ~= size(B, 2))
    error('anim_c3iiia_basis; mismatched basis counts in A/B\n');
end

for f=1:frame_count,
    A_sum = zeros(size(A,1),size(A,2));
    for b=1:basis_count,
        A_sum = A_sum + squeeze(A(:,:,b)).*B(f,b);
    end
    fig = plot_basis(A_sum, frac, fig);
    figure(fig);
    text(1,1,int2str(start_frame+f-1));
    text(1,20,int2str(class));
    pause(pt);
end

return

end


function [ fig ] = plot_basis( A, frac, fig )
% Do a nice visual plot of the self-regression basis in A, showing how it
% relates to a graphical model defined over the EEG electrodes used in the
% dataset for BCI competition 3, dataset iiia.
%
% Parameters:
%   A: the basis matrix to draw
%   frac: the fraction of edges in the basis to draw (optional)
%   fig: the figure on which to draw this basis
%
% Output:
%   fig: the figure on which the basis was drawn
%


% For now, symmetrize the basis prior to drawing
A = (A + A')./2;

if ~exist('frac','var')
    frac = 1.0;
end
if ~exist('fig','var')
    fig = figure();
end

% Set the active figure and clear it
figure(fig);
cla();
hold on;
axis equal;
% Make the axes clean
set(gca,'XTick',[]);
set(gca,'YTick',[]);
% Set the figure title
set(gca,'YColor','w');
set(gca,'XColor','w');

% Define the locations of each electrode
scale = 48;
xp = [1 2 3 4 5 6 7 8 9 10 11];
yp = [1 2 3 4 5 6 7 8 9 10];
xp = xp .* scale;
yp = yp .* scale;
run_len = 0;
run_num = 1;
run_lens = [1 3 5 7 9 11 9 7 5 3];
run_xs = [6 5 4 3 2 1 2 3 4 5];
run_ys = [10 9 8 7 6 5 4 3 2 1];
elec_pos = zeros(60,2);
for e_idx=1:60,
    elec_pos(e_idx,1) = xp(run_xs(run_num) + run_len);
    elec_pos(e_idx,2) = yp(run_ys(run_num));
    run_len = run_len + 1;
    if (run_len == run_lens(run_num))
        run_len = 0;
        run_num = run_num + 1;
    end
end

% Define size of electrodes and compute electrode center points
elec_size = scale / 2;
elec_cent = elec_pos + (elec_size / 2);

% Draw a 'head'
x_max = max(elec_pos(:,1));
x_min = min(elec_pos(:,2));
y_max = max(elec_pos(:,2));
y_min = min(elec_pos(:,2));
x_span = x_max - x_min;
y_span = y_max - y_min;
hx_size = x_span + (1 * elec_size);
hy_size = y_span + (3 * elec_size);
rectangle(...
    'Position', [x_min y_min-elec_size hx_size hy_size],...
    'Curvature', [1 1], 'FaceColor',[1.0 1.0 1.0],...
    'LineWidth', 1);
% Draw a 'nose'
annotation(fig,'arrow',[0.5194 0.5194],[0.95 0.96],'HeadWidth',20);

% Set a threshold and define a colormap
thresh = quantile(abs(A(:)),(1 - frac));
a_max = max(A(:));
a_min = min(A(:));
abs_max = max(abs(A(:)));
cmap = colormap(gray(128));
% Draw the weights/edges in the basis
for e1=1:60,
    for e2=e1+1:60,
        val = A(e1,e2);
        if (abs(val) > thresh)
            x1 = elec_cent(e1,1);
            y1 = elec_cent(e1,2);
            x2 = elec_cent(e2,1);
            y2 = elec_cent(e2,2);
            %c_idx = round((((val - a_min) / (a_max - a_min)) * 127) + 1);
            c_idx = 100 - round(((abs(val)-thresh) / (abs_max-thresh)) * 99);
            c = cmap(c_idx,:);
            lw = 1 + (3 * (abs(val) / abs_max));
            if (val > 0)
                line([x1 x2], [y1 y2],'LineWidth',lw,'Color',c,'LineStyle','-');
            else
                line([x1 x2], [y1 y2],'LineWidth',lw,'Color',c,'LineStyle','--');
            end
        end
    end
end

% Draw the electrodes after the lines, so they are on top
for e_idx=1:60,
    rectangle(...
        'Position', [elec_pos(e_idx,1) elec_pos(e_idx,2) elec_size elec_size],...
        'Curvature', [1 1], 'FaceColor',[0.75 0.75 0.75],...
        'LineWidth', 1);
    %text(elec_cent(e_idx,1),elec_cent(e_idx,2),int2str(e_idx)),...
    %    'VerticalAlignment','middle','HorizontalAlignment','middle',...
    %    'FontSize',14);
end

% axis off;

drawnow;

return

end

