function [ fig ] = draw_c3iiia_rcsp( rcsp_filt, fig_name )
% Do a nice visual plot of the given RCSP filter
%
% Parameters:
%   rcsp_filter: the basis matrix to draw
%   fig_name: title for the figure
%
% Output:
%   fig: the figure in which the basis was drawn
%

% Make  a new figure to hold this drawing
fig = figure();

if ~exist('frac','var')
    frac = 1.0;
end
if ~exist('fig_name','var')
    fig_name = 'TITLE';
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
set(get(gca,'YLabel'),'Color','k')
ylabel(fig_name,'FontSize',14);


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
elec_size = scale / 1.5;

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
a_max = max(rcsp_filt);
a_min = min(rcsp_filt);
cmap = colormap(gray(128));

% Draw the electrodes colored based on the rcsp filter
for e_idx=1:60,
    e_val = rcsp_filt(e_idx);
    e_cidx = 1 + round( 127 * ((e_val - a_min) / (a_max - a_min)) );
    rectangle(...
        'Position', [elec_pos(e_idx,1) elec_pos(e_idx,2) elec_size elec_size],...
        'Curvature', [1 1], 'FaceColor',cmap(e_cidx,:),...
        'LineWidth', 1);
    %text(elec_cent(e_idx,1),elec_cent(e_idx,2),int2str(e_idx)),...
    %    'VerticalAlignment','middle','HorizontalAlignment','middle',...
    %    'FontSize',14);
end

% axis off;

drawnow;

return

end

