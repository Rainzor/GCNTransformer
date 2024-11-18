clear
% 读取 poisson_disk_sample3.txt 文件
points = [];
fid = fopen('../poisson_disk_sample3.txt', 'r');
tline = fgetl(fid);
while ischar(tline)
    values = str2num(tline);
    points = [points; values]; 
    tline = fgetl(fid);
end
fclose(fid);
disp(size(points));

% 读取 conflict3.txt 文件
num = [];
data = [];
fid = fopen('../conflict3.txt', 'r');
tline = fgetl(fid);
while ischar(tline)
    values = str2num(tline); %#ok<ST2NM>
    if length(values) == 1
        num = [num; values + 1]; %#ok<AGROW> % 索引从 1 开始
    else
        data = [data; values]; %#ok<AGROW>
    end
    tline = fgetl(fid);
end
fclose(fid);
disp(size(data));


fid = fopen('../surface_sample3.txt', 'r');
samples_data = textscan(fid, '%f', 'Delimiter', '\n');
fclose(fid);
samples_data = samples_data{1};
samples = {};
i = 1;
idx = 1;
while(i<=length(samples_data))
    num_s = samples_data(i);
    samples{idx} = reshape(samples_data(i+1:i+num_s*3),3,[])';
    idx = idx + 1;
    i = i + 1 + num_s*3;
end


hulls = {};
centers = {};
bounary_vertices = {};
faces_vertices = {};
idx = 1;
for i = 1:length(num)
    center = data(idx, :); 
    centers{i} = center; %#ok<SAGROW> % 存储中心
    vertices = data(idx+1:idx+num(i)-1, :);
    faces_vertices{i} = vertices;
    vertices = unique(vertices, 'rows'); % 去除重复的顶点
    bounary_vertices{i} = vertices;

    vertices_pos = vertices(:, 1:3);
    K = convhull(vertices_pos(:, 1), vertices_pos(:, 2), vertices_pos(:, 3)); % 计算凸包
    hulls{i} = K; %#ok<SAGROW> % 存储凸包
    idx = idx + num(i);
end
centers = cell2mat(centers');
disp(size(centers));

idx = 3;
faces_vnum = size(faces_vertices{idx}, 1);

simplices = reshape(1:faces_vnum, 3,[])';

selected_points = samples{idx};
disp(size(selected_points))

% plotPolygons(simplices, faces_vertices{idx}, centers(idx, :));
plotPolygons(hulls{idx}, bounary_vertices{idx}, centers(idx, :),selected_points);


% 绘制函数：绘制多边形
function plotPolygons(faces, vertices, center, inside_points)
    if nargin < 4
        inside_points = [];
    end
    figure;
    hold on;
    axis equal;
    axis off;
    
    % 绘制多面体
    trisurf(faces, vertices(:, 1), vertices(:, 2), vertices(:, 3), ...
        'FaceColor', 'cyan', 'EdgeColor', 'black', 'FaceAlpha', 0.2);
    
    % 绘制顶点
    scatter3(vertices(:, 1), vertices(:, 2), vertices(:, 3), 30, 'b', 'filled');
    scatter3(center(1), center(2), center(3), 30, 'r', 'filled');
    if length(inside_points)>0
        scatter3(inside_points(:, 1), inside_points(:, 2), inside_points(:, 3), ...
                5, 'r','filled'); 
    end

    % 绘制球体
    [X_sphere, Y_sphere, Z_sphere] = sphere(20); % 生成单位球体
    for i = 1:size(vertices, 1)
        radius = sqrt(vertices(i, 4)); % 取半径平方值的平方根
    %    缩放球体并平移到每个顶点的位置
        surf(radius * X_sphere + vertices(i, 1), ...
             radius * Y_sphere + vertices(i, 2), ...
             radius * Z_sphere + vertices(i, 3), ...
             'FaceColor', 'cyan', 'FaceAlpha', 0.1); % 设置球体颜色和透明度
    end

    % radius = sqrt(center(4));
    % surf(radius * X_sphere + center(1), ...
    %          radius * Y_sphere + center(2), ...
    %          radius * Z_sphere + center(3), ...
    %          'FaceColor', 'y', 'FaceAlpha', 0.1);

    
    % % 绘制法向量
    % for i = 1:size(faces, 1)
    %     % 获取当前面的三个顶点
    %     v1 = vertices(faces(i, 1), :);
    %     v2 = vertices(faces(i, 2), :);
    %     v3 = vertices(faces(i, 3), :);
    % 
    %     % 计算两个边的向量
    %     edge1 = v2 - v1;
    %     edge2 = v3 - v1;
    % 
    %     % 计算法向量（叉乘两个边）
    %     normal = cross(edge1(1:3), edge2(1:3));
    % 
    %     % 归一化法向量
    %     normal = normal / norm(normal);
    % 
    %     % 计算当前面的中心点
    %     face_center = (v1(1:3) + v2(1:3) + v3(1:3)) / 3;
    % 
    %     % 绘制法向量
    %     quiver3(face_center(1), face_center(2), face_center(3), ...
    %         normal(1), normal(2), normal(3), 0.5, 'Color', 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    % end

    hold off;
end