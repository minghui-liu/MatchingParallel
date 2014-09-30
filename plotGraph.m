function h = plotGraph(nodes1, nodes2, edges1, edges2)
    h = figure;
    subplot(1,2,1);
    scatter(nodes1(:,1), nodes1(:,2),200,'red','fill');
    hold on;
    for i=1:size(nodes1,1)
        for j=1:i
            if edges1(i,j) ~= 0
                plot( [nodes1(i,1);nodes1(j,1)],[nodes1(i,2);nodes1(j,2)]);
            end
        end
    end
    hold off;
    title('Image 1');
    subplot(1,2,2);
    scatter(nodes2(:,1), nodes2(:,2),200,'red','fill');
    hold on;
    for i=1:size(nodes2,1)
        for j=1:i
            if edges2(i,j) ~= 0
                plot( [nodes2(i,1);nodes2(j,1)],[nodes2(i,2);nodes2(j,2)]);
            end
        end
    end
    hold off;
    title('Image 2');