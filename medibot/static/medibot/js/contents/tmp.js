가로 스택막대그래프
function drawBulletBMI(bmiValue, container){
    var color = ["#BDBDBD", "#ABF200", "#FFE400", "#FFBB00", "#FF5E00", "#FF0000"];
    var data = {
        "ranges":[18.5, 23.0, 25.0, 30.0, 35.0, 40.0],
        "measure":[bmiValue], //측정시 변경되는 값
        "categorys":["저체중", "정상", "과체중", "경도 비만", "중등도 비만", "고도 비만"],
    };
    var containerRect = d3.selectAll(container).node().getBoundingClientRect();
    var containerWidth = containerRect.width,
        containerHeight = containerRect.height;
    var  padding = {top:40, right:40, bottom:40, left:40},
    width = containerWidth - padding.left - padding.right,
    height = containerHeight - padding.top - padding.bottom,
    division = 6,
    duration = 1000;

    d3.select(container).select("svg").remove();
    var svg = d3.select(container).append("svg")
            .attr("width", width + padding.right + padding.left)
            .attr("height", height + padding.top + padding.bottom)


    // 구역 생성
    var rectGroup = svg.append("g")
                    .attr("class","rectGroup")
                    .attr("fill-opacity" , "0.9");
    rectGroup.selectAll("rect.ranges").data(data.ranges)
        .enter().append("rect")
            .attr("class", function(d, i) { return "range" + i + " ranges" })
            .attr("x", function(d, i) { return i * (width / division) + padding.left; })
            .attr("y", padding.top)
            .attr("width", width / division)
            .attr("height", height)
            .attr("fill", function(d,i){
                return color[i];
            })
            .on("mouseover", function(d, i){
                d3.select(this).attr("fill-opacity" , "1")
                var tooltip = d3.select(container).select("#tooltip")
                    .style("top", padding.top - 20 + "px")
                    .style("left", i * (width / division) + padding.left + padding.right + "px")
                    .transition()
                        .duration(duration/2)
                        .style("top", padding.top - padding.bottom  + "px");

                tooltip.select(".text1")
                    .text("bmi : " + (i == 0 ? 0 : data.ranges[i-1]) + "~" + d)

                d3.select(container).select("#tooltip").classed('hidden', false);
            })
            .on("mouseout", function(d, i){
                d3.select(this).attr("fill-opacity" , "0.9")
                d3.select(container).select("#tooltip").classed('hidden', true);
            });

    var categoryGroup = svg.append("g")
        .attr("font-size", "1em")
        .attr("font-weight", "bold")
        .attr("text-anchor", "middle")

    //분류 표시
    categoryGroup.selectAll("text.category").data(data.categorys)
        .enter().append("text")
            .attr("class", "category")
            .attr("x", function(d, i) { return i * (width / division) + (width / division * 0.5) + padding.left; })
            .attr("y", 0)
            .attr("fill", "#FFF")
            .text(function(d) { return d; })
         .transition()
         .duration(duration)
            .attr("y", (height/2) + padding.top);

    var resultGroup = svg.append("g")
            .attr("transform","translate(" + (measureWidth(data.ranges,  width/division, data.measure) - (width/24)) + "," + (-10) + ")")
    // 측정 결과 표시
    resultGroup.selectAll("rect.measure").data(data.measure)
        .enter().append("rect")
            .attr("class", "measure")
            .attr("width", width / 12)
            .attr("height", padding.bottom)
            .attr("fill","#378AFF")
    resultGroup.selectAll("rect.measure").data(data.measure)
        .enter().append("text")
        .text(function(d){ return "나의 BMI: " + d + " / 정상";} )

    var lineData = [{"x": data.measure, "y": 20}, {"x": data.measure, "y": 80},]
    var resultLine = d3.line()
                        .x(function(d) { return measureWidth(data.ranges,  width/division, d.x);})
                        .y(function(d) { return d.y; })

    var lineGroup = svg.append("path")
                        .attr("d", resultLine(lineData))
                        .attr("stroke", "#378AFF")
                        .attr("stroke-width", 2)
                        .attr("fill", "none");

}