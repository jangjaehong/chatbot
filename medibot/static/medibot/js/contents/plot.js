 function measureWidth(data, range, target){
        var min_val = 0,
            max_val = 0,
            index = 0;

        for(var i = 0; i < data.length; i++)
        {
            if(data[i] >= target){
                index = i;
                break;
            }
        }

        if(index == 0){
            min_val = 0;
            max_val = data[index];
        }else{
            min_val = data[index-1];
            max_val = data[index];
        }
        var w = d3.scaleLinear()
            .domain([min_val, max_val]) //실제 값범위
            .range([0, range]); //변환할 값의 범위

        return  range * index + w(target);
    }

function bulletDefaultSettings(){
    return {
        gender: 0,
        age: 0,
        minValue: 0,
        maxValue: 100,
        xlabel:"",
        ylabel:"",
        rectColor: ["#BDBDBD", "#ABF200", "#FFE400", "#FFBB00", "#FF5E00", "#FF0000"],
        category: ["저체중", "정상", "과체중", "경도 비만", "중등도 비만", "고도 비만"],
        range: [18.5, 23.0, 25.0, 30.0, 35.0, 40.0],
        standard: 0,
        division: 1,
        duration: 1000,
        paddingTop:30,
        paddingRight:30,
        paddingBottom:20,
        paddingLeft:30,
    }
}

function drawBulletBMI(value, state, container){
    var config = bulletDefaultSettings();
    config.minValue = 0;
    config.maxValue = 40.0;
    config.rectColor = ["#FF0000", "#FF5E00", "#FFBB00", "#FFE400", "#ABF200","#B2EBF4"];
    config.range = [18.5, 23.0, 25.0, 30.0, 35.0, 40.0];
    config.ylabel = "체질량지수(BMI)";
    var bullet = loadBullet(container, value, state, config);
}
function drawBulletWHR(value, state, gender, age, container){
    var config = bulletDefaultSettings();
    config.gender = gender;
    config.age = age;
    config.minValue = 0;
    config.maxValue = 1.5;
    config.rectColor = ["#FF0000", "#ABF200", "#B2EBF4"];
    if(gender == 1){
        config.range = [0.5, 1, 1.5];
    }
    if(gender == 2){
        config.range = [0.5, 0.85, 1.5];
    }
    config.standard = (gender == 1 ? 1 : 0.86);
    config.ylabel = "복부비만도(WHR)"
    var bullet = loadBullet(container, value, state, config);
}
function drawBulletEnergy(value, state,  gender, age, container){
    var config = bulletDefaultSettings();
    config.gender = gender;
    config.age = age;
    config.minValue = 0;
    config.maxValue = 3000;
    config.rectColor = ["#FF0000", "#FF0000", "#FF0000","#ABF200", "#B2EBF4"];
    config.ylabel = "기초대사량(kcal)"
    if(gender == 1){
        if( age >= 20 && age <= 29 ){
            config.range = [1359.8, 1728, 2096.2, 2500.0, 3000.0];
        }
        if( age >= 30 && age <= 49 ){
            config.range = [1367.4, 1669.5,  1971.6, 2500.0, 3000.0];
        }
        if( age >= 50 ){
            config.range = [1359.8, 1728, 2096.2, 2500.0, 3000.0];
        }
    }
    if(gender == 2){
        if( age >= 20 && age <= 29 ){
            config.range = [1078.5, 1311.5, 1544.5, 2500.0, 3000];
        }
        if( age >= 30 && age <= 49 ){
            config.range = [1090.9, 1316.8, 1542.7, 2500.0, 3000.0];
        }
        if( age >= 50 ){
            config.range = [1023.9, 1252.5, 1481.1, 2500.0, 3000.0];
        }
    }
    var bullet = loadBullet(container, value, state, config);
}
function loadBullet(container, value, state, config){
     if(config == null) config = bulletDefaultSettings();

    var containerRect = d3.select(container).node().getBoundingClientRect();
    var containerWidth = containerRect.width,
        containerHeight = containerRect.height;

    var width = containerWidth - config.paddingLeft - config.paddingRight,
        height = containerHeight - config.paddingTop - config.paddingBottom;

    d3.select(container).select("svg").remove();
    var svg = d3.select(container).append("svg")
            .attr("width", width + config.paddingLeft + config.paddingRight)
            .attr("height", height + config.paddingTop  +  config.paddingBottom)

    // 범위를 표시하는 사각형
    var rangeGroup = svg.append("g")
        .attr("class", "rangeGroup")
        .attr("transform", "translate(" + config.paddingLeft + "," + config.paddingTop + ")")
    rangeGroup.selectAll("rect").data(config.range.reverse())
        .enter().append("rect")
        .attr("class", "ranges")
        .attr("width", 0)
        .attr("height", height)
        .attr("fill", function(d,i){ return config.rectColor[i]; })
        .transition()
        .duration(config.duration)
            .attr("width", function(d, i){ return (width/config.maxValue) * d; });

    // x축 수치
    var xScale = d3.scaleLinear()
        .domain([config.minValue , config.maxValue])
        .range([config.minValue, (width/config.maxValue) * config.maxValue ]);

    var xAxis = svg.append("g")
        .attr("class","xaxis")
        .attr("transform","translate(" + config.paddingLeft + "," + ( config.paddingTop + height) + ")")
        .attr("fill", "#fff")
        .attr("font-weight", "bold")
        .attr("text-anchor", "start")
        .call(d3.axisBottom(xScale));

    var lineData = [{"x": value, "y":0 }, {"x": value, "y": height}];
    var line = d3.line()
        .x(function(d) {return xScale(d.x); })
        .y(function(d) { return d.y; })

    //측정값을 표시하는 라인
    var resultLineGroup = svg.append("g")
        .attr("class","lineGroup")
        .attr("transform", "translate(" + (xScale(value) - 35) + "," + config.paddingTop + ")");

    resultLineGroup.append("path")
        .datum(lineData)
        .attr("class", "line")
        .attr("fill","none")
        .attr("stroke-width", "2")
        .attr("stroke", "#378AFF")
        .attr("d", line)

    resultLineGroup.transition()
        .duration(config.duration*4)
        .attr("transform", "translate(" + (xScale(value) - 35) + "," + config.paddingTop + ")");
        .attr("transform", "translate(" + config.paddingLeft + "," + config.paddingTop + ")");

    d3.select(container).selectAll("#arrow").classed('hidden', false);
    var arrow = d3.select(container).selectAll("#arrow")
        .style("top", (-config.paddingTop - 10)+ "px")
        .style("left", "0px")
        .transition()
        .duration(config.duration*4)
            .style("left", (xScale(value) - 35) + "px");
}

//꺽은선 그래프
function curveDefaultSettings(){
    return {
        color: ["#BDBDBD", "#ABF200", "#FFE400", "#FFBB00", "#FF5E00", "#FF0000"],
        range: [0, 18.5, 23.0, 25.0, 30.0, 35.0, 40.0],
        duration: 1000,
    }
}
function drawCurveBMI(data, container){
    var config = curveDefaultSettings();
    loadCurveChart(data, container, config)
}
function drawCurveWHR(data, container, gender){
    var config = curveDefaultSettings();
    if(gender == 1){
        config.range = [0.5, 1, 1.5];
    }
    if(gender == 2){
        config.range = [0.5, 0.85, 1.5];
    }
    config.color = ["#eee", "#ddd", "#ccc","#989898","#747474"];
    loadCurveChart(data, container, config)
}
function drawCurveEnergy(data, container, gender, age){
     var config = curveDefaultSettings();
      if(gender == 1){
        if( age >= 20 && age <= 29 ){
            config.range = [1359.8, 1728, 2096.2, 3000.0, 4000.0];
        }
        if( age >= 30 && age <= 49 ){
            config.range = [1367.4, 1669.5,  1971.6, 3000.0, 4000.0];
        }
        if( age >= 50 ){
            config.range = [1359.8, 1728, 2096.2, 3000.0, 4000.0];
        }
    }
    if(gender == 2){
        if( age >= 20 && age <= 29 ){
            config.range = [1078.5, 1311.5, 1544.5, 3000.0, 4000];
        }
        if( age >= 30 && age <= 49 ){
            config.range = [1090.9, 1316.8, 1542.7, 3000.0, 4000.0];
        }
        if( age >= 50 ){
            config.range = [1023.9, 1252.5, 1481.1, 3000.0, 4000.0];
        }
    }
    config.color = ["#eee", "#ddd", "#ccc","#989898","#747474"];
    loadCurveChart(data, container, config)
}
function loadCurveChart(data, container, config){
    function transition(path) {
        path.transition()
            .duration(2000)
            .attrTween("stroke-dasharray", tweenDash);
    }
    function tweenDash() {
        var l = this.getTotalLength(),
            i = d3.interpolateString("0," + l, l + "," + l);
        return function (t) { return i(t); };
    }
    if(config == null) config = curveDefaultSettings();
    var containerRect = d3.select(container).node().getBoundingClientRect()
        containerWidth = containerRect.width,
        containerHeight = containerRect.height;

    var margin = {top:40, right:40, bottom:40, left:40},
        width = containerWidth - margin.left - margin.right,
        height = containerHeight - margin.top - margin.bottom,
        duration = 1000;

    /* 공간 생성 */
    d3.select(container).select("svg").remove();
    var svg = d3.select(container).append("svg")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)


      /* 축 범위 설정 */
    var xScale = d3.scaleTime()
        .rangeRound([0, width])
        .domain(d3.extent(data, function(d) {return new Date(d.date);}))
        .nice(d3.timeDay);

    var yScale = d3.scaleLinear()
        .rangeRound([height, 0])
        .domain([0, d3.max(config.range, function(d){ return d; })]);

     /* 축 위치 및 간격 설정 */
    var xAxis = d3.axisBottom(xScale)
        .ticks(d3.timeDay)
        .tickFormat(d3.timeFormat("%Y/%m/%d"));

    var yAxis = d3.axisLeft(yScale);

    var groupAxis = svg.append("g")
        .attr("transform","translate(" + margin.left + "," + margin.top +  ")")
     /* 축 추가 */
    groupAxis.append("g")
        .attr("class","xaxis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);
    groupAxis.append("g")
        .attr("class","yaxis")
        .call(yAxis);

    //Draw Line
    var line = d3.line()
        .x(function(d) {return xScale(new Date(d.date)); })
        .y(function(d) { return yScale(d.value); })

    //Input Line
    var groupLine = svg.append("g")
            .attr("class", "path")
            .attr("transform", "translate(" + (21 - margin.left) + "," + margin.top + ")")
    groupLine.append("path")
        .datum(data)
        .attr("class", "line")
        .attr("fill","none")
        .attr("stroke-width", "2")
        .attr("stroke", "#fff")
        .attr("d", line)
        .call(transition)

    var groupDot = svg.append("g")
            .attr("class", "dots")
            .attr("transform", "translate(" + (21 - margin.left) + "," + margin.top + ")")
    groupDot.selectAll("circle").data(data)
        .enter().append("circle")
        .attr("r", 8)
        .attr("cx", function(d) { return xScale(new Date(d.date)); })
        .attr("cy", height)
        .attr("fill",function(d){
            if(d.value == 0){
                return "#000";
            }
            else if(d.value <= 18.5){
                return config.color[0];
            }
            else if(d.value >= 18.5 && d.value < 23.0){
                return config.color[1];
            }
            else if(d.value >= 23.0 && d.value < 25.0){
                return config.color[2];
            }
            else if(d.value >= 25.0 && d.value < 30.0){
                return config.color[3];
            }
            else if(d.value >= 30.0 && d.value < 40.0){
                return config.color[4];
            }
            else if(d.value >= 40.0){
                return config.color[5];
            }
        })
        .on("mouseover", function(d){
            d3.select(container).selectAll("#arrow").classed('hidden', false);
            var arrow = d3.select(container).selectAll("#arrow")
                .style("top",  (d3.select(this).attr("cy")/2) + "px")
                .style("left", (d3.select(this).attr("cx") - 50) + "px")
                .transition()
                    .duration(duration/2)
                    .style("top", (d3.select(this).attr("cy") - 10)+ "px");
        })
        .on("mouseout", function(d){
            d3.select(container).select("#arrow")
                .style("top", d3.select(this).attr("cy") + "px")
                .transition()
                .duration(duration/2)
                    .style("top", 0);
            d3.select(container).select("#arrow").classed('hidden', true);
        })
        .transition()
            .duration(config.duration)
            .attr("cy", function(d) { return yScale(d.value); })
}

// 원형그래프
function drawCircleChart(data, container, flag){
    function dataUnit(unit){
        if(unit == 5 || unit == 6){
            return "g";
        }else{
            return "kcal";
        }
    }
    function calc(total_energy, flag){
        var standards = new Array();85
        if(flag == false){
            standards[0] = (total_energy * 0.55); //탄수화물
            standards[1] = (total_energy * 0.07); //단백질
            standards[2] = (total_energy * 0.1); //지방
        }else{
            standards[0] = (total_energy * 0.65);
            standards[1] = (total_energy * 0.20);
            standards[2] = (total_energy * 0.2);
        }
        standards[3] =  (total_energy * 0.1) ; //당류
        standards[4] = 1957; //나트륨
        standards[5] = 759; // 남자, 콜레스테롤
        standards[6] = (total_energy * 0.07); //불포화지방
        standards[7] = (total_energy * 0.01); //트랜스지방
        //standards[3] = 600; // 여자
        return standards;
    }
    function convertKcal(data, container){
         var kcalBase = [4, 4, 9, 4, 1, 1, 9, 9];
         var reData = [];
         for(var i=0; i < data.length; i++)
         {
            reData[i] = data[i] * kcalBase[i];
         }
         return reData;
    }
    function dataRender(data, flag){
        var dataValue = [];
        var dataPercent = [];
        var BaseEnergy = data[0];
        var intakeEnergy = data[2];
        var standardsData = calc(BaseEnergy, flag); //섭취기준
        var intakeData = convertKcal(data.slice(3, data.length))

        dataValue[0] = [BaseEnergy, intakeEnergy];
        for(var i=0; i < intakeData.length; i++){
            dataValue[i+1] =  [standardsData[i].toFixed(1), intakeData[i].toFixed(1)];
        }
        dataPercent[0] = (intakeEnergy/BaseEnergy)*100;
        for(var i=0; i< intakeData.length; i++){
            if(intakeData[i] == 0){
                    dataPercent[i+1] = 0;
            }else{
                dataPercent[i+1] = (intakeData[i]/standardsData[i])*100;
            }
        }
        return {
            dataValue :dataValue,
            dataPercent: dataPercent
        };
    }

    var dataSet = dataRender(data, flag);

    var containerRect = d3.select(container).node().getBoundingClientRect()
        containerWidth = containerRect.width,
        containerHeight = containerRect.height;

    var svgWidth = (containerWidth/3),
        svgHeight = (containerHeight/3);

    var dataName = ["총섭취량", "탄수화물", "단백질", "지방", "당류", "나트륨", "콜레스테롤", "포화지방", "트랜스지방"];

    var chart_container = d3.select(container);
    for(var c=0; c < dataName.length; c++){
        var config = liquidFillGaugeDefaultSettings();
        config.circleThickness = 0.1;
        config.textVertPosition = 0.5;
        config.waveAnimateTime = 2000;
        config.waveHeight = 0.3;
        config.waveCount = 1;
        config.valueCountUp = false;
        //config.textColor = "#A4DBf8";
        config.textColor = "#333";

        var svg = chart_container.append("svg")
            .attr("id", "fillgauge" + c)
            .attr("class","fillgauge fillgauge" + c)
            .attr("width", svgWidth)
            .attr("height", svgHeight)
            .on("mouseover", function(d, i){
                d3.select(this).selectAll(".frontGroup").classed('hidden', true);
                d3.select(this).selectAll(".backGroup").classed('hidden', false);
            })
            .on("mouseout", function(d, i){
                d3.select(this).selectAll(".frontGroup").classed('hidden', false);
                d3.select(this).selectAll(".backGroup").classed('hidden', true);
            });

        var value = dataSet.dataPercent[c],
            title = dataName[c],
            standard = "권장량:" + dataSet.dataValue[c][0].toString() + dataUnit(c),
            intake = "섭취량:" + dataSet.dataValue[c][1].toString() + dataUnit(c);

        if(value> 100){
            config.circleColor = "#FF0000";
            config.waveColor = "#FF0000";
            config.waveTextColor = "#fff"
        }else{
            config.circleColor = "#178BCA";
            config.waveColor = "#178BCA";
        }

        var gauge = loadLiquidFillGauge("fillgauge" + c, value, title, standard, intake, config);
    }
}
// 원 그래프 그리기
function liquidFillGaugeDefaultSettings(){
    return {
        minValue: 0, // The gauge minimum value.
        maxValue: 100, // The gauge maximum value.
        circleThickness: 0.05, // The outer circle thickness as a percentage of it's radius.
        circleFillGap: 0.05, // The size of the gap between the outer circle and wave circle as a percentage of the outer circles radius.
        circleColor: "#178BCA", // The color of the outer circle.
        waveHeight: 0.05, // The wave height as a percentage of the radius of the wave circle.
        waveCount: 1, // The number of full waves per width of the wave circle.
        waveRiseTime: 1000, // The amount of time in milliseconds for the wave to rise from 0 to it's final height.
        waveAnimateTime: 18000, // The amount of time in milliseconds for a full wave to enter the wave circle.
        waveRise: true, // Control if the wave should rise from 0 to it's full height, or start at it's full height.
        waveHeightScaling: true, // Controls wave size scaling at low and high fill percentages. When true, wave height reaches it's maximum at 50% fill, and minimum at 0% and 100% fill. This helps to prevent the wave from making the wave circle from appear totally full or empty when near it's minimum or maximum fill.
        waveAnimate: true, // Controls if the wave scrolls or is static.
        waveColor: "#178BCA", // The color of the fill wave.
        waveOffset: 0, // The amount to initially offset the wave. 0 = no offset. 1 = offset of one full wave.
        titleVertPosition: .9,
        textVertPosition: .5, // The height at which to display the percentage text withing the wave circle. 0 = bottom, 1 = top.
        frontTextSize: 1, // The relative height of the text to display in the wave circle. 1 = 50%
        backTextSize:0.4,
        titleSize: 0.5,
        valueCountUp: true, // If true, the displayed value counts up from 0 to it's final value upon loading. If false, the final value is displayed.
        displayPercent: true, // If true, a % symbol is displayed after the value.
        textColor: "#045681", // The color of the value text when the wave does not overlap it.
        waveTextColor: "#A4DBf8" // The color of the value text when the wave overlaps it.
    };
}
function loadLiquidFillGauge(elementId, value, title, standard, intake, config) {
    if(config == null) config = liquidFillGaugeDefaultSettings();

    var gauge = d3.select("#" + elementId);
    var radius = Math.min(parseInt(gauge.style("width")) - 40, parseInt(gauge.style("height"))- 40)/2;
    var locationX = parseInt(gauge.style("width"))/2 - radius;
    var locationY = parseInt(gauge.style("height"))/2 - radius;
    var fillPercent = Math.max(config.minValue, Math.min(config.maxValue, value))/config.maxValue;

    var waveHeightScale;
    if(config.waveHeightScaling){
        waveHeightScale = d3.scaleLinear()
            .range([0,config.waveHeight,0])
            .domain([0,50,100]);
    } else {
        waveHeightScale = d3.scaleLinear()
            .range([config.waveHeight,config.waveHeight])
            .domain([0,100]);
    }

    var titlePixels = (config.titleSize*radius/2);
    var frontTextPixels = (config.frontTextSize*radius/2);
    var backTextPixels  = (config.backTextSize*radius/2);
    var textFinalValue = parseFloat(value).toFixed(1);
    var textStartValue = config.valueCountUp?config.minValue:textFinalValue;
    var percentText = config.displayPercent?"%":"";
    var circleThickness = config.circleThickness * radius;
    var circleFillGap = config.circleFillGap * radius;
    var fillCircleMargin = circleThickness + circleFillGap;
    var fillCircleRadius = radius - fillCircleMargin;
    var waveHeight = fillCircleRadius*waveHeightScale(fillPercent*100);

    var waveLength = fillCircleRadius*2/config.waveCount;
    var waveClipCount = 1+config.waveCount;
    var waveClipWidth = waveLength*waveClipCount;

    // Rounding functions so that the correct number of decimal places is always displayed as the value counts up.
    var textRounder = function(value){ return Math.round(value); };
    if(parseFloat(textFinalValue) != parseFloat(textRounder(textFinalValue))){
        textRounder = function(value){ return parseFloat(value).toFixed(1); };
    }
    if(parseFloat(textFinalValue) != parseFloat(textRounder(textFinalValue))){
        textRounder = function(value){ return parseFloat(value).toFixed(2); };
    }

    // Data for building the clip wave area.
    var data = [];
    for(var i = 0; i <= 40*waveClipCount; i++){
        data.push({x: i/(40*waveClipCount), y: (i/(40))});
    }

    // Scales for drawing the outer circle.
    var gaugeCircleX = d3.scaleLinear().range([0,2*Math.PI]).domain([0,1]);
    var gaugeCircleY = d3.scaleLinear().range([0,radius]).domain([0,radius]);

    // Scales for controlling the size of the clipping path.
    var waveScaleX = d3.scaleLinear().range([0,waveClipWidth]).domain([0,1]);
    var waveScaleY = d3.scaleLinear().range([0,waveHeight]).domain([0,1]);

    // Scales for controlling the position of the clipping path.
    var waveRiseScale = d3.scaleLinear()
        // The clipping area size is the height of the fill circle + the wave height, so we position the clip wave
        // such that the it will overlap the fill circle at all when at 0%, and will totally cover the fill
        // circle at 100%.
        .range([(fillCircleMargin+fillCircleRadius*2+waveHeight),(fillCircleMargin-waveHeight)])
        .domain([0,1]);
    var waveAnimateScale = d3.scaleLinear()
        .range([0, waveClipWidth-fillCircleRadius*2]) // Push the clip area one full wave then snap back.
        .domain([0,1]);

    // Scale for controlling the position of the text within the gauge.
    var textRiseScaleY = d3.scaleLinear()
        .range([fillCircleMargin+fillCircleRadius*2,(fillCircleMargin+frontTextPixels*0.7)])
        .domain([0,1]);

    // Center the gauge within the parent SVG.
    var gaugeGroup = gauge.append("g")
        .attr('transform','translate('+locationX+','+locationY+')');

    // Draw the outer circle.
    var gaugeCircleArc = d3.arc()
        .startAngle(gaugeCircleX(0))
        .endAngle(gaugeCircleX(1))
        .outerRadius(gaugeCircleY(radius))
        .innerRadius(gaugeCircleY(radius-circleThickness));
    gaugeGroup.append("path")
        .attr("d", gaugeCircleArc)
        .style("fill", config.circleColor)
        .attr('transform','translate('+radius+','+radius+')');

    var title1 = gaugeGroup.append("text")
        .text(title)
        .attr("text-anchor", "middle")
        .attr("font-size", titlePixels + "px")
        .style("fill", config.textColor)
        .style("font-weight","bold")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.titleVertPosition)+')');

    var frontGroup1 = gaugeGroup.append("g")
        .attr("class","frontGroup")
        .attr("text-anchor", "middle")
        .attr("font-size", frontTextPixels + "px")
        .style("fill", config.textColor)
        .classed('hidden', false);

    // Text where the wave does not overlap.
    var frontText1 = frontGroup1.append("text")
        .text(textRounder(textStartValue) + percentText)
        .attr("class", "liquidFillGaugeText")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.textVertPosition)+')');

    var backGroup1 = gaugeGroup.append("g")
        .attr("class","backGroup")
        .attr("text-anchor", "middle")
        .attr("font-size", backTextPixels + "px")
        .style("fill", config.textColor)
        .classed('hidden', true);

    var backText1 = backGroup1.append("text")
        .text(standard)
        .attr("class", "liquidFillGaugeText")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.textVertPosition + 0.1)+')');

    var backText3 = backGroup1.append("text")
        .text(intake)
        .attr("class", "liquidFillGaugeText")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.textVertPosition - 0.1)+')');

    // The clipping wave area.
    var clipArea = d3.area()
        .x(function(d) { return waveScaleX(d.x); } )
        .y0(function(d) { return waveScaleY(Math.sin(Math.PI*2*config.waveOffset*-1 + Math.PI*2*(1-config.waveCount) + d.y*2*Math.PI));} )
        .y1(function(d) { return (fillCircleRadius*2 + waveHeight); } );
    var waveGroup = gaugeGroup.append("defs")
        .append("clipPath")
        .attr("id", "clipWave" + elementId);
    var wave = waveGroup.append("path")
        .datum(data)
        .attr("d", clipArea)
        .attr("T", 0);

    // The inner circle with the clipping wave attached.
    var fillCircleGroup = gaugeGroup.append("g")
        .attr("clip-path", "url(#clipWave" + elementId + ")");
    fillCircleGroup.append("circle")
        .attr("cx", radius)
        .attr("cy", radius)
        .attr("r", fillCircleRadius)
        .style("fill", config.waveColor);


    var title2 = fillCircleGroup.append("text")
        .text(title)
        .attr("text-anchor", "middle")
        .attr("font-size", titlePixels + "px")
        .style("fill", config.waveTextColor)
        .style("font-weight","bold")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.titleVertPosition)+')');

    var frontGroup1 = fillCircleGroup.append("g")
        .attr("class","frontGroup")
        .attr("text-anchor", "middle")
        .attr("font-size", frontTextPixels + "px")
        .style("fill", config.waveTextColor)
        .classed('hidden', false);

    // Text where the wave does overlap.
    var frontText2 = frontGroup1.append("text")
        .text(textRounder(textStartValue) + percentText)
        .attr("class", "liquidFillGaugeText")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.textVertPosition)+')');

    var backGroup2 = fillCircleGroup.append("g")
        .attr("class","backGroup")
        .attr("text-anchor", "middle")
        .attr("font-size", backTextPixels + "px")
        .style("fill", config.waveTextColor)
        .classed('hidden', true);

    var backText2 = backGroup2.append("text")
        .text(standard)
        .attr("class", "liquidFillGaugeText")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.textVertPosition + 0.1)+')');

    var backText4 = backGroup2.append("text")
        .text(intake)
        .attr("class", "liquidFillGaugeText")
        .attr('transform','translate('+radius+','+textRiseScaleY(config.textVertPosition - 0.1)+')');


    // Make the value count up.
    if(config.valueCountUp){
        var textTween = function(){
            var i = d3.interpolate(this.textContent, textFinalValue);
            return function(t) { this.textContent = textRounder(i(t)) + percentText; }
        };
        frontText1.transition()
            .duration(config.waveRiseTime)
            .tween("text", textTween);
        frontText2.transition()
            .duration(config.waveRiseTime)
            .tween("text", textTween);
    }


    // Make the wave rise. wave and waveGroup are separate so that horizontal and vertical movement can be controlled independently.
    var waveGroupXPosition = fillCircleMargin+fillCircleRadius*2-waveClipWidth;
    if(config.waveRise){
        waveGroup.attr('transform','translate('+waveGroupXPosition+','+waveRiseScale(0)+')')
            .transition()
            .duration(config.waveRiseTime)
            .attr('transform','translate('+waveGroupXPosition+','+waveRiseScale(fillPercent)+')')
            .on("start", function(){
                wave.attr('transform','translate(1,0)');
            }); // This transform is necessary to get the clip wave positioned correctly when waveRise=true and waveAnimate=false. The wave will not position correctly without this, but it's not clear why this is actually necessary.
    } else {
        waveGroup.attr('transform','translate('+waveGroupXPosition+','+waveRiseScale(fillPercent)+')');
    }

    if(config.waveAnimate) animateWave();

    function animateWave() {
        wave.attr('transform','translate('+waveAnimateScale(wave.attr('T'))+',0)');
        wave.transition()
            .duration(config.waveAnimateTime * (1-wave.attr('T')))
            .ease(d3.easeLinear)
            .attr('transform','translate('+waveAnimateScale(1)+',0)')
            .attr('T', 1)
            .on("end", function(){
                wave.attr('T', 0);
                animateWave(config.waveAnimateTime);
            });
    }

    function GaugeUpdater(){
        this.update = function(value){
            var newFinalValue = parseFloat(value).toFixed(2);
            var textRounderUpdater = function(value){ return Math.round(value); };
            if(parseFloat(newFinalValue) != parseFloat(textRounderUpdater(newFinalValue))){
                textRounderUpdater = function(value){ return parseFloat(value).toFixed(1); };
            }
            if(parseFloat(newFinalValue) != parseFloat(textRounderUpdater(newFinalValue))){
                textRounderUpdater = function(value){ return parseFloat(value).toFixed(2); };
            }

            var textTween = function(){
                var i = d3.interpolate(this.textContent, parseFloat(value).toFixed(2));
                return function(t) { this.textContent = textRounderUpdater(i(t)) + percentText; }
            };

            text1.transition()
                .duration(config.waveRiseTime)
                .tween("text", textTween);
            text2.transition()
                .duration(config.waveRiseTime)
                .tween("text", textTween);

            var fillPercent = Math.max(config.minValue, Math.min(config.maxValue, value))/config.maxValue;
            var waveHeight = fillCircleRadius*waveHeightScale(fillPercent*100);
            var waveRiseScale = d3.scaleLinear()
                // The clipping area size is the height of the fill circle + the wave height, so we position the clip wave
                // such that the it will overlap the fill circle at all when at 0%, and will totally cover the fill
                // circle at 100%.
                .range([(fillCircleMargin+fillCircleRadius*2+waveHeight),(fillCircleMargin-waveHeight)])
                .domain([0,1]);
            var newHeight = waveRiseScale(fillPercent);
            var waveScaleX = d3.scaleLinear().range([0,waveClipWidth]).domain([0,1]);
            var waveScaleY = d3.scaleLinear().range([0,waveHeight]).domain([0,1]);
            var newClipArea;
            if(config.waveHeightScaling){
                newClipArea = d3.area()
                    .x(function(d) { return waveScaleX(d.x); } )
                    .y0(function(d) { return waveScaleY(Math.sin(Math.PI*2*config.waveOffset*-1 + Math.PI*2*(1-config.waveCount) + d.y*2*Math.PI));} )
                    .y1(function(d) { return (fillCircleRadius*2 + waveHeight); } );
            } else {
                newClipArea = clipArea;
            }

            var newWavePosition = config.waveAnimate?waveAnimateScale(1):0;
            wave.transition()
                .duration(0)
                .transition()
                .duration(config.waveAnimate?(config.waveAnimateTime * (1-wave.attr('T'))):(config.waveRiseTime))
                .ease('linear')
                .attr('d', newClipArea)
                .attr('transform','translate('+newWavePosition+',0)')
                .attr('T','1')
                .each("end", function(){
                    if(config.waveAnimate){
                        wave.attr('transform','translate('+waveAnimateScale(0)+',0)');
                        animateWave(config.waveAnimateTime);
                    }
                });
            waveGroup.transition()
                .duration(config.waveRiseTime)
                .attr('transform','translate('+waveGroupXPosition+','+newHeight+')')
        }
    }
    return new GaugeUpdater();
}

