$(function(){
    // 최초 화면 표시
    $('#content-box1').show();
    $('.result-box').hide();

    $("input:radio[name ='gender']:input[value='{{ physical_report.gender }}']").attr("checked", true);

    // 일일체크
    $('#physicalFrm').submit(function(e){
        e.preventDefault();
        $.ajax({
            type: 'POST',
            url:"/physical",
            dataType: 'json',
            data: $('#physicalFrm').serialize(),
            beforeSend: function(){
                //유저 메세지 표시
            },
            success: function(response){
                // 결과데이터
                var category = ["체질량지수(BMI)", "복부비만도(WHR)", "기초대사량(kcal)"];
                var value = [response.bmi,  response.whr, response.energy];
                var state = [response.bmi_state, response.whr_state, response.energy_state];

                // 결과 표시

                for(var i = 0; i < 3; i++){
                    var container = $("#empty-result-container").clone().show().attr("id","");
                    container.find(".title").html(category[i])
                    container.find(".text-info").html("<p>" + category[i] + ": " + value[i] + "로 " + state[i] + "입니다.</p>");
                    container.find(".container").addClass("bullet-container").addClass("bullet-container" + (i + 1));
                    $('#result-box1').append(container);
                }
                drawBulletBMI(response.bmi, "div.bullet-container1");
                drawBulletWHR(response.whr, response.gender, response.age, "div.bullet-container2");
                drawBulletEnergy(response.energy, response.gender, response.age, "div.bullet-container3");
            },
            error: function(request, status, error) {
               alert("code:"+request.status+"\n"+"message:"+request.responseText+"\n"+"error:"+error);
            }
        });
        return false;
    });
    // 식단체크
    $('#foodFrm').submit(function(e){
        e.preventDefault();
        if( $('#fcount').val() == 0){
            alert("오늘 섭취한 음식을 추가해주세요!");
        }else{
            var table_base_info = "<tr id='info_ment' class='info_men'><td colspan='5' height='150' align='center'>음식을 검색 후 선택하시면 추가됩니다.</td></tr>";
            $.ajax({
                type: 'POST',
                url:"/diet",
                dataType: 'json',
                data: $('#foodFrm').serialize(),
                beforeSend: function(){
                    //유저 메세지 표시
                },
                success: function(data){
                    var nutrition = ["섭취량(g)", "에너지(kcal)", "탄수화물(g)", "단백질(g)", "지방(g)",
                    "당류(g)", "나트륨(mg)", "클레스테롤(mg)", "포화지방(g)", "트랜스지방(g)"];
                    if( data.energy == 0)
                    {
                        alert("상단에 일일점검을 최초 1회이상 하셔야 결과를 확인하실수 있습니다");
                        $("#sel_food_list").html(table_base_info);
                    }else{
                        var foodAry = new Array(data.energy, data.gram, data.kcal, data.carbohydrate, data.protein, data.fat
                        ,data.sugars, data.salt, data.cholesterol ,data.saturatedfat ,data.transfat);

                        var info = "<table class='tb-diet'>";
                        info += "<tr>";
                        for(var i=0; i < nutrition.length; i++){
                            info += "<th>" + nutrition[i];
                        }
                        info += "<tr>";
                        for(var i=1; i < foodAry.length; i++){
                            info += "<td>" + foodAry[i].toFixed(2);
                        }
                        var resultBox = $('#empty-result-container').clone().show().attr("id","");
                        resultBox.find('.title').html("<h4>{{ user.username }}님이 오늘 섭취하신 음식의 영양정보입니다.</h4>");
                        resultBox.find('.text-info').html(info);
                        resultBox.find('.container').addClass('circle-container').addClass('circle-container1');
                        $('#result-box2').append(resultBox);
                         var flag = false;
                        drawCircleChart(foodAry, "div.circle-container1", flag);
                    }
                },
                error: function(request, status, error) {
                   alert("code:"+request.status+"\n"+"message:"+request.responseText+"\n"+"error:"+error);
                }
            });
            return false;
        }
    });
    // 기록보기
    $('#histFrm').submit(function(e){
        var gubun = $('#hist-gubun').val();
        var type = $('#hist-type').val();
        e.preventDefault();
        $.ajax({
            type: 'POST',
            url:"/hist",
            dataType: 'json',
            data: $('#histFrm').serialize(),
            beforeSend: function(){
                //유저 메세지 표시
            },
            success: function(data){
                if(data.length == 0){
                    if(type == "1")
                        alert("기록이 없습니다. 먼저 일일체크 측정을 해보시기 바랍니다.");
                    if(type == "2")
                        alert("기록이 없습니다. 먼저 식단체크 측정을 해보시기 바랍니다.");
                }else{
                    var resultBox = $('#empty-result-container').clone().show().attr("id","");
                    resultBox.find('.title').remove();
                    resultBox.find('.text-info').remove();
                    resultBox.find('.container').addClass('hist-container').addClass('hist-container' + type);
                    if(type == 1){
                        $('#hist-box1').append(resultBox);
                        if(gubun == 1) {
                            drawCurveBMI(data, 'div.hist-container' + type);
                        }
                        else if(gubun == 2){
                            drawCurveWHR(data, 'div.hist-container' + type, data[data.length-1].gender);
                        }
                        else if(gubun == 3){
                            drawCurveEnergy(data, 'div.hist-container' + type, data[data.length-1].gender, data[data.length-1].age);
                        }
                    }
                    if(type == 2){
                        $('#hist-box2').append(resultBox);
                        drawCurveChart(data, 'div.hist-container' + type);
                    }
                }
            },
            error: function(request, status, error) {
               alert("code:"+request.status+"\n"+"message:"+request.responseText+"\n"+"error:"+error);
            }
        });
    });

    $('input[name=reset]').on('click',function(){
        var gubun = $(this).attr("id").replace('reset-btn','');
        if(gubun == 1){
            $('#physicalFrm input[type=number]').val('');
        }else if(gubun == 2){
            $("#sel_food_list").html(table_base_info);
            $("#fcount").val(0);
        }
    });

    $('input[name=measure]').on('click',function(){
        $('.result-wrap').show();
        $('.hist-wrap').hide();
        $('.result-box').hide();

        var gubun = $(this).attr("id").replace('measure-btn','');
        $('#result-box' + gubun).empty();
        $('#result-box' + gubun).show();


        if(gubun == 1){
            $('#physicalFrm').submit();
        }else{
            $('#foodFrm').submit();
        }
    });

    $('input[name=hist]').on('click',function(){
        $('.result-wrap').hide();
        $('.hist-wrap').show();

        var type = $(this).attr("id").replace('hist-btn','');
        $('#hist-box' + type).empty();
        $('#hist-box' + type).show();

        if(type == 1){
            $('.hist-menu').show();
        }else{
            $('.hist-menu').hide();
        }
        $('#hist-type').val(type);
        $('#histFrm').submit();
    });

    // 입력창 메뉴 이벤트
    $('#content-menu li').on('click',function(){
        var targetId = $(this).attr('id').replace('content','');
        $('#content-menu li').removeClass('on');
        $(this).addClass('on')

        $('.result-box, .hist-box, .contents-box').hide()
        $('#result-box' + targetId + ', #content-box'  + targetId + ', #hist-box'  + targetId).show();
    });
    // 결과창 메뉴 이벤트
    $('#hist-menu li').on('click',function(){
        var gubun = $(this).attr('id').replace('hist','');
        $('#hist-menu li').removeClass('on');
        $(this).addClass('on')

        $('#hist-gubun').val(gubun);
        $('#histFrm').submit();
    });
});
