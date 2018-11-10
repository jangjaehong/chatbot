$(function(){
    $('#searchText').autocomplete({
        autoFocus: true,
        position:{ my: "left top", at: "left bottom", collision: "flip" },
        source : function( request, response ) {
             $.ajax({
                type: 'POST',
                url: "/autocomplete",
                dataType: "json",
                data: { value : request.term },
                success: function(data) {
                    //서버에서 json 데이터 response 후 목록에 뿌려주기 위함
                    response( $.map(data, function(item){
                        return {
                            value: item.name,
                            label: item.name + " | 1회 제공량 : "  + item.gram + " | 섭취량 : " + item.kcal,
                            name: item.name,
                            gram: item.gram,
                            kcal: item.kcal,
                            pk: item.pk,

                        }
                    }))
                }
            });
        },
        minLength: 1,
        open: function () {
            $(this).autocomplete('widget').css('height', '400px').css('z-index', '9999')
            return false;
        },
        select: function( event, ui ) {
            // 만약 검색리스트에서 선택하였을때 선택한 데이터에 의한 이벤트발생
            $('#sel_food_list > #info_ment').remove();

            // 검색창 초기화
            $('#searchText').val('');
            var foodNextNum = $('#sel_food_list > tr').length + 1;
            $('#fcount').val(foodNextNum);

            var selectFood = "<tr>"
            selectFood += "<td><input type='hidden' name='fpk' value='" + ui.item.pk + "' /><input type='hidden' name='fname' value='" + ui.item.name + "' />" + ui.item.name ;
            selectFood += "<td>" + ui.item.gram + "g";
            selectFood += "<td>" + ui.item.kcal + "kcal";
            selectFood += "<td><input type='number' name='fserving' value='1' min='1.0' />" + "인분";
            selectFood += "<td><input type='button' name='frBtn' onclick='foodRemove(this)' value='삭제' />";
            $('#tb-food').last().append(selectFood);

            return false;
        },
        focus: function( event, ui ){
            return false;
        }
    })
});


function foodRemove(obj){
    var count = $("#fcount").val();
    $("#fcount").val(count-1);
    $(obj).closest('tr').remove();

    if( $("#fcount").val() == 0 ){
        $("#sel_food_list").append("<tr><td colspan='5' height='150' align='center'>음식을 검색 후 선택하시면 추가됩니다.</td></tr>");
    }
}