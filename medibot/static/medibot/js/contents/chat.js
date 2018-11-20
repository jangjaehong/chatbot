$(function(){
    $('#chatFrm').submit(function(e){
        /* 새로 고침 방지 */
        e.preventDefault();
        var input_msg = $('#input_message').val()
        $('#msg').val(input_msg);
        $('#input_message').val('');

        $.ajax({
            type: 'POST',
            url:"/ajaxpost",
            dataType: 'json',
            data: $('#chatFrm').serialize(),
            beforeSend: function(){
                //유저 메세지 표시
                addMessage('user', '{{ user.username }}', input_msg, );
                move_chat_scroll();
            },
            success: function(data, textStatus){
                 $.each(data, function(i, item){
                    addMessage('com', item.name, item.message);
                 });
            },
            complete: function(){
                move_chat_scroll();
            },
            error: function(request, status, error) {
               alert("code:"+request.status+"\n"+"message:"+request.responseText+"\n"+"error:"+error);
            }
        });
        return false;
    });
    /* 입력 버튼 이벤트 */
    $('#input_message').on('keydown', function(e){
        if(e.keyCode == 13){
            if(e.shiftKey == true){
                var tmp = $('#input_message').val();
                tmp = tmp + "\n";
                $('#input_message').val(tmp)
            }else{
                $('#messageFrm').submit();
            }
            return false;
        }
    });

})

function addMessage(sender, name, message){
    var user_img = "{% static 'medibot/img/icon/ic_user.png' %}";
    var com_img = "{% static 'medibot/img/icon/ic_com.png' %}";

    var speechBubble = '<div class="msg-box ' + sender + ' mb_15">';
    if( sender == 'user'){
        speechBubble += '<div class="profile-wrap"><img src="' + user_img + '"/></div>';
    }else{
        speechBubble += '<div class="profile-wrap"><img src="' + com_img + '"/></div>';
    }
    speechBubble += '<div class="content-wrap">';
    speechBubble += '<div class="name">' + name + '</div>';
    speechBubble += '<div class="content">' + message + '</div>';
    speechBubble += '</div>';
    $('#sentence').append(speechBubble);
    move_chat_scroll();
};

function move_chat_scroll(){
    var sentence_last_child = $("#sentence").children(":last");
    var scrollPosition = sentence_last_child.offset().top;
    $('#sentence').animate({scrollTop : scrollPosition}, 300);
};