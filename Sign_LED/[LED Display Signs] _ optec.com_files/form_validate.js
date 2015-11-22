function valid_required(field)
{
         if(field=="")          {
         return false;
         }
    return true;
}

function echeck(str) {

		var at="@"
		var dot="."
		var lat=str.indexOf(at)
		var lstr=str.length
		var ldot=str.indexOf(dot)
		if (str.indexOf(at)==-1){
		   alert("Invalid E-mail ID")
		   return false
		}
		if (str.indexOf(at)==-1 || str.indexOf(at)==0 || str.indexOf(at)==lstr){
		   alert("Invalid E-mail ID - 1")
		   return false
		}

		if (str.indexOf(dot)==-1 || str.indexOf(dot)==0 || str.indexOf(dot)==lstr){
		    alert("Invalid E-mail ID - 2")
		    return false
		}

		 if (str.indexOf(at,(lat+1))!=-1){
		    alert("Invalid E-mail ID - 3")
		    return false
		 }

		 if (str.substring(lat-1,lat)==dot || str.substring(lat+1,lat+2)==dot){
		    alert("Invalid E-mail ID - 4")
		    return false
		 }

		 if (str.indexOf(dot,(lat+2))==-1){
		    alert("Invalid E-mail ID - 5")
		    return false
		 }
		
		 if (str.indexOf(" ")!=-1){
		    alert("Invalid E-mail ID - 6")
		    return false
		 }
		 
		 if (str.indexOf("optec.com") >0){
		    alert("Invalid E-mail ID -7")
		    return false
		 }
 		 return true					
	}
	
	function focus_message(){
		if(document.myform.message.value=="Message"){
			document.myform.message.value='';
			document.myform.message.focus();
			return;
		}
		
	}

	function check_required()
{
        if(!valid_required(document.myform.name.value) || document.myform.name.value=="Name")
        {
			alert("Name is a required field!")
			document.myform.name.focus();
			return false;
        }
		if(!echeck(document.myform.email.value))
		{
			document.myform.email.focus();
			return false;
			
		}
		
		if(!valid_required(document.myform.phone.value) || document.myform.phone.value=="Phone")
        {
			alert("Phone is a required field!")
			document.myform.phone.focus();
			return false;
        }		
		
			if(!valid_required(document.myform.questions.value) || document.questions.name.value=="Website")
			{
				alert("Questions is a required field!")
				document.myform.questions.focus();
				return false;
			}
			
    return true;
}

function so(){
	$('#so').addClass('yourClass');
}