$(document).ready(function () {

    /* Initially load the table json file containing data*/
    $.getJSON('./table.json', function(data){
        let keys = Object.keys(data[0]);

        const loadMoreButton = document.getElementById("showallbutton");

        let startIndex = 0;
        const table = document.getElementById("inTable");
        /* Create the headers for the table*/
        let head = table.createTHead();
        let hrow = head.insertRow(0);
        for (let i = 0; i < keys.length; i++){
            let hcol = hrow.insertCell(i);
            hcol.innerHTML = keys[i];
        }
        /* Load the table rows using below function*/
        let table2 = loadTableElements(startIndex, data);

        /*let jtable = $('#inTable').DataTable({searching: false, paging: false, info: false, "bAutoWidth": false,});*/
        /* Make the checkboxed for each column and sort the columns by selected sort column*/
        makeCheckBoxes(data);
        /*loadTableData(data);*/
        let stat_index = get_plot_column(table2, "lineaware_stat");
        const sortdrop = document.getElementById("sortby");
        sortdrop.selectedIndex = stat_index;
        const sortorder = document.getElementById("sortorder");
        sortorder.selectedIndex = 1;
        /* reload the page the set each of these*/
        reloadPage()


    });


    /* initialise the data table for jquery*/
    $('#inTable').load(function () {
        let table = $('#inTable').DataTable({"ordering": false, searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true,});
        let row = table.row(0).data();
        let index = get_plot_column(table, "plot_path");
        var newpath = row[index];
        changeImg(newpath);
        makeInfoTable(table, row);
        table.column(index).visible(true);
        table.column(index).visible(false);

        
    });

    /* Show images and the infor table when a row is clicked in the table*/
    $('#inTable').on('click', 'tbody tr', function () {
        /*
        let table = $('#inTable').DataTable({retrieve:true});
        let index = get_plot_column(table, "plot_path");

        var d0 = $(this).find("td").eq(0).text();
        var d1 = $(this).find("td").eq(1).text();
        const info = document.getElementById("plotinfo");
        */
        let table = $('#inTable').DataTable({retrieve:true});
        let index = get_plot_column(table, "plot_path");
        var rowdata=$('#inTable').dataTable({retrieve:true}).fnGetData($(this)); 
        var newpath = rowdata[index];
        /*var newpath = table.rows(table.rows()[0][this.rowIndex]-1).data()[0][index];*/
        const image = document.getElementById("image");
        image.setAttribute("value", $(this).context.sectionRowIndex);
        changeImg(newpath);
        makeInfoTable(table, rowdata);
        $('#inTable tr').removeClass("selected");
        $(this).addClass('selected');

    });
});

function nextTablePage(event){
    /* Move to the next page of the table*/
    /* This selects the next 10 items from the entire json dataset and displays them in a table*/
    let itemsPerPage = 10;
    $.getJSON('./table.json', function(data){
        let lpage = document.getElementById("lastpagebutton");
        let npage = document.getElementById("lastpagebutton");
        npage.value = parseInt(npage.value) + parseInt(itemsPerPage);
        lpage.value = parseInt(lpage.value) + parseInt(itemsPerPage);
        
        const image = document.getElementById("image");
        image.setAttribute("value", 0);
        
        data = checkFilters(data);
        /*event.target.value = parseInt(event.target.value) + parseInt(itemsPerPage);*/
        if (parseInt(npage.value) > parseInt(data.length)){
            window.alert("End of table");
        }
        else{
            let table = loadTableElements(npage.value, data);
            setVisible(data);
        }
});
};

function lastTablePage(event){
    $.getJSON('./table.json', function(data){
        /* Move to the previous page of the table*/
        /* This selects the previous 10 items from the entire json dataset and displays them in a table*/
        let itemsPerPage = 10;
        let lpage = document.getElementById("lastpagebutton");
        let npage = document.getElementById("lastpagebutton");
        npage.value = parseInt(npage.value) - parseInt(itemsPerPage);
        lpage.value = parseInt(lpage.value) - parseInt(itemsPerPage);
        
        const image = document.getElementById("image");
        image.setAttribute("value", 9);
        
        data = checkFilters(data);
        if (lpage.value < 0){
            window.alert("End of table");
        }
        else{
            let table = loadTableElements(lpage.value, data);
            setVisible(data);
        }
});
};

function reloadPage(){
    /* reload the entire page checking the filters and loading appropriate table rows*/
    $.getJSON('./table.json', function(data){
        let lpage = document.getElementById("lastpagebutton");
        let npage = document.getElementById("lastpagebutton");
        lpage.value = 0;
        npage.value = 0;
        data = checkFilters(data);
        loadTableElements(0, data);
        setVisible(data);
    });
}

function sortJSONTable(jdata){
    let dropdown = document.getElementById("sortby");
    let order = document.getElementById("sortorder");
    var text = dropdown.options[dropdown.selectedIndex].text;
    if (order.value == "Ascending"){
        jdata.sort(function(a,b) { return parseFloat(a[text]) - parseFloat(b[text]) } );
    }
    else{
        jdata.sort(function(a,b) { 
            return parseFloat(b[text]) - parseFloat(a[text]) } );
    }
    return jdata
}

function filterJSONFrequencies(data){
    
    let minfreq = document.getElementById("minfreq");
    let maxfreq = document.getElementById("maxfreq");
    let result = Object.entries(data).filter(([key, value]) => parseFloat(value.fmin) > parseFloat(minfreq.value)).filter(([key, value]) => parseFloat(value.fmax) < parseFloat(maxfreq.value)).map(([key, value]) => value);
    return result
}

function checkNull(value, keepnull=true, includestring="line", keepstring=false, other=true){
    if (value == null){
        return keepnull
    }
    else if (value.includes(includestring)){
        return keepstring
    }
    else{
        return other
    }
}

function filterHWinjs(data){

    let onlyhwinjbutton = document.getElementById("onlyhwinjs");
    let hidehwinjbutton = document.getElementById("hidehwinjs");
    if (onlyhwinjbutton.checked & hidehwinjbutton.checked == false){
        let result = Object.entries(data).filter(([key, value]) => checkNull(value.info, keepnull=false, includestring="hwinj",keepstring=true,other=false)).map(([key, value]) => value);
        return result
    }
    else if (onlyhwinjbutton.checked == false & hidehwinjbutton.checked){
        let result = Object.entries(data).filter(([key, value]) => checkNull(value.info, keepnull=true, includestring="hwinj",keepstring=false)).map(([key, value]) => value);
        return result
    }
    else if (onlyhwinjbutton.checked & hidehwinjbutton.checked){
        return data
    }
    else{
        return data
    }
    
}


function filterKnownLines(data){

    let onlylinebutton = document.getElementById("onlyknownlines");
    let hidelinebutton = document.getElementById("hideknownlines");
    if (onlylinebutton.checked & hidelinebutton.checked == false){
        let result = Object.entries(data).filter(([key, value]) => checkNull(value.info, keepnull=false, includestring="line",keepstring=true, other=false)).map(([key, value]) => value);
        return result
    }
    else if (onlylinebutton.checked == false & hidelinebutton.checked){
        let result = Object.entries(data).filter(([key, value]) => checkNull(value.info, keepnull=true, includestring="line",keepstring=false)).map(([key, value]) => value);
        return result
    }
    else if (onlylinebutton.checked & hidelinebutton.checked){
        return data
    }
    else{
        return data
    }
    
}

function checkFilters(data){

    data = filterJSONFrequencies(data);
    data = filterHWinjs(data);
    data = filterKnownLines(data);
    data = sortJSONTable(data);
    
    return data
}

function loadTableElements(startIndex, data){
    let itemsPerPage = 10;

    const table = document.getElementById("inTable");
    table.innerHTML = "";
    /*var rowCount = table.rows.length; 
    while(--rowCount) table.deleteRow(rowCount);*/

    let keys = Object.keys(data[0]);
    let head = table.createTHead();
    let hrow = head.insertRow(0);
    for (let i = 0; i < keys.length; i++){
        let hcol = hrow.insertCell(i);
        hcol.innerHTML = keys[i];
    }

    let body = table.createTBody();
    for (let i = 0; i < itemsPerPage; i++) {
        let rdata = data[i + parseInt(startIndex)];
        if (rdata != null){
        let row = body.insertRow(i);
            for (let j = 0; j < keys.length; j++){
                let col = row.insertCell(j);
                col.innerHTML = rdata[keys[j]];
            }
        }
    }

    let ttable = $('#inTable').DataTable({"ordering": false,searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true, sortable:false});
    return ttable;

    

}


function resetTable(event){
    let minfreq = document.getElementById("minfreq");
    let maxfreq = document.getElementById("maxfreq");
    minfreq.value = "0";
    maxfreq.value = "2000"
    reloadPage();

}

function filterDivs(event){
    if (document.getElementById("sortdiv").style.display == "none"){
        document.getElementById('sortdiv').style.display = "";
        document.getElementById('checkboxdiv').style.display = "";
        document.getElementById('filterdiv').style.display = "";
        document.getElementById('filterdivbutton').innerHTML = "Hide filters";
        document.getElementById('filterlinediv').style.display = "";
    }
    else{
        document.getElementById('sortdiv').style.display = 'none';
        document.getElementById('checkboxdiv').style.display = 'none';
        document.getElementById('filterdiv').style.display = 'none';
        document.getElementById('filterdivbutton').innerHTML = "Show filters";
        document.getElementById('filterlinediv').style.display = "none";
    }

}

function makeHeadings(){
    const nbar = document.getElementById("runnav");
    var runname = '<a href="#">O1</a>';
    nbar.appendChild(runname);

}

function makeInfoTable(table, rowdata){

    let headers = table.columns().header().map(d => d.textContent).toArray();
    let infoTable = document.getElementById("infoTable");
    let head = infoTable.createTHead();
    while(infoTable.rows.length > 0) {
        infoTable.deleteRow(0);
      }
    let hrow = head.insertRow(0);
    let nrow = head.insertRow(1);
    let tabInd=0;
    for (let i = 0; i < headers.length; i++){
        if (headers[i] != "plot_path"){
            let hcol = hrow.insertCell(tabInd);
            hcol.innerHTML = headers[i];
            let ncol = nrow.insertCell(tabInd);
            /*var inh = table.rows(row.rowIndex).data()[0][i];*/
            ncol.innerHTML = rowdata[i];
            /*ncol.innerHTML = $(row).find("td").eq(i).text();*/
            tabInd += 1;
        }
    }


}

function get_plot_column(table, key){
    let headers = table.columns().header().map(d => d.textContent).toArray();
    let index = null;
    for (var i =0; i<headers.length; i++){
        if (headers[i] == key){
            index = i;
        }
    }
    return index;
}

function highlightRow(table, row){
    table.removeClass("highlight");
    row.addClass("highlight");
}

function nextImage(event){

    let table = $('#inTable').DataTable({retrieve:true, searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true, sortable:false});
    const image = document.getElementById("image");
    $("#image").css({'opacity': 0.5});
    let value = image.getAttribute("value");
    let newrow = parseInt(value) + 1
    if (newrow > $('#inTable tr').length - 2){
        nextTablePage(event);
        /*window.alert("After end of table");*/
        newrow = 0;
    }
    else{
        let row = table.row(newrow).data();
        let index = get_plot_column(table, "plot_path");
        var newpath = row[index];
        changeImg(newpath);
        image.setAttribute("value",newrow);
        makeInfoTable(table, row);
        $('#inTable tr').removeClass("selected");
        $('#inTable tr').each(function(){
            if (this._DT_RowIndex == newrow){
                $(this).addClass('selected');
            }
        })
    }
    
    $("#image").css({'opacity': 1.0});

}

function previousImage(event){

    let table = $('#inTable').DataTable({retrieve:true, searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true, sortable:false});
    const image = document.getElementById("image");
    let value = image.getAttribute("value");
    let newrow = parseInt(value) - 1
    $("#image").css({'opacity': 0.5});
    if (newrow < 0){
        table = lastTablePage(event);
        /*window.alert("Before start of table");*/
        newrow = 9;
    }
    else{
        let row = table.row(newrow).data();
        let index = get_plot_column(table, "plot_path");
        var newpath = row[index];
        changeImg(newpath);
        image.setAttribute("value",newrow);
        makeInfoTable(table, row);
        $('#inTable tr').removeClass("selected");
        $('#inTable tr').each(function(){
            if (this._DT_RowIndex == newrow){
                $(this).addClass('selected');
            }
        })
    }
    
    
    
    $("#image").css({'opacity': 1.0});
}

function changeImg(newpath) { 
    /* change the image based on the input new path */
    /* If the button for hiding and showing track is clicked the show the appropriate image*/

    const button = document.getElementById("trackbutton");
    if (button.value == 1){
        newpath = newpath.replace("/track_", "/notrack_");
    }
    else{
        newpath = newpath.replace("/notrack_", "/track_")
    }

    $("#plotlink").attr("href",newpath);

    $("#image").attr("src",newpath); 

} 

function changeColumn(event){
    let table = $('#inTable').DataTable({searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true, sortable:false});
    let col_ind = get_plot_column(table, event.target.id);
    /*let column = $('td:nth-child('+(col_ind+1)+')');*/
    let column = table.column(col_ind);
    if (document.getElementById(event.target.id).checked){
        /*column.show();*/
        column.visible(true);
    }
    else{
        /*column.hide();*/
        column.visible(false);
    }
}

function showHideTrack(event){
    const button = document.getElementById("trackbutton");
    var oldpath = $("#image").attr("src");
    if (button.value == 0){
        var newpath = oldpath.replace("/track_", "/notrack_");
        button.value = 1;
        button.innerHTML = "Show track";
    }
    else{
        var newpath = oldpath.replace("/notrack_", "/track_");
        button.value = 0;
        button.innerHTML = "Hide track";
    }

    $("#plotlink").attr("href",newpath);

    $("#image").attr("src",newpath); 

}


function defaultVisible(items){
    let keys = Object.keys(items[0]);
    let table = $('#inTable').DataTable({retrieve:true, searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true, sortable:false});
    for (let i = 0; i < keys.length; i++){
        if (keys[i] == "fmin" | keys[i] == "fmax" | keys[i] == "info" | keys[i] == "lineaware_stat" |keys[i] == "H1_viterbistat"|keys[i] == "L1_viterbistat" | keys[i] == "Viterbi"){
            table.column(i).visible(true);
        }
        else{
            table.column(i).visible(false);
        }
    }
}   

function setVisible(data){
    let keys = Object.keys(data[0]);
    let table = $('#inTable').DataTable({retrieve:true, searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true, sortable:false});
    for (let i = 0; i < keys.length; i++){
        let element = document.getElementById(keys[i]);
        if (element.checked){
            table.column(i).visible(true);
        }
        else{
            table.column(i).visible(false);
        }
    }
}

function makeCheckBoxes(items){
    const checkform = document.getElementById("checkboxform");
    const sortdrop = document.getElementById("sortby");
    let table = $('#inTable').DataTable({searching: false, paging: false, info: false, "bAutoWidth": false,"bDestroy" : true,});

    let keys = Object.keys(items[0]);
    for (let i = 0; i < keys.length; i++){
        var elemcheck = document.createElement('label');
        var elemsort = document.createElement('option');
        elemsort.value = i+1;
        elemsort.innerHTML = keys[i];
        if (keys[i] == "lineaware_stat"){
            elemsort.selected = "selected";
        }
        
        if (keys[i] == "fmin" | keys[i] == "fmax" | keys[i] == "info" | keys[i] == "lineaware_stat" |keys[i] == "H1_viterbistat"| keys[i] == "H1_viterbistat" | keys[i] == "Viterbi"){
            elemcheck.innerHTML = '<input type="checkbox" id="' + keys[i] + '" name="' + keys[i] + '" value="' + (i) + '" checked onclick=changeColumn(event)>' + keys[i];
            /*$('td:nth-child(' + (i+1) + ')').show();*/
            /*table.column(i).visible(true);*/
        }
        else{
            elemcheck.innerHTML = '<input type="checkbox" id="' + keys[i] + '"" name="' + keys[i] + '" value="' + (i) + '" onclick=changeColumn(event)>' + keys[i];
            /*$('td:nth-child(' + (i+1) + ')').hide();*/
            /*table.column(i).visible(false);*/
            
        }
        checkform.appendChild(elemcheck);
        sortdrop.appendChild(elemsort);
        }

    defaultVisible(items);
    /*let stat_index = get_plot_column(table, "lineaware_stat");
    let element = document.getElementById("sortby");
    element.value = stat_index;
    sortTableIndex(0);*/
}

function loadTableData(items) {
    const table = document.getElementById("inTable");

    let keys = Object.keys(items[0]);
    let head = table.createTHead();
    let hrow = head.insertRow(0);
    for (let i = 0; i < keys.length; i++){
        let hcol = hrow.insertCell(i);
        hcol.innerHTML = keys[i];
    }

    let body = table.createTBody();
    for (let i = 0; i < items.length; i++){
        let row = body.insertRow(i);
        for (let j = 0; j < Object.keys(items[i]).length; j++){
        let col = row.insertCell(j);
        col.innerHTML = items[i][keys[j]];
        }
    }
        
    
    $('#inTable').dataTable({  
            "bAutoWidth": false,
        });

    };
