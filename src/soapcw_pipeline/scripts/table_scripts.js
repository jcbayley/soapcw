$(document).ready(() => {
    let dataset = [];
    let filteredDataset = [];
    let currentPage = 0;
    let currentRowIndex = 0;


    // Load JSON Data
    $.getJSON("./table.json", function(data) {
        if (data.length === 0) {
            console.error("JSON file is empty.");
            return;
        }
        dataset = [...data];
        filteredDataset = [...dataset];
        columns = Object.keys(dataset[0]).filter(col => col !== "plot_path"); // Exclude 'plot_path' column
        populateSortOptions();
        createTableHeaders();
        loadTable();
        checkURLForSelection();
    }).fail(function() {
        console.error("Error loading JSON file")
    });

    function createTableHeaders() {
        const theadRow = $("#tableHead");
        theadRow.empty();
        columns.forEach(col => {
            theadRow.append(`<th>${col}</th>`);
        });
    }

    function populateSortOptions() {
        const sortColumnDropdown = $("#sortColumn");
        sortColumnDropdown.empty();
        if (dataset.length > 0) {
            Object.keys(dataset[0]).forEach(key => {
                if (key !== "plot_path") { 
                    sortColumnDropdown.append(`<option value="${key}">${key}</option>`);
                }
            });
        }
    }

    function loadTable() {
        const tbody = $("#dataTable tbody");
        tbody.empty();

        $("#pageCounter").text(`Page ${currentPage} of ${Math.ceil(filteredDataset.length / 10) - 1}`);

        const start = currentPage * 10;
        const end = Math.min(start + 10, filteredDataset.length);
        const rows = filteredDataset.slice(start, end);

        $("#tableHeader").empty();
        Object.keys(filteredDataset[0]).forEach(key => {
            if (key !== "plot_path") {
                $("#tableHeader").append(`<th>${key}</th>`);
            }
        });

        rows.forEach((item, index) => {
            let row = `<tr data-index="${start + index}" data-image="${item.plot_path}" data-fmin="${item.fmin}">`;
            Object.entries(item).forEach(([key, value]) => {
                if (key !== "plot_path") {
                    row += `<td>${value}</td>`;
                }
            });
            row += "</tr>";
            tbody.append(row);
        });
    }

    // Handle row click to update the image and URL
    $("#dataTable").on("click", "tr", function () {
        $("#dataTable tr").removeClass("selected");
        $(this).addClass("selected");

        const imageSrc = $(this).data("image");
        $("#displayedImage").attr("src", imageSrc);
        const fminValue = $(this).data("fmin");
        updateURL(fminValue);
    });

    function updateURL(index) {
        const newURL = `${window.location.pathname}?selected_fmin=${index}`;
        window.history.pushState({ path: newURL }, "", newURL);
    }

    function checkURLForSelection() {
        const urlParams = new URLSearchParams(window.location.search);
        const selectedFmin = urlParams.get("selected_fmin");
    
        if (selectedFmin && dataset.length > 0) {
            // Find the index of the matching entry in the full dataset
            const selectedIndex = dataset.findIndex(item => item.fmin == selectedFmin);
    
            if (selectedIndex !== -1) {
                // Calculate the correct page number
                currentPage = Math.floor(selectedIndex / 10);
                loadTable(); // Load the correct page
    
                // Wait for the table to render before triggering the click
                setTimeout(() => {
                    const rowIndex = selectedIndex % 10;
                    const row = $(`#dataTable tbody tr`).eq(rowIndex);
                    if (row.length) {
                        row.trigger("click");
                    }
                }, 500);
            }
        }
    }

    function applyFilters() {
        let minFreq = parseFloat($("#minFreq").val()) || -Infinity;
        let maxFreq = parseFloat($("#maxFreq").val()) || Infinity;
        let onlyHwInjs = $("#onlyhwinjs").is(":checked");
        let hideHwInjs = $("#hidehwinjs").is(":checked");

        let onlyknownlines = $("#onlyknownlines").is(":checked");
        let hideknownlines = $("#hideknownlines").is(":checked");

    
        // Apply filtering to create a new filtered dataset
        filteredDataset = dataset.filter((item) => {
            let freq = parseFloat(item.fmin);
            let info = item.info.toLowerCase(); // Ensure case-insensitive match

            // Frequency range filter
            let withinFreqRange = freq >= minFreq && freq < maxFreq;

            // HW Injection filters
            let containsHwInj = info.includes("hwinj");
            let passesHwFilter = true;
            if (onlyHwInjs) {
                passesHwFilter = containsHwInj;
            } 
            if (hideHwInjs){
                passesHwFilter = !containsHwInj
            }  

            // line filters
            let containsline = info.includes("line");
            let passeslineFilter = true;
            if (onlyknownlines) {
                passeslineFilter = containsline;
            }
            if (hideknownlines) {
                passeslineFilter = !containsline;
            }

            return withinFreqRange && passesHwFilter && passeslineFilter;
        });

        // Reload the table using the filtered dataset
        loadTable();
    }


    $("#applySort").on("click", function () {
        const column = $("#sortColumn").val();
        const order = $("#sortOrder").val(); // Get sorting order
    
        if (column) {
            filteredDataset.sort((a, b) => {
                let valA = a[column];
                let valB = b[column];
    
                // Handle numeric and string sorting
                if (!isNaN(valA) && !isNaN(valB)) {
                    return order === "asc" ? valA - valB : valB - valA;
                } else {
                    return order === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
                }
            });
    
            currentPage = 0; // Reset to first page after sorting
            loadTable();
        }
    });

    // apply filters
    $("#onlyhwinjs, #hidehwinjs, #onlyknownlines, #hideknownlines, #applyFilter").on("change click", applyFilters);


    // Preset Frequency Filters
    $(".preset-filter").on("click", function () {
        $("#minFreq").val($(this).data("min"));
        $("#maxFreq").val($(this).data("max"));
        $("#applyFilter").trigger("click");
    });

    // Reset Filter
    $("#resetFilter").on("click", function () {
        filteredDataset = [...dataset];
        loadTable();
    });

    // Pagination
    $("#nextPage").on("click", function () {
        if ((currentPage + 1) * 10 < filteredDataset.length) {
            currentPage++;
            loadTable();
        }
    });

    $("#previousPage").on("click", function () {
        if (currentPage > 0) {
            currentPage--;
            loadTable();
        }
    });

    $("#cycleNext").on("click", function () {
        const rows = $("#dataTable tbody tr");
        if (rows.length === 0) return;

        $("#dataTable tr").removeClass("selected");

        currentRowIndex++;
        if (currentRowIndex >= rows.length) {
            currentRowIndex = rows.length; // Reset to first row if at the end
        }

        $(rows[currentRowIndex]).trigger("click");

    });


    $("#cyclePrevious").on("click", function () {
        const rows = $("#dataTable tbody tr");
        if (rows.length === 0) return;

        $("#dataTable tr").removeClass("selected");

        currentRowIndex--;
        if (currentRowIndex <= 0) {
            currentRowIndex = 0; // Reset to first row if at the end
        }

        $(rows[currentRowIndex]).trigger("click");
    });

    $("#trackbutton").on("click", showHideTrack);

    function showHideTrack(){
        const button = document.getElementById("trackbutton");
        var oldpath = $("#displayedImage").attr("src");
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
    
        $("#displayedImage").attr("src",newpath); 
    
    }
});