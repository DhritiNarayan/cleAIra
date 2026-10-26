function showSection(id) {
    document.querySelectorAll('.section').forEach(function (section) {
        section.classList.remove('active');
    });

    const section = document.getElementById(id);

    if (section) {
        section.classList.add('active');
    }
}


document.addEventListener('DOMContentLoaded', function () {

    /*
     * =========================================================
     * THREE-DOT HISTORY MENU
     * =========================================================
     */

    const menuToggles = document.querySelectorAll('.menu-toggle');

    menuToggles.forEach(function (toggle) {

        toggle.addEventListener('click', function (event) {

            event.preventDefault();
            event.stopPropagation();

            const menuWrap = toggle.closest('.menu-wrap');

            if (!menuWrap) {
                return;
            }

            const alreadyOpen = menuWrap.classList.contains('open');

            // Close all menus first
            document.querySelectorAll('.menu-wrap').forEach(function (wrap) {

                wrap.classList.remove('open');

                const button = wrap.querySelector('.menu-toggle');

                if (button) {
                    button.setAttribute('aria-expanded', 'false');
                }

            });

            // Open the clicked menu
            if (!alreadyOpen) {

                menuWrap.classList.add('open');

                toggle.setAttribute('aria-expanded', 'true');

            }

        });

    });


    /*
     * Close the three-dot menu when clicking outside it
     */

    document.addEventListener('click', function (event) {

        if (event.target.closest('.menu-wrap')) {
            return;
        }

        document.querySelectorAll('.menu-wrap').forEach(function (wrap) {

            wrap.classList.remove('open');

            const button = wrap.querySelector('.menu-toggle');

            if (button) {
                button.setAttribute('aria-expanded', 'false');
            }

        });

    });


    /*
     * =========================================================
     * SELECT MULTIPLE MODE
     * =========================================================
     */

    const selectMultipleButtons =
        document.querySelectorAll('[data-select-multiple]');

    const checkboxes =
        document.querySelectorAll('.history-checkbox');

    const selectedCount =
        document.getElementById('selected-count');

    const deleteSelectedButton =
        document.getElementById('delete-selected-button');

    const cancelSelectButton =
        document.getElementById('cancel-select-mode');


    /*
     * Get all currently selected history entries
     */

    function getSelectedCheckboxes() {

        return Array.from(
            document.querySelectorAll('.history-checkbox:checked')
        );

    }


    /*
     * Update "Selected: X"
     */

    function updateSelectedCount() {

        if (!selectedCount) {
            return;
        }

        const selected = getSelectedCheckboxes();

        selectedCount.textContent = selected.length;


        if (deleteSelectedButton) {

            deleteSelectedButton.disabled =
                selected.length === 0;

        }

    }


    /*
     * Close every three-dot menu
     */

    function closeAllMenus() {

        document.querySelectorAll('.menu-wrap').forEach(function (wrap) {

            wrap.classList.remove('open');

            const button = wrap.querySelector('.menu-toggle');

            if (button) {
                button.setAttribute('aria-expanded', 'false');
            }

        });

    }


    /*
     * Enter "Select multiple" mode
     */

    function enterMultipleSelectMode() {

        document.body.classList.add('multi-select-mode');

        closeAllMenus();

        updateSelectedCount();

    }


    /*
     * Exit "Select multiple" mode
     */

    function exitMultipleSelectMode() {

        document.body.classList.remove('multi-select-mode');

        document.querySelectorAll('.history-checkbox').forEach(function (checkbox) {

            checkbox.checked = false;

        });

        updateSelectedCount();

    }


    /*
     * "Select multiple" button
     */

    selectMultipleButtons.forEach(function (button) {

        button.addEventListener('click', function (event) {

            event.preventDefault();
            event.stopPropagation();

            enterMultipleSelectMode();

        });

    });


    /*
     * Checkbox changes
     */

    checkboxes.forEach(function (checkbox) {

        checkbox.addEventListener('change', function () {

            updateSelectedCount();

        });

    });


    /*
     * Cancel multiple selection
     */

    if (cancelSelectButton) {

        cancelSelectButton.addEventListener('click', function () {

            exitMultipleSelectMode();

        });

    }


    /*
     * =========================================================
     * DELETE SELECTED HISTORY ENTRIES
     * =========================================================
     */

    if (deleteSelectedButton) {

        deleteSelectedButton.addEventListener('click', async function () {

            const selected = getSelectedCheckboxes();


            // Nothing selected
            if (selected.length === 0) {

                return;

            }


            // Confirmation
            const confirmed = confirm(
                'Delete ' +
                selected.length +
                ' selected review ' +
                (selected.length === 1 ? 'entry' : 'entries') +
                '?'
            );


            if (!confirmed) {

                return;

            }


            deleteSelectedButton.disabled = true;

            deleteSelectedButton.textContent = 'Deleting...';


            try {

                /*
                 * Delete each selected history entry
                 *
                 * Uses the existing Flask route:
                 *
                 * /history/<review_id>/delete
                 */

                for (const checkbox of selected) {

                    const reviewId = checkbox.value;

                    await fetch(
                        '/history/' +
                        encodeURIComponent(reviewId) +
                        '/delete',
                        {
                            method: 'POST'
                        }
                    );

                }


                /*
                 * Reload the history page after deletion
                 */

                window.location.reload();

            } catch (error) {

                console.error(
                    'Error deleting selected reviews:',
                    error
                );

                alert(
                    'Something went wrong while deleting the selected reviews.'
                );

                deleteSelectedButton.disabled = false;

                deleteSelectedButton.textContent =
                    'Delete selected';

            }

        });

    }


    /*
     * Initial count
     */

    updateSelectedCount();

});