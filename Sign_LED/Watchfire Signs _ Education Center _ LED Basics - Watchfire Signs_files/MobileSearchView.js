/**
 * @fileOverview Display search input for mobile
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');

    var MobileSearchView = function($element) {
        this.$element = $element;

        this.init();
    };

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call, followed by createChildren and layout.
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {DisplayVideo}
     * @private
     */
    MobileSearchView.prototype.init = function() {
        this.setupHandlers()
            .createChildren()
            .enable();

        return this;
    };

    /**
     * Binds the scope of any handler functions.
     * Should only be run on initialization of the view.
     *
     * @method setupHandlers
     * @returns {TabAccordionView}
     * @private
     */
    MobileSearchView.prototype.setupHandlers = function() {
        // Bind event handlers scope here
        this.onClickHandler = $.proxy(this.onClick, this);

        return this;
    };

    /**
     * Create any child objects or references to DOM elements
     * Should only be run on initialization of the view
     *
     * @method createChildren
     * @chainable
     */
    MobileSearchView.prototype.createChildren = function() {
        // Create any other dependencies here
        this.$mobileSearchInput = this.$element.find('.js-mobileSearch-input');
        this.$mobileSearchTrigger = this.$element.find('.js-mobileSearch-trigger');

        return this;
    };

    /**
     * Enables the view
     * Performs any event binding to handlers
     * Exits early if it is already enabled
     *
     * @method enable
     * @chainable
     */
    MobileSearchView.prototype.enable = function() {
        // Setup any event handlers
        this.$mobileSearchTrigger.on('click', this.onClickHandler);

        return this;
    };

    /**
     * Disables the view
     * Tears down any event binding to handlers
     * Exits early if it is already disabled
     *
     * @method disable
     * @chainable
     */
    MobileSearchView.prototype.disable = function() {
        // Tear down any event handlers
        this.$mobileSearchTrigger.off('click', this.onClickHandler);

        return this;
    };

    /**
     * Destroys the view
     * Tears down any events, handlers, elements
     * Should be called when the object should be left unused
     *
     * @method destroy
     * @chainable
     */
    MobileSearchView.prototype.destroy = function() {
        this.disable();

        for (var key in this) {
            if (this.hasOwnProperty(key)) {
                this[key] = null;
            }
        }
        
        return this;
    };

    /**
     * onClick Handler
     *
     * @method onClick
     * @param {MouseEvent} event Click event
     */
    MobileSearchView.prototype.onClick = function() {
        this.$mobileSearchInput.toggleClass('mobileSearch-input_isActive');

        return this;
    };

    return MobileSearchView;
});