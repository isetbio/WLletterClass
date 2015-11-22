/**
 * @fileOverview Flyout menu for mobile navigation
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');

    var FlyOutNavView = function($element) {
        /**
         * A reference to the trigger DOM element.
         *
         * @default null
         * @property $element
         * @type {jQuery}
         * @public
         */
        this.$element = $element || null;

        this.init();
    };

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call, followed by createChildren and layout.
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {FlyOutNav}
     * @private
     */
    FlyOutNavView.prototype.init = function() {
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
    FlyOutNavView.prototype.setupHandlers = function() {
        // Bind event handlers scope here
        this.onClickHandler = $.proxy(this.onClick, this);
        this.onResizeHandler = $.proxy(this.onResize, this);

        $(window).resize(this.onResize);

        return this;
    };

    /**
     * Create any child objects or references to DOM elements
     * Should only be run on initialization of the view
     *
     * @method createChildren
     * @chainable
     */
    FlyOutNavView.prototype.createChildren = function() {
        // Create any other dependencies here
        this.$mainNav = $('.mainNav') || null;
        this.$html = $('html');

        this.windowHeight = window.innerHeight || 0;
        this.windowWidth = 0;
        this.headerHeight = $('.header').height() || 0;

        return this;
    };

    /**
     * Performs measurements and applys any positiong style logic
     * Should be run anytime the parent layout changes
     *
     * @method layout
     * @chainable
     */
    FlyOutNavView.prototype.layout = function() {
        // Perform any layout and measurement here

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
    FlyOutNavView.prototype.enable = function() {
        // Setup any event handlers
        this.$element.on('click', this.onClickHandler);
        this.$mainNav.on('resize', this.onResizeHandler);

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
    FlyOutNavView.prototype.disable = function() {
        // Tear down any event handlers
        this.$element.off('click', this.onClickHandler);
        this.$mainNav.off('resize', this.onResizeHandler);

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
    FlyOutNavView.prototype.destroy = function() {
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
    FlyOutNavView.prototype.onClick = function() {
        this.$mainNav
            .toggleClass('mainNav_isActive')
            .css({
                'height' : this.windowHeight
            });

        return this;
    };

    FlyOutNavView.prototype.onResize = function() {
        var headerHeight = $('.header').height();
        var mainNav = $('.mainNav');
        this.windowWidth = $(window).width();
        
        if(this.windowWidth > 800) { // 800 is the breakpoint
            mainNav.css('height', 'auto');
        }

        return this;
    };

    return FlyOutNavView;
});

