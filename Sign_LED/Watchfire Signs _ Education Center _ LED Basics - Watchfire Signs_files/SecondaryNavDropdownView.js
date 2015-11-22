/**
 * @fileOverview SecondaryNavDropdownView module definition
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');
    var GlobalEventDispatcher = require('GlobalEventDispatcher');
    var GlobalResizeListener = require('GlobalResizeListener');
    var NAV_OPEN_CLASS = 'secondaryNav-hd_isOpen';
    /**
     * Switches between tabs
     *
     * @class SecondaryNavDropdownView
     * @param {jQuery} $element A reference to the containing DOM element.
     * @constructor
     */
    var SecondaryNavDropdownView = function($element) {
        /**
         * A reference to the containing DOM element.
         *
         * @property $element
         * @type {jQuery}
         * @public
         */
        this.$secondaryNav = $($element);

        /**
         * Default screen size not set until it runs the method. Need to track this on a global level to compare to current context and only run actions if it changes
         *
         * @default null
         * @type {string}
         * @public
         */
        this.currentScreenSize = null;

        /**
         * A reference to the global event dispatcher, provided by the application.
         *
         * @property eventDispatcher
         * @type {}
         * @private
         */
        this.eventDispatcher = GlobalEventDispatcher.getEventDispatcher(); // event dispatcher that will call the event, provided by the application


        /**
         * A reference to the global resize listener.
         *
         * @property getResizeListener
         * @type {}
         * @private
         */
        this.globalResizeListener = GlobalResizeListener.getResizeListener();

        this.init();
    };

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call, followed by createChildren and layout.
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {SecondaryNavDropdownView}
     * @private
     */
    SecondaryNavDropdownView.prototype.init = function() {
        this.setupHandlers()
            .createChildren()
            .checkScreenSize();

        return this;
    };

    /**
     * Binds the scope of any handler functions.
     * Should only be run on initialization of the view.
     *
     * @method setupHandlers
     * @returns {SecondaryNavDropdownView}
     * @private
     */
    SecondaryNavDropdownView.prototype.setupHandlers = function() {
        // Bind event handlers scope here
        this.onClickDropdownHandler = this.toggleDropdown.bind(this);
        this.eventDispatcher.subscribe(GlobalEventDispatcher.EVENTS.WINDOW_RESIZE, this.checkScreenSize.bind(this));

        return this;
    };

    /**
     * Create any child objects or references to DOM elements.
     * Should only be run on initialization of the view.
     *
     * @method createChildren
     * @returns {SecondaryNavDropdownView}
     * @private
     */
    SecondaryNavDropdownView.prototype.createChildren = function() {
        this.$secondaryNavTrigger = this.$secondaryNav.find('.js-secondaryNav-hd');
        this.$secondaryNavOptions = this.$secondaryNav.find('.js-secondaryNav-list');

        return this;
    };

    SecondaryNavDropdownView.prototype.layout = function() {
        this.$secondaryNavOptions.hide();
        return this;
    };

    SecondaryNavDropdownView.prototype.undoLayout = function() {
        this.$secondaryNavOptions.show();
        return this;
    };

    /**
     * Remove any child objects or references to DOM elements.
     *
     * @method removeChildren
     * @returns {SecondaryNavDropdownView}
     * @public
     */
    SecondaryNavDropdownView.prototype.removeChildren = function() {

        return this;
    };

    /**
     * Enables the component.
     * Performs any event binding to handlers.
     * Exits early if it is already enabled.
     *
     * @method enable
     * @returns {SecondaryNavDropdownView}
     * @public
     */
    SecondaryNavDropdownView.prototype.enable = function() {
        this.$secondaryNavTrigger.on('click', this.onClickDropdownHandler);

        return this;
    };

    /**
     * Disables the component.
     * Tears down any event binding to handlers.
     * Exits early if it is already disabled.
     *
     * @method disable
     * @returns {SecondaryNavDropdownView}
     * @public
     */
    SecondaryNavDropdownView.prototype.disable = function() {

        this.$secondaryNavTrigger.off('click', this.onClickDropdownHandler);

        return this;
    };

    /**
     * Destroys the component.
     * Tears down any events, handlers, elements.
     * Should be called when the object should be left unused.
     *
     * @method destroy
     * @returns {SecondaryNavDropdownView}
     * @public
     */
    SecondaryNavDropdownView.prototype.destroy = function() {
        this.disable()
            .removeChildren();

        return this;
    };

    //////////////////////////////////////////////////////////////////////////////////
    // EVENT HANDLERS
    //////////////////////////////////////////////////////////////////////////////////

    SecondaryNavDropdownView.prototype.toggleDropdown = function(e) {
        e.preventDefault();
        $(e.currentTarget).toggleClass(NAV_OPEN_CLASS);
        this.$secondaryNavOptions.toggle();
    };

    SecondaryNavDropdownView.prototype.checkScreenSize = function() {
        var screenSize = this.globalResizeListener.getCurrentContext();

        //don't change anything if it's within the same context
        if (this.currentScreenSize === screenSize) {
            return this;
        }

        //make sure if all accordion content was hidden that active content is shown on tabbed interface
        if (screenSize === 'lgScreen') {
            this.disable();
            this.undoLayout();
        } else {
            this.enable();
            this.layout();
        }

        this.currentScreenSize = screenSize;
    };


    return SecondaryNavDropdownView;
});