/**
 * @fileOverview Display video or image, depending on screen size.
 */

define(function(require) {
    'use strict';

    var _ = require('underscore');
    var $ = require('jquery');
    var GlobalEventDispatcher = require('GlobalEventDispatcher');
    var GlobalResizeListener = require('GlobalResizeListener');

    var DisplayVideoView = function($element) {

        /**
         * Default screen size not set until it runs the method. Need to track this on a global level to compare to current context and only run actions if it changes
         *
         * @default null
         * @property  this.$selectListOptions
         * @type {string}
         * @public
         */
        this.currentScreenSize = null;

        /**
         * The delay between resize events being published during a window resize.
         *
         * @default 50
         * @property timeDuration
         * @type {int}
         */
        this.timeDuration = 50;

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

        this.$element = $element;

        this.init();
    };

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call, followed by createChildren and layout.
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {DisplayVideoView}
     * @private
     */
    DisplayVideoView.prototype.init = function() {
        this.setupHandlers()
            .createChildren()
            .enable()
            .checkScreenSize();

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
    DisplayVideoView.prototype.setupHandlers = function() {
        // Bind event handlers scope here
        this.eventDispatcher.subscribe(GlobalEventDispatcher.EVENTS.WINDOW_RESIZE, this.checkScreenSize.bind(this));
        
        $(window).on('resize', _.debounce(this.videoOffsetX, this.timeDuration).bind(this));

        return this;
    };

    /**
     * Create any child objects or references to DOM elements
     * Should only be run on initialization of the view
     *
     * @method createChildren
     * @chainable
     */
    DisplayVideoView.prototype.createChildren = function() {
        // Create any other dependencies here
        this.$videoContainer = this.$element.find('.js-video-video') || null;
        this.$videoSource = this.$element.find('.js-video-source') || null;
        this.$videoFallback = this.$element.find('.js-video-fallback') || null;

        return this;
    };

    /**
     * Performs measurements and applys any positiong style logic
     * Should be run anytime the parent layout changes
     *
     * @method layout
     * @chainable
     */
    DisplayVideoView.prototype.layout = function() {
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
    DisplayVideoView.prototype.enable = function() {
        // Setup any event handlers

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
    DisplayVideoView.prototype.disable = function() {
        // Tear down any event handlers

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
    DisplayVideoView.prototype.destroy = function() {
        this.disable();

        for (var key in this) {
            if (this.hasOwnProperty(key)) {
                this[key] = null;
            }
        }
        
        return this;
    };

    /**
     * Determine which screen size we are in or if the screen sized has changed to determine course of action
     *
     * @method checkScreenSize
     * @chainable
     */
    DisplayVideoView.prototype.checkScreenSize = function() {
        var screenSize = this.globalResizeListener.getCurrentContext();
        var numOfSources = this.$videoSource.length;
        
        if(this.$videoSource.length > 0) {
            var srcLength = this.$videoSource.attr('src').length;
        }

        if(screenSize === 'lgScreen') {
            for(var i = 0; i < numOfSources; i++) {
                var path = this.$videoSource.eq(i).data('src');
                if(srcLength < 2) {
                    this.$videoSource.eq(i).attr('src', path);
                    this.$videoContainer.load();
                }
            }
            this.$videoFallback.css('display', 'none');
        } else if (screenSize === 'smScreen') {
            this.$videoFallback.css('display', 'block');
            this.$videoSource.attr('src', '');
        }

        this.videoOffsetX();
        return this;
    };

    /**
     * Determine which screen size we are in or if the screen sized has changed to determine course of action
     *
     * @method videoOffSet
     * @chainable
     */
    DisplayVideoView.prototype.videoOffsetX = function() {
        var windowWidth = $(window).width();
        var videoWidth = 1920; /* Client would like the video to have a 300px height. To 
                                    acheive that height, the video needs to be 1920
                                    because of the aspect ratio.
                                */
        var offSet = -(videoWidth - windowWidth) / 2;

        if (windowWidth < 1920) {
            this.$videoContainer.css('margin-left', offSet);
        } else if (windowWidth > 1920) {
            this.$videoContainer.css('margin-left', '0');
        }
        

        return this;
    };

    return DisplayVideoView;
});

