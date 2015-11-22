/**
 * @fileOverview ResizeListener module definition
 *
 * @author Jeff Maki <jmaki@nerdery.com>
 * @contributor Mark Spooner <mspooner@nerdery.com>
 *
 *
 * @version 1.0
 *
 */
define(function (require) {
    'use strict';

    var $ = require('jquery');
    var _ = require('underscore');
    var GlobalEventDispatcher = require('GlobalEventDispatcher');

    /**
     * This is a simple ResizeListener that fires off a global 'resize' event when the window is resized, it uses
     * underscores 'debounce' function in order to keep the event from triggering too often. In addition it tracks the
     * current 'Context' state, which can be accessed via its getCurrentContext function.  In the case that the resize event
     * results in the application changing context, the ResizeListener also publishes the new context that the application
     * is in leveraging the GlobalEventDispatcher.
     *
     * @class ResizeListener
     * @param time {int} Sets how often the resize event will be tracked, by default it is 1/10th of a second
     * @constructor
     */
    var ResizeListener = function (time) {

        /**
         * The global instance of an event dispatcher, used to subscribe to or publish events of global interest
         *
         * @default GlobalEventDispatcher.getEventDispatcher()
         * @property eventDispatcher
         * @type {jQuery}
         */
        this.eventDispatcher = GlobalEventDispatcher.getEventDispatcher();

        /**
         * The delay between resize events being published during a window resize.
         *
         * @default 100
         * @property timeDuration
         * @type {int}
         */
        this.timeDuration = time || 50;

        /**
         * A flag that checks to see if the ResizeListener is currently enabled, determines if it should publish
         * events or not.
         *
         * @default false
         * @property isEnabled
         * @type {Boolean}
         */
        this.isEnabled = false;

        /**
         * The current width of the browser window, used to check the context that the browser is currently in and
         * to see if that context needs to be updated after a resize event.
         *
         * @default 0
         * @property currentWidth
         * @type {Number}
         */
        this.currentWidth = 0;

        /**
         * The current context of the application,
         * values ['smallContext', 'mediumContext', 'largeContext', luxuryContext' ]
         *
         * @default largeContext
         * @property currentContext
         * @type {String}
         */
        this.currentContext = 'largeContext';

        this.init();
    };

    /**
     * initializes the ResizeListener
     *
     * @method init
     * @returns {ResizeListener}
     */
    ResizeListener.prototype.init = function () {

//        if (Modernizr.mq('only all') === false || navigator.userAgent.indexOf("Trident/5")>-1) {
//            return false;
//        }

        this.setupListener()
            .checkSize()
            .enableListener();
    };

    /**
     * Attaches an event listener to the window, uses debounce to ensure that the event it not sent to the app more
     * often then the rimeDuration variable that is passed in when the ResizeListener is instantiated
     *
     * @method setupListener
     * @returns {ResizeListener}
     */
    ResizeListener.prototype.setupListener = function () {
        //$.proxy(this.listener, this);
        
        $(window).on('resize', _.debounce(this.publishEvent, this.timeDuration).bind(this));
        return this;
    };

    /*ResizeListener.prototype.listener = function () {
        $(window).on('resize', _.debounce(this.publishEvent, this.timeDuration));

        return this;
    };*/

    /**
     * Used to disable the publishing of events if needed.
     *
     * @method disableListener
     * @returns {ResizeListener}
     */
    ResizeListener.prototype.disableListener = function () {
        this.isEnabled = false;
        return this;
    };

    /**
     * Used to enable the publishing of events if needed.
     *
     * @method enableListener
     * @returns {ResizeListener}
     */
    ResizeListener.prototype.enableListener = function() {
        this.isEnabled = true;
        return this;
    };

    /**
     * This grabs the pseudo-element inserted into the DOM via CSS in the breakpoints
     * stylesheet. It will then, on resize, compare the returned string. If the string is the
     * same as the saved context, nothing happens. Otherwise a new event is published. Normalization
     * requires an or statement in the code to compare returned strings from all browsers except
     * IE and then IE browsers.
     *
     * @method checkSize
     * @returns {ResizeListener}
     */
    ResizeListener.prototype.checkSize = function() {
        this.currentWidthLabel = window.getComputedStyle(document.body, ':before').content;

        var newContext;

        // Here we have to check the string versus the passed in psuedo-element. Then we set the newContext
        // variable to the event. The reason for the or is the way that IE passes in the string, the string
        // is passed in with quotes.
        this.currentWidthLabel = this.currentWidthLabel.replace(/[\",\']/g, '');
        if (this.currentWidthLabel === 'smScreen' || this.currentWidthLabel === '"smScreen"') {
            newContext = 'smScreen';
        }  else if (this.currentWidthLabel === 'lgScreen' || this.currentWidthLabel === '"lgScreen"') {
            newContext = 'lgScreen';
        }

        if (newContext === this.currentContext) {
            return this;
        } else {
            this.currentContext = newContext;
            this.eventDispatcher.publish(GlobalEventDispatcher.EVENTS.CONTEXT_CHANGE);
            this.eventDispatcher.publish(this.currentContext);
        }

        return this;

    };

    /**
     * If the resizeListener is enabled, this publishes the resize event to the Applications event dispatcher,
     * any objects subscribed to this event will execute their callbacks.
     *
     * @method publishedEvent
     */
    ResizeListener.prototype.publishEvent = function () {
        this.checkSize();
        if (!this.eventDispatcher) {
            throw new Error('Event dispatcher not defined');
        }

        if (this.isEnabled === true) {
            this.eventDispatcher.publish(GlobalEventDispatcher.EVENTS.WINDOW_RESIZE);
        }
    };

    /**
     * Provides a way for other objects to access the current context of an application, without a resize and
     * context change event having to occur.
     *
     * @method getCurrentContext
     * @return {String}
     */
    ResizeListener.prototype.getCurrentContext = function () {
        return this.currentContext;
    };

    /**
     * Exposes the ability for the application to manually trigger the publish current context event
     *
     * @method triggerCurrentContext
     */
    ResizeListener.prototype.triggerCurrentContext = function () {
        this.eventDispatcher.publish(this.currentContext);
    };

    return ResizeListener;
});


