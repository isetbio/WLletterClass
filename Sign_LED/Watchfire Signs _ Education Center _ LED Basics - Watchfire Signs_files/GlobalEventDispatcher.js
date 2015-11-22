/**
 * @fileOverview GlobalEventDispatcher creates a single instance of a EventDispatcher that can be included in, and
 * shared with modules throughout the application.  This makes sure that all objects are using the same instance
 * of an EventDispatcher, while removing the need to have the application file manage it.
 *
 * @author Kevin Moot <kmoot@nerdery.com>
 * @contributor Jeff Maki <jmaki@nerdery.com>
 *
 *
 * @version 1.0
 */
define(function (require) {
    'use strict';

    var EventDispatcher = require('models/EventDispatcher');

    /**
     * Traps an instance of the event dispatcher for us in various components
     * @constructor
     * @static
     */
    var GlobalEventDispatcher = {};

    GlobalEventDispatcher.eventDispatcher = new EventDispatcher();

    /**
     * Returns the local instance of an EventDispatcher
     *
     * @return {EventDispatcher}
     */
    GlobalEventDispatcher.getEventDispatcher = function () {
        return this.eventDispatcher;
    };

    /**
     * Creates a centralized list of Events that can be used with the EventDispatcher.  These events can be triggered
     * and subscribed to with or without using the EVENTS object but should use the EVENTS object except in the case
     * where the specific event is defined in as an data attribute within a DOM element.
     *
     * When adding events, add a brief description for what the event represents.
     *
     * @type {Object}
     */
    GlobalEventDispatcher.EVENTS = {
        CONTEXT_CHANGE: 'ContextChange',    /* the page transitions between breakpoints */
        WINDOW_RESIZE: 'windowResize',      /* the window changes size, could be on orientation change*/
        LARGE_CONTEXT: 'lgScreen',      /* Entered Large Context*/
        SMALL_CONTEXT: 'smScreen'
    };

    return GlobalEventDispatcher;
});


