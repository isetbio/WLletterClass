/**
 * @fileOverview GlobalResizeListener Creates a single instance of the Resize Listener that can be accessed by other
 * modules as needed.  This is preferable to including mutliple resize listeners, as they would broadcast events multiple
 * times to the same channel causing shinanigans.
 *
 * @author Jeff Maki <jmaki@nerdery.com>
 *
 *
 * @version 1.0
 */
define(function (require) {
    'use strict';

    var ResizeListener = require('models/ResizeListener');

    /**
     * Traps an instance of the resize listener for us to use in other components
     *
     * @constructor
     * @static
     */
    var GlobalEventDispatcher = {};

    GlobalEventDispatcher.resizeListener = new ResizeListener(50);

    /**
     * Returns the local instance of an ResizeListener
     *
     * @return {ResizeListener}
     */
    GlobalEventDispatcher.getResizeListener = function () {
        return this.resizeListener;
    };

    return GlobalEventDispatcher;
});

