/**
 * @fileOverview EventDispatcher module definition
 *
 *
 *
 * @author Jeff Maki <jmaki@nerdery.com>
 * @version 1.0
 */
define(function (require) {
    'use strict';

    var $ = require('jquery');


    /**
     * Acts as a Mediator, provides a central point for application level event notification
     * Defines a channels object, which is used to store the events something could subscribe to,
     * it also sets the base tokenID which provides listeners a way to un-subscribe themselves from
     * specific events.
     *
     * @class EventDispatcher
     * @constructor
     */
    var EventDispatcher = function () {
        /**
         * An object that contains a series of events
         *
         * @property channels
         * @type {Object}
         */
        this.channels = {};

        /**
         * The initial starting token for an event, it is used to allow a subscriber to un-subscribe at a later time
         *
         * @property tokenID
         * @type {Number}
         */
        this.tokenID = 0;
    };

    /**
     * This method is called in order for a function to listen for certain JavaScript triggers.
     *
     *
     * @method subscribe
     * @param channel {string} a subject that is to be subscribed to
     * @param fn {function} a callback function to execute when the event is published
     * @return {String} a string used to identify the subscription later, so the object can un-subscribe
     * on an event by event basis
     */
    EventDispatcher.prototype.subscribe = function (channel, fn) {
        var self = this;
        if (!this.channels[channel]) {
            self.channels[channel] = [];
        }
        var token = (++this.tokenID).toString();
        this.channels[channel].push({ ID: token, context: self, callback: fn });
        return token;
    };

    /**
     * This will trigger an event, iterating through the list of subscribers to that event and executing
     * their callbacks.
     *
     * @method publish
     * @param channel {string} a subject that you wish to trigger
     * @return {EventDispatcher}
     */
    EventDispatcher.prototype.publish = function (channel) {
        var self = this;
        var args;
        if (!this.channels[channel]) {
            return false;
        }

        args = Array.prototype.slice.call(arguments, 1);

        for (var i = 0, l = this.channels[channel].length; i < l; i++) {
            var subscription = self.channels[channel][i];
            subscription.callback.apply(subscription.context, args);
        }

        return this;
    };

    /**
     * This method is called in order for a function to stop looking for certain JavaScript circumstances.
     *
     *
     * @method unSubscribe
     * @param token {string} this is a token that was received by the subscribing object and represents
     * the subscription, so that it can be deleted later.
     * @return {*}
     */
    EventDispatcher.prototype.unSubscribe = function(token) {
        var self = this;

        //loop through the channels object
        for (var m in self.channels) {
            // if channels exists at the current index
            if (self.channels[m]) {
                //loop through the array stored at the current object to see if it has the property ID
                for (var i = 0, j = self.channels[m].length; i < j; i++) {
                    //if the id = token, delete it and return
                    if (self.channels[m][i].ID === token) {
                        self.channels[m].splice(i, 1);
                        return token;
                    }
                }
            }
        }
    };

    return EventDispatcher;
});

