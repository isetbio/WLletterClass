define(function(require, exports, module) { // jshint ignore:line
    'use strict';
    //TODO: these might need to get handled with data attributes

    require('nerdery-function-bind');

    var $ = require('jquery');
    var Modernizr = require('plugins/modernizr.custom');
    var touchSwipe = require('plugins/jquery.touchSwipe.min');
    var TabAccordionView = require('views/TabAccordionView');
    var SecondaryNavDropdownView = require('views/SecondaryNavDropdownView');
    var ModalView = require('views/ModalView');
    var FlyOutNav = require('views/FlyOutNavView');
    var DisplayVideo = require('views/DisplayVideoView');
    var MobileSearch = require('views/MobileSearchView');
    var CarouselView = require('views/CarouselView');
    var MobileSpecifications = require('views/MobileSpecifications');

    /**
     * Initial application setup. Runs once upon every page load.
     *
     * @class App
     * @constructor
     */
    var App = function() {
        this.init();
    };

    var proto = App.prototype;

    /**
     * Initializes the application and kicks off loading of prerequisites.
     *
     * @method init
     * @private
     */
    proto.init = function() {
        // Create your views here

        var tabAccordion = new TabAccordionView($('.js-tabs-accordion'));
        var mobileSpecifications = new MobileSpecifications($('.js-mobileSpecifications'));
        var secondaryDropdownNav = new SecondaryNavDropdownView('.js-secondaryNav');
        var modal = new ModalView('.js-modal-trigger');
        var floutOutNav = new FlyOutNav($('.js-navMenuIcon'));
        var displayVideo = new DisplayVideo($('.video'));
        var mobileSearch = new MobileSearch($('.mobileSearch'));
        var carousel = new CarouselView($('.js-carousel'));
    };


    return App;

});
