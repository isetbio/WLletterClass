/**
 * @fileOverview MobileSpecification module definition
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');

    /**
     * Switches between tabs
     *
     * @class MobileSpecification
     * @param {jQuery} $element A reference to the containing DOM element.
     * @constructor
     */
    var MobileSpecification = function($element) {
        if(!$element.length) {
            return this;
        }

        /**
         * A reference to the containing DOM element.
         *
         * @property $element
         * @type {jQuery}
         * @public
         */
        this.$element = $element;


        this.init();
    };

    var proto = MobileSpecification.prototype;

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call, followed by createChildren and layout.
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {SecondaryNavDropdownView}
     * @private
     */
    proto.init = function() {
        this.enable();

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
    proto.enable = function() {
        this.buildMobileSpecs();

        return this;
    };

    proto.buildMobileSpecs = function() {
        var specs = [];

        // Build specs arrays
        $('.specs:first table:first tr:first th').each(function(i){
            specs[i] = [];

            $('.specs:first table:first tr').each(function(j){
                specs[i].push(this.cells[i].innerHTML);
                // console.log(this.cells[i].innerHTML);
            });
        });

        // Verify specs arrays
        // console.log(specs);

        // Log data in the order i need it output
        for(var spec = 1; spec < specs.length; spec++) {
            // console.log(specs[spec]);
            // console.log('specs-'+spec);
            $('#mobileProductSpecs').append('<div class="specs spec-' + spec + '"></div>');

            for(var specAttr = 0; specAttr < specs[spec].length; specAttr++) {
                if(specAttr != 0) {
                    $('#mobileProductSpecs .spec-' + spec + ' dl.specs-list').append('<dt>' + specs[0][specAttr] + '</dt>');
                    $('#mobileProductSpecs .spec-' + spec + ' dl.specs-list').append('<dd>' + specs[spec][specAttr] + '</dd>');
                } else {
                    $('#mobileProductSpecs .spec-' + spec).append('<div class="hdg hdg_5 mix-hdg_sm">' + specs[spec][specAttr] + '</div>');
                    $('#mobileProductSpecs .spec-' + spec).append('<dl class="specs-list"></dl>');
                }
                // console.log(specs[0][specAttr]);
                // console.log(specs[spec][specAttr]);
            }
        }
    };

    return MobileSpecification;
});