/**
 * @fileOverview Handles basic modal behavior.  This is the parent class for modals.
 * It handles open and closing and grabbing content and defers to the child class to handle modal specific
 * behavior if relevant. If there is a child class this gets called from the modal specific view,
 * else it gets called directly.
 *
 * (Update: video child class VideoModalView removed and implmented instead in initializeVideos in this class - jbutts)
 *
 * Modals must have the following attributes and the base modal markup must be present on the bottom of the page:

 * data-js-modal-target="" //where it should pull the main content for the modal from
 * data-js-module="modal'
 *
 * @author Angela Norlen
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');
    var GlobalEventDispatcher = require('GlobalEventDispatcher');
    var GlobalResizeListener = require('GlobalResizeListener');

    /**
     * @class ModalView
     * @param {jQuery} modalTriggerClass A reference to the class used to trigger the modal
     * @param modalView A reference to the name of the modal view it should instantiate
     * @constructor
     */
    var ModalView = function(modalTriggerClass) {
        /**
         * Close element on modal
         * @property this.$modalClose
         * @type {jQuery}
         * @public
         */
        this.$modalClose = null;

        /**
         * Modal page background overlay
         * @property this.$modalOverlay
         * @type {jQuery}
         * @public
         */
        this.$modalOverlay = null;

        /**
         * Modal on the page
         * @property this.$modal
         * @type {jQuery}
         * @public
         */
        this.$modal = null;

        /**
         * Modal class of modal to show
         * @property this.modalClass
         * @type {string}
         * @public
         */
        this.modalClass = null;

        /**
         * Scroll position when modal is opened
         *
         * @default 50
         * @name scrollPositionOffset
         * @type {number}
         */
        this.scrollPositionOffset = 50;

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

        /**
         * Checks if modal class is passed in.
         * This is when a modal doesn't need a child view to control behavior it will call modal directly.
         * Else this gets through child class.

         * @name  this.$modalTrigger
         * @type {string}
         */
        if (typeof modalTriggerClass !== 'undefined') {
            this.$modalTrigger = $(modalTriggerClass);
            this.init();
        }
    };

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {ModalView}
     * @private
     */
    ModalView.prototype.init = function() {
        this.onInit();
        this.setupHandlers()
            .createChildren()
            .enable();

        return this;
    };

    ModalView.prototype.createChildren = function() {
        this.$modalOverlay = $('.js-modal-overlay');
        this.$modal = $('.js-modal');
        this.$modalContent = $('.js-modal-content');
        return this;
    };

    /**
     * Binds the scope of any handler functions.
     * Should only be run on initialization of the view.
     *
     * @method setupHandlers
     * @returns {ModalView}
     * @private
     */
    ModalView.prototype.setupHandlers = function() {
        // Bind event handlers scope here
        this.onClickModalTriggerHandler = this.setupModal.bind(this);
        this.onClickModalCloseHandler = this.closeModal.bind(this);

        this.eventDispatcher.subscribe(GlobalEventDispatcher.EVENTS.WINDOW_RESIZE, this.positionModal.bind(this));

        this.onSetupHandlers();

        return this;
    };


    /**
     * Setup modal with proper content pulled from static html file if referenced in the data-js-modal-target else by grabbing js template in modal specific view
     * @method setupModal
     * @param modalTrigger object
     * @private
     */
    ModalView.prototype.setupModal = function(e) {
        e.preventDefault();
        var modalTrigger = $(e.currentTarget);
//        this.modalClass = modalTrigger.data('modal');
        var modalContentUrl = modalTrigger.attr('href');

        this.openModal();
        var self = this;
        $.ajax({
            url: modalContentUrl,
            type: "GET"
        }).done(function(data) {
                self.loadContent(data);
            });
    };

    /**
     * Binds events
     * @method enable
     * @private
     */
    ModalView.prototype.enable = function() {
        if (this.$modalTrigger === undefined) {
            throw new Error('Modal trigger class needs to be defined.');
        }
        this.$modalTrigger.on('click', this.onClickModalTriggerHandler);

        this.onEnable();
        return this;
    };

    ModalView.prototype.disable = function() {
        this.$modalTrigger.off('click', this.onClickModalTriggerHandler);

        return this;
    };

    /**
     * This runs immediately after modal content is loaded and markup is present on page
     * @param {data} send back the html needed to display in the modal
     */
    ModalView.prototype.loadContent = function(data) {
        this.$modalContent.html(data);

        // All elements have been appended to the page at this point
        this.$modalClose = $('.js-modal-close');
        this.$modalClose.on('click', this.onClickModalCloseHandler);

        this.onContentLoaded();
    };


    /**
     * Shows modal and overlay
     *
     * @method openModal
     * @private
     */
    ModalView.prototype.openModal = function() {
        this.$modal = this.$modal.attr('class' , 'modal js-modal').addClass(this.modalClass);
        this.positionModal();
        this.$modal.show();
        this.$modalOverlay.show();
    };

    /**
     * Hides modal and overlay and empties out container so next one can be injected
     *
     * @method hideModal
     * @private
     */
    ModalView.prototype.closeModal = function(e) {
        //not always called from an event
        if (e !== undefined) {
            e.preventDefault();
        }
        this.$modal.hide();
        this.$modalContent.empty();
        this.$modalOverlay.fadeOut();
    };

    /**
     * Positions modal relative to scroll position
     *
     * @method hideModal
     * @private
     */
    ModalView.prototype.positionModal = function() {
        var modalWidthHalved = $('.js-modal').width() /2;
        var modalTopPosition = ($(window).scrollTop()) + this.scrollPositionOffset;
        this.$modal.css({
            'top': modalTopPosition,
            'margin-left': - modalWidthHalved
        });
    };

    ////////////////////////////////////
    // Child functions to be implemented
    ////////////////////////////////////

    ModalView.prototype.onInit = function() {
        // This function is intended to be implemented by child classes
        // If no implementation is found, this will do nothing!
    };

    ModalView.prototype.onSetupHandlers = function() {
        // This function is intended to be implemented by child classes
        // If no implementation is found, this will do nothing!
    };

    ModalView.prototype.onEnable = function() {
        // This function is intended to be implemented by child classes
        // If no implementation is found, this will do nothing!
    };

    ModalView.prototype.onContentLoaded = function() {
        // This function is intended to be implemented by child classes
        // If no implementation is found, this will do nothing!
    };

    return ModalView;
});