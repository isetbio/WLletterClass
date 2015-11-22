/**
 * @fileOverview TabAccordionView module definition
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');
    var GlobalEventDispatcher = require('GlobalEventDispatcher');
    var GlobalResizeListener = require('GlobalResizeListener');
    var ACTIVE_TAB_CLASS = 'tabs-nav_isActive';
    var OPEN_CONTENT_CLASS = 'tabs-content_isOpen';
    var ACCORDION_HD_ACTIVE_CLASS = 'tabs-navAccordion_isActive'
    /**
     * Switches between tabs
     *
     * @class TabAccordionView
     * @param {jQuery} $element A reference to the containing DOM element.
     * @constructor
     */
    var TabAccordionView = function($element) {
        /**
         * A reference to the containing DOM element.
         *
         * @default null
         * @property $element
         * @type {jQuery}
         * @public
         */
        this.$Tabs = $($element);

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
     * @returns {TabAccordionView}
     * @private
     */
    TabAccordionView.prototype.init = function() {
        this.setupHandlers()
            .createChildren()
            .enable()
            .layout();

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
    TabAccordionView.prototype.setupHandlers = function() {
        // Bind event handlers scope here
        this.onClickTabsHandler = this.onClickTabs.bind(this);
        this.onClickAccordionHandler = this.onClickAccordion.bind(this);
        this.eventDispatcher.subscribe(GlobalEventDispatcher.EVENTS.WINDOW_RESIZE, this.checkScreenSize.bind(this));

        return this;
    };

    /**
     * Create any child objects or references to DOM elements.
     * Should only be run on initialization of the view.
     *
     * @method createChildren
     * @returns {TabAccordionView}
     * @private
     */
    TabAccordionView.prototype.createChildren = function() {
        this.$tabsContent = this.$Tabs.find('.js-tabs-content');
        this.$tabsContentClosed = this.$Tabs.find('.js-tabs-content:not(.tabs-content_isOpen)');
        this.$tabsNav = this.$Tabs.find('.js-tabs-nav');
        this.$tabsNavLink = this.$tabsNav.find('a');
        this.$accordionHd = this.$Tabs.find('.js-tabs-navAccordion');

        return this;
    };

    /**
     * Remove any child objects or references to DOM elements.
     *
     * @method removeChildren
     * @returns {TabAccordionView}
     * @public
     */
    TabAccordionView.prototype.removeChildren = function() {

        return this;
    };

    /**
     * Enables the component.
     * Performs any event binding to handlers.
     * Exits early if it is already enabled.
     *
     * @method enable
     * @returns {TabAccordionView}
     * @public
     */
    TabAccordionView.prototype.enable = function() {
        this.$tabsNavLink.on('click', this.onClickTabsHandler);
        this.$accordionHd.on('click', this.onClickAccordionHandler);

        return this;
    };

    /**
     * Disables the component.
     * Tears down any event binding to handlers.
     * Exits early if it is already disabled.
     *
     * @method disable
     * @returns {TabAccordionView}
     * @public
     */
    TabAccordionView.prototype.disable = function() {

        this.$tabsNavLink.off('click', this.onClickTabsHandler);
        this.$accordionHd.off('click', this.onClickAccordionHandler);

        return this;
    };

    /**
     * Destroys the component.
     * Tears down any events, handlers, elements.
     * Should be called when the object should be left unused.
     *
     * @method destroy
     * @returns {TabAccordionView}
     * @public
     */
    TabAccordionView.prototype.destroy = function() {
        this.disable()
            .removeChildren();

        return this;
    };

    //////////////////////////////////////////////////////////////////////////////////
    // EVENT HANDLERS
    //////////////////////////////////////////////////////////////////////////////////

    /**
     * Handles a tab click on large screens and routes actions
     *
     * @method onClickTabs
     * @private
     */
    TabAccordionView.prototype.onClickTabs = function(e) {
        e.preventDefault();
        var activeTab = $(e.currentTarget);
        this.setActiveTab(activeTab);
        this.getActiveTab(activeTab);
        this.hideTabContent();
        this.showTabContent(activeTab.attr('href'));
    };

    /**
     * Determine which screen size we are in or if the screen sized has changed to determine course of action
     *
     * @method checkScreenSize
     * @private
     */
    TabAccordionView.prototype.checkScreenSize = function() {
        var screenSize = this.globalResizeListener.getCurrentContext();

        //don't change anything if it's within the same context
        if (this.currentScreenSize === screenSize) {
            return this;
        }

        //if we've entered a new screen size we need to hide all the content that shouldn't be shown
        if (this.currentScreenSize !== screenSize) {
            this.layout();
        }

        //make sure if all accordion content was hidden that active content is shown on tabbed interface
        if (screenSize === 'lgScreen') {
            this.showTabContent($('.tabs-nav_isActive').children('a').attr('href'));
        }

        this.currentScreenSize = screenSize;
    };

    //
    TabAccordionView.prototype.layout = function() {
        //don't want to cache this variable as we want to catch changes to which ones are open
        this.$Tabs.find('.js-tabs-content:not(.tabs-content_isOpen)').css({
            'position' : 'static',
            'display'  : 'none'
        });
    };

    /**
     * onClickAccordion handler.
     * Toggles accordion content on click and hides all others
     *
     * @method onClick
     * @param e {Event} JavaScript event object.
     * @private
     */
    TabAccordionView.prototype.onClickAccordion = function(e) {
        e.preventDefault();
        var activeTab = $(e.currentTarget);

        if(activeTab.hasClass('tabs-navAccordion_isActive') === false) {
            //set active accordion head
            this.setActiveAccordion(activeTab);
            this.getActiveAccordion(activeTab);

            this.$tabsContent.removeClass('isActive');
            activeTab.next('.js-tabs-content')
                .css('position', 'static')
                .addClass(OPEN_CONTENT_CLASS + ' isActive')
                .slideToggle();

            this.$Tabs.find('.js-tabs-content:not(.isActive)').slideUp();
        }
    };

    /**
     *Determine which accordion is active and get information for setting active tab so it's in sync when transitioning to larger screens.
     *
     * @method getActiveAccordion
     * @param activeTab {jquery} active tab nav item
     * @private
     */
    TabAccordionView.prototype.getActiveAccordion = function(activeTab) {
        var activeNavItem = activeTab.attr('href');
        var activeAnchor = 'a[href="' + activeNavItem + '"]'
        var activeTab = this.$tabsNav.find(activeAnchor);
        this.setActiveTab(activeTab);
    };

    /**
     *Determine which tab is active so you can set accordion hd properly when transitioning to small screens.
     *
     * @method getActiveTab
     * @param activeTab {jquery} active tab nav item
     * @private
     */
    TabAccordionView.prototype.getActiveTab = function(activeTab) {
        var activeNavItem = activeTab.attr('href');
        var activeAnchor = 'a.js-tabs-navAccordion[href="' + activeNavItem + '"]';
        var activeTab = $(activeAnchor);
        this.setActiveAccordion(activeTab);
    };

    /**
     * Show Tab content
     * shows active tab content
     *
     * @method showTabContent
     * @param tabContent {jquery} show tab content
     * @private
     */
    TabAccordionView.prototype.showTabContent = function(tabContent) {
        this.$tabsContent.removeClass(OPEN_CONTENT_CLASS);
        $(tabContent).css('position' , 'static')
            .show()
            .addClass(OPEN_CONTENT_CLASS);
    };

    /**
     * Hide Tab content
     * Hides all tab content
     *
     * @method hideTabContent
     * @param tabContent {jquery} hide tab content
     * @private
     */
    TabAccordionView.prototype.hideTabContent = function() {
        this.$tabsContent.hide()
            .removeClass(OPEN_CONTENT_CLASS);
    };

    /**
     * Sets active Tab
     *
     * @method setActiveTab
     * @param activeTab {jquery} Active tab nav item
     * @private
     */
    TabAccordionView.prototype.setActiveTab = function(activeTab) {
        this.$tabsNavLink.parent('li')
            .removeClass(ACTIVE_TAB_CLASS);
        activeTab.parent('li')
            .addClass(ACTIVE_TAB_CLASS);
    };

    /**
     * Activates accordion head item that was clicked on by adding class
     *
     * @method setActiveAccordion
     * @param activeAccordionHead {jquery} Active Accordion Head that was clicked on
     * @private
     */
    TabAccordionView.prototype.setActiveAccordion = function(activeAccordionHead) {
        //set active state on accordion head
        $('.js-tabs-navAccordion').removeClass(ACCORDION_HD_ACTIVE_CLASS);
        activeAccordionHead.toggleClass(ACCORDION_HD_ACTIVE_CLASS);
    };

    return TabAccordionView;
});