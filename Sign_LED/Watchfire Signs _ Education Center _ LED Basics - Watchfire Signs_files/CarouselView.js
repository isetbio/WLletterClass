/**
 * @fileOverview CarouselView handles the creation of a responsive, touch-swipeable
 * carousel. It uses the jquery touchSwipe plugin and modernizr as dependencies.
 *
 * @author Mark Spooner
 */

define(function(require) {
    'use strict';

    var $ = require('jquery');

    var PROFILES = {

        /**
         * Mobile safari interaction events (touch)
         *
         * @property MOBILE
         * @static
         * @final
         * @type {string}
         */
        MOBILE: 'touch',

        /**
         * Desktop mouse interaction events (mouse)
         *
         * @property MOBILE_IE
         * @static
         * @final
         * @type {string}
         */
        DESKTOP: 'mouse'
    };

     var CarouselView = function ($element) {

        /**
         * Interaction profile that the current features support
         *
         * @property interactionProfile
         * @type {string}
         * @default PROFILES.DESKTOP
         */
        this.interactionProfile = PROFILES.DESKTOP;

        this.executeTests();

        /**
         * Static profiles reference to the class-level profiles
         * @see PROFILES
         *
         * @property PROFILES
         * @type {object}
         */
        this.PROFILES = PROFILES;

        /**
         * A reference to the containing DOM element.
         *
         * @default null
         * @property $element
         * @type {jQuery}
         * @public
         */
        this.$element = $element;

        this.autoRotateInterval = null;

        this.jsControlClass = 'js-carousel-controls_';

        /**
         * Tracks whether component is enabled.
         *
         * @default false
         * @property isEnabled
         * @type {bool}
         * @public
         */
        this.isEnabled = false;

        this.slideWidth = null;

        this.currentImg = 0;

        this.maxImages = null;

        this.speed = 500;

        this.init();
    };

    /**
     * Initializes the UI Component View.
     * Runs a single setupHandlers call, followed by createChildren and layout.
     * Exits early if it is already initialized.
     *
     * @method init
     * @returns {CarouselView}
     * @private
     */
    CarouselView.prototype.init = function () {
        // Check for element and if not present, don't run carousel
        if(!this.$element.length) {
            return;
        }

        this.setOptions()
            .setupHandlers()
            .createChildren()
            .layout()
            .enable();

        return this;
    };

    /**
     *
     * This takes in data options from the markup and overwrites base values. If there is no value in the data
     * options tag then defaults are shown below.
     *
     * @method setOptions
     * @returns {CarouselView}
     * @private
     */

    CarouselView.prototype.setOptions = function() {
        var options = this.$element.data().options;

        var defaults = {
            scrollControlsStylingClass: 'carousel-controls-button',
            isInTab: false,
            autoRotation: false
        };
        this.settings = $.extend({}, defaults, options);

        return this;
    };

    /**
     * Binds the scope of any handler functions.
     * Should only be run on initialization of the view.
     *
     * @method setupHandlers
     * @returns {CarouselView}
     * @private
     */
    CarouselView.prototype.setupHandlers = function () {
        // Bind event handlers scope here
        // this.onClickHandler = this.onClick.bind(this);
        this.onClickNextHandler = this.onClickNext.bind(this);
        this.onClickPrevHandler = this.onClickPrev.bind(this);
        this.onClickIndicatorHandler = this.onClickIndicator.bind(this);
        this.resizeCarouselHandler = this.resizeCarousel.bind(this);
        this.onCarouselHoverHandler = this.onCarouselHover.bind(this);
        this.offCarouselHoverHandler = this.offCarouselHover.bind(this);


        return this;
    };

    /**
     * Create any child objects or references to DOM elements.
     * Should only be run on initialization of the view.
     *
     * @method createChildren
     * @returns {CarouselView}
     * @private
     */
    CarouselView.prototype.createChildren = function () {
        this.$carouselList = this.$element.find('.js-carousel-list');
        this.$carouselRight = null;
        this.$carouselLeft = null;
        this.$carouselControlList = this.$element.find('.js-carousel-controls-list');
        this.$carouselControlIndicator = this.$element.find('.js-carousel-indicator');
        this.$carouselControlIndicatorList = this.$carouselControlIndicator.find('.carousel-indicator-list');
        this.$carouselControlIndicatorLink = null;

        return this;
    };

    /**
     * Remove any child objects or references to DOM elements.
     *
     * @method removeChildren
     * @returns {CarouselView}
     * @public
     */
    CarouselView.prototype.removeChildren = function () {
        this.$carouselList = null;

        return this;
    };

    /**
     * Performs measurements and applys any positioning style logic.
     * Should be run anytime the parent layout changes.
     *
     * @method layout
     * @returns {CarouselView}
     * @public
     */
    CarouselView.prototype.layout = function () {
        this.layoutCarousel();

        if (this.settings.autoRotation) {
            this.autoRotateCarousel();
        }

        if (this.interactionProfile == PROFILES.DESKTOP && this.$carouselList.find('li').length > 1) {
            this.buildScrollControls();
        }

        return this;
    };

    /**
     * showHiddenSlides
     * All slides except the first slide are hidden by default to prevent poor visuals during page load.
     * This makes the slides visible as required by the carousel after the page has loaded.
     */
    CarouselView.prototype.showHiddenSlides = function () {
        this.$carouselList.find('li.hidden').removeClass('hidden');
    }


    /**
     * Enables the component.
     * Performs any event binding to handlers.
     * Exits early if it is already enabled.
     *
     * @method enable
     * @returns {CarouselView}
     * @public
     */
    CarouselView.prototype.enable = function () {
        if (this.isEnabled) {
            return this;
        }

        this.isEnabled = true;
        // Check this.interactionProfile to see if it's desktop or mobile
        if (this.interactionProfile == PROFILES.DESKTOP) {

            if (this.$carouselList.find('li').length > 1) {
                // Enable the carousel arrow controls
                this.$carouselRight.on('click', this.onClickNextHandler);
                this.$carouselLeft.on('click', this.onClickPrevHandler);
                this.$carouselControlIndicatorLink.on('click', this.onClickIndicatorHandler);
            }

            // If autoRotation is enabled then set up the hover handlers
            if (this.settings.autoRotation) {
                this.$element.on('mouseover', this.onCarouselHoverHandler);
                this.$element.on('mouseout', this.offCarouselHoverHandler);
            }

        } else {
            // Set up our standard swipe options.
            var swipeOptions = {
                triggerOnTouchEnd: true,
                threshold: 75,
                allowPageScroll: 'vertical',
                swipeStatus: this.swipeStatus.bind(this) // Bind this so we can target carousel functions and variables correctly.
            };

            // Enable swipe
            this.$element.swipe(swipeOptions);
        }

        this.showHiddenSlides();

        // Initialize the resize handler.
        $(window).resize(this.resizeCarouselHandler);

        return this;
    };

    /**
     * Disables the component.
     * Tears down any event binding to handlers.
     * Exits early if it is already disabled.
     *
     * @method disable
     * @returns {CarouselView}
     * @public
     */
    CarouselView.prototype.disable = function () {
        if (!this.isEnabled) {
            return this;
        }
        this.isEnabled = false;
        // this.exampleItem.off('click', this.onClickHandler);


        return this;
    };


    /**
     * Destroys the component.
     * Tears down any events, handlers, elements.
     * Should be called when the object should be left unused.
     *
     * @method destroy
     * @returns {CarouselView}
     * @public
     */
    CarouselView.prototype.destroy = function () {
        this.disable()
            .removeChildren();

        return this;
    };

//////////////////////////////////////////////////////////////////////////////////
// EVENT HANDLERS
//////////////////////////////////////////////////////////////////////////////////

    /**
     * onClickNext handler
     * This handles when the next arrow/button is pressed.
     *
     * @method onClickNext
     * @param e {Event} JavaScript event object.
     * @private
     */
    CarouselView.prototype.onClickNext = function (e) {
        e.preventDefault();

        this.nextImage();
        this.setHeightScrollControls();
        this.updateScrollControls();
    };

    /**
     * onClickPrev handler
     * This handles when the previous arrow/button is pressed.
     *
     * @method onClickPrev
     * @param e {Event} JavaScript event object.
     * @private
     */
    CarouselView.prototype.onClickPrev = function (e) {
        e.preventDefault();

        this.previousImage();
        this.setHeightScrollControls();
        this.updateScrollControls();
    };

    /**
     * onCarouselHover handler
     * This handles when a user hovers on a rotating carousel.
     *
     * @method onCarouselHover
     * @private
     */

    CarouselView.prototype.onCarouselHover = function () {
        this.stopAutoRotateCarousel();
    };

    /**
     * offCarouselHover handler
     * This handles when a user mousesout of a rotating carousel.
     *
     * @method offCarouselHover
     * @private
     */

    CarouselView.prototype.offCarouselHover = function () {
        this.autoRotateCarousel();
    };

    /**
    * offCarouselHover handler
    * This handles when a user mousesout of a rotating carousel.
    *
    * @method offCarouselHover
    * @private
    */

    CarouselView.prototype.onClickIndicator = function (e) {
        this.goToImage(e);
        this.setHeightScrollControls();
        this.updateScrollControls();
    };

//////////////////////////////////////////////////////////////////////////////////
// EVENTS
//////////////////////////////////////////////////////////////////////////////////

    /**
     * changeSlide
     * This takes the current slide's width and index, multiplies them by each other
     * and then moves the left margin of the carousel to the correct position. CSS
     * transitions are used in order to get the animation effect.
     *
     * @method changeSlide
     * @param $currentSlide
     * @private
     */

    CarouselView.prototype.changeSlide = function ($currentSlide) {
        this.currentImg = parseInt($currentSlide.index());
        var slidePosition = (this.currentImg * this.slideWidth);

        this.scrollImages(slidePosition, this.speed);
    };


    /**
     * resizeCarousel
     * This calls the layout function on the carousel whenever the window is resized.
     * This allows the carousel to responsively re-scale when viewport size changes.
     *
     * @method resizeCarousel
     * @private
     */

    CarouselView.prototype.resizeCarousel = function () {
        this.layoutCarousel();
        this.scrollImages(this.slideWidth * this.currentImg, this.speed);
    };

    /**
     * swipeStatus
     * This function checks the swipe status. It checks to see which phase it is in as well as the
     * direction of the swipe. In the move phase it calculates the direction and then scrolls the
     * slider left or right keeping the position of the user's finger. It also accounts for if the
     * interaction is canceled and returns the current slide to focus. On end if the threshold is
     * reached it will move to the next slide completely.
     *
     * @method swipeStatus
     * @private
     */

    CarouselView.prototype.swipeStatus = function(event, phase, direction, distance, duration, fingerCount) {
        //If we are moving before swipe, and we are going Lor R in X mode, or U or D in Y mode then drag.
        if( phase == "move" && (direction == "left" || direction == "right") ) {
            var duration = 0;
            if (this.settings.autoRotation) {
                this.stopAutoRotateCarousel();
            }

            if (direction == "left") {
                this.scrollImages((this.slideWidth * this.currentImg) + distance, duration);
            }

            else if (direction == "right") {
                this.scrollImages((this.slideWidth * this.currentImg) - distance, duration);
            }
        }

        else if ( phase == "cancel") {
            this.scrollImages(this.slideWidth * this.currentImg, this.speed);
        }

        else if ( phase =="end" ) {
            if (direction == "right")
                this.previousImage();
            else if (direction == "left")
                this.nextImage();
        }
    };

    /**
     * previousImage
     * This triggers the previous slide to scroll into view. This is
     * done by getting the current slide index and the width of the slide,
     * then pass it into the scrollImages function where the heavy lifting
     * is done.
     *
     * @method previousImage
     * @private
     */

    CarouselView.prototype.previousImage = function () {
        this.currentImg = Math.max(this.currentImg - 1, 0);
        this.scrollImages( this.slideWidth * this.currentImg, this.speed);
    };

    /**
     * nextImage
     * This triggers the next slide to scroll into view. This is
     * done by getting the current slide index and the width of the slide,
     * then pass it into the scrollImages function where the heavy lifting
     * is done.
     *
     * @method nextImage
     * @private
     */

    CarouselView.prototype.nextImage = function () {
        this.currentImg = Math.min(this.currentImg + 1, this.maxImages - 1);
        this.scrollImages( this.slideWidth * this.currentImg, this.speed);
    };

    CarouselView.prototype.goToImage = function(e) {
        var $clickedEl = $(e.currentTarget);
        var currentIndex = $clickedEl.index();
        this.currentImg = currentIndex;

        this.$carouselControlIndicatorLink.removeClass('isActive');

        this.scrollImages(this.slideWidth * currentIndex, this.speed);
    };

    CarouselView.prototype.updateCarouselIndicator = function(currentImg) {
        this.$carouselControlIndicatorLink.removeClass('isActive');
        this.$carouselControlIndicatorLink.eq(currentImg).addClass('isActive');
    };

    /**
     * scrollImages
     * This does the heavy lifting of the scroll, it is accomplished by
     * feeding the distance into a css transform. It also affects the duration
     * by feeding the move rate of the finger into the transition duration.
     *
     * @method scrollImages
     * @private
     */

    CarouselView.prototype.scrollImages = function (distance, duration) {
        this.$carouselList.css({
            '-webkit-transition-duration':  (duration/1000).toFixed(1) + "s",
            '-moz-transition-duration':     (duration/1000).toFixed(1) + "s",
            '-o-transition-duration':       (duration/1000).toFixed(1) + "s",
            'transition-duration':          (duration/1000).toFixed(1) + "s"
        });

        //inverse the number we set in the css
        var value = (distance<0 ? "" : "-") + Math.abs(distance).toString();

        if ($('.mod-csstransforms3d').length) {
            this.$carouselList.css({
                '-webkit-transform':    'translate3d(' + value + 'px,0px,0px)',
                '-moz-transform':       'translate3d(' + value + 'px,0px,0px)',
                '-ms-transform':        'translate3d(' + value + 'px,0px,0px)',
                '-o-transform':         'translate3d(' + value + 'px,0px,0px)',
                transform:              'translate3d(' + value + 'px,0px,0px)'
            });
        } else {
            this.$carouselList.css({
                'margin-left': value + 'px'
            });
        }

    };

    /**
     * autoRotateCarousel
     * This sets the auto rotation function to a module scoped variable, so that
     * this variable can be called in a clearInterval at will.
     *
     * @method scrollImages
     * @private
     */

    CarouselView.prototype.autoRotateCarousel = function () {
        var self = this;
        this.autoRotateInterval = setInterval(this.playCarouselLoop.bind(this), 5000);
    };

    /**
     * stopAutoRotateCarousel
     * This simply clears the set interval.
     *
     * @method stopAutoRotateCarousel
     * @private
     */

    CarouselView.prototype.stopAutoRotateCarousel = function () {
        clearInterval(this.autoRotateInterval);
    };

    /**
     * playCarouselLoop
     * This function allows the looping of carousel images. So when the end
     * is reached the first slide snaps into view. This gives the user a visual
     * clue that the end has been reached and the slide show has been restarted.
     *
     * @method playCarouselLoop
     * @private
     */

    CarouselView.prototype.playCarouselLoop = function () {
        if (this.currentImg < (this.maxImages - 1)) {
            this.nextImage();
            this.updateScrollControls();
            this.setHeightScrollControls();
        } else {
            this.currentImg = 0;
            this.updateScrollControls();
            this.setHeightScrollControls();
            this.scrollImages(this.slideWidth * this.currentImg, 0);
        }
    };

    /**
     * layoutCarousel
     * This is the function that sets up the carousel on load and on resize.
     * It gets the slide width and the number of items in the carousel then uses them
     * to set the width of the carousel. It also sets the width of the slides, since they
     * are responsive and will grow to fill the available space otherwise, to the width of
     * the viewable area. Doing it this way also fixes slides wrapping to the next row.
     *
     * @method layoutCarousel
     * @private
     */

    CarouselView.prototype.layoutCarousel = function () {

        var slideWidth = parseInt(this.$element.outerWidth(true));
        var carouselLength = parseInt(this.$carouselList.find('li').length);
        var carouselWidth = (slideWidth * carouselLength) + 'px';

        this.$carouselList.find('li').css('width', slideWidth);

        this.$carouselList.css('width', carouselWidth);

        this.slideWidth = slideWidth;

        this.maxImages = carouselLength;
    };

    /**
     * buildScrollControls
     * This builds the scroll controls based on if it is a touch device or not.
     *
     * @method buildScrollControls
     * @private
     */

    CarouselView.prototype.buildScrollControls = function () {
        var numberOfDirections = 2;
        var i = 0;
        var numberOfSlides = this.$carouselList.find('li').length;

        for (; i < numberOfDirections; i++) {
            var direction;
            if (i == 0) {
                direction = 'previous';
            } else if (i == 1) {
                direction = 'next';
            }

            this.$carouselControlList.append('<li class="carousel-controls-link"><a class="' + this.settings.scrollControlsStylingClass + ' ' + this.settings.scrollControlsStylingClass + '-' + direction + ' ' + this.jsControlClass + direction + '" href="#"><span class="isVisuallyHidden">' + direction + '</span></a></li>');

        }
        this.$carouselRight = this.$element.find('.js-carousel-controls_next');
        this.$carouselLeft = this.$element.find('.js-carousel-controls_previous');

        if (this.settings.isInTab) {
            this.$carouselRight.addClass('carousel-controls-button_low');
            this.$carouselLeft.addClass('carousel-controls-button_low');
        }

        for(i = 0; i < numberOfSlides; i++) {
            this.$carouselControlIndicatorList.append('<button class="carousel-indicator-list-link">' + (i + 1) + '</button>');
        }

        this.$carouselControlIndicatorLink = this.$carouselControlIndicatorList.find('.carousel-indicator-list-link');

        this.$carouselControlIndicatorLink.first().addClass('isActive');

        this.updateScrollControls();
        this.setHeightScrollControls();
    };

    /**
     * setHeightScrollControls
     * This sets the height of the scroll controls based on the current image.
     *
     * @method setHeightScrollControls
     * @private
     */

    CarouselView.prototype.setHeightScrollControls = function () {
        var _self = this;

        var $currentImg = _self.$carouselList.find('li:eq(' + _self.currentImg + ') img');
        var currentImgSrc = $currentImg.attr('src');

        $('<img />').attr('src', currentImgSrc).load(function() {
            var newHeight = $currentImg.height();

            _self.$carouselControlIndicator.css('top', (newHeight + 5) + 'px');
            _self.$carouselRight.height(newHeight);
            _self.$carouselLeft.height(newHeight);
        });
    };

    /**
     * updateScrollControls
     * This will show / hide the controls based on the current image position.
     *
     * @method updateScrollControls
     * @private
     */

    CarouselView.prototype.updateScrollControls = function () {
        if(this.currentImg == 0) {
            this.$carouselLeft.hide();
            this.$carouselRight.show();
        } else if (this.currentImg == this.$carouselList.find('li').length-1) {
            this.$carouselLeft.show();
            this.$carouselRight.hide();
        } else {
            this.$carouselLeft.show();
            this.$carouselRight.show();
        }

        this.updateCarouselIndicator(this.currentImg);
    };

//////////////////////////////////////////////////////////////////////////////////
// Test for events
// Written by Adam Ranfelt
//////////////////////////////////////////////////////////////////////////////////

    /**
     * Executes all tests that the feature testing support requires
     *
     * @method executeTests
     */
    CarouselView.prototype.executeTests = function() {
        this.testInteraction();
    };

    /**
     * Tests the current interaction profile to determine which event family to use
     *
     * @method testInteraction
     */
    CarouselView.prototype.testInteraction = function() {
        var profile;

        // UNTESTED POINTER TEST
        // Test code gathered from: http://msdn.microsoft.com/en-us/library/ie/hh673557(v=vs.85).aspx
        if ('ontouchstart' in document.documentElement) {
            profile = PROFILES.MOBILE;
        } else {
            profile = PROFILES.DESKTOP;
        }

        this.interactionProfile = profile;

    };

    return CarouselView;

});
