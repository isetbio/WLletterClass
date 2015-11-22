/**
 * Application configuration declaration.
 *
 * This configuration file is shared between the website and the build script so
 * that values don't have to be duplicated across environments. Any non-shared,
 * environment-specific configuration should placed in appropriate configuration
 * files.
 *
 * Paths to vendor libraries may be added here to provide short aliases to
 * otherwise long and arbitrary paths. If you're using bower to manage vendor
 * scripts, running `grunt install` will automatically add paths aliases as
 * needed.
 *
 * @example
 *     paths: {
 *         'jquery': '../vendor/jquery/jquery',
 *         'jquery-cookie': '../vendor/jquery-cookie/jquery-cookie'
 *     }
 *
 * Shims provide a means of managing dependencies for non-modular, or non-AMD
 * scripts. For example, jQuery UI depends on jQuery, but it assumes jQuery is
 * available globally. Because RequireJS loads scripts asynchronously, jQuery
 * may or may not be available which will cause a runtime error. Shims solve
 * this problem.
 *
 * @example
 *     shim: {
 *         'jquery-cookie': {
 *             deps: ['jquery'],
 *             exports: null
 *          }
 *     }
 *
 */
require.config({
    paths: {

        'polyfillGetComputedStyle': '../../../../static/web/assets/scripts/lib/polyfill.getComputedStyle',
        'nerdery-function-bind': '../../../../tools/cache/nerdery-function-bind/nerdery-function-bind',
        'requirejs': '../../../../src/assets/vendor/requirejs/require',
        'jquery': '../../../../src/assets/vendor/jquery/jquery',
        'underscore': '../../../../src/assets/vendor/underscore/underscore'
    },
    shim: {

        requirejs: '../../../../src/assets/vendor/requirejs/require',
        jquery: '../../../../src/assets/vendor/jquery/jquery',
        underscore: '../../../../src/assets/vendor/underscore/underscore',
        jqueryTouchSwipe: 'plugins/jquery.touchSwipe.min.js',

        jqueryTouchSwipe: {
            deps: ['jquery'],
            exports: null
        }
    },

    deps: [
        'polyfillGetComputedStyle'
    ],
    waitSeconds: 120
});
