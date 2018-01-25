import inspect

from bottle import PluginError, error


class SKMPlugin(object):
    ''' This plugin passes an SKMPlugin handle to route callbacks
    that accept a `sciknowmap` keyword argument. If a callback does not expect
    such a parameter, no connection is made. You can override the sciknowmap
    settings on a per-route basis. '''

    name = 'sciknowmap'
    api = 2

    def __init__(self, keyword='sciknowmap'):
        self.keyword = keyword

        
    def setup(self, app):
        ''' Make sure that other installed plugins don't affect the same
            keyword argument.'''
        for other in app.plugins:
            if not isinstance(other, SKMPlugin): continue
            if other.keyword == self.keyword:
                raise PluginError("Found another sqlite plugin with "\
                "conflicting settings (non-unique keyword).")

    def apply(self, callback, context):
        # Override global configuration with route-specific values.
        conf = context.config.get('sciknowmap') or {}
        keyword = conf.get('keyword', self.keyword)

        # Test if the original callback accepts a 'db' keyword.
        # Ignore it if it does not need a database handle.
        args = inspect.getargspec(context.callback)[0]
        if keyword not in args:
            return callback

        def wrapper(*args, **kwargs):
            
            # Add the tagger handle as a keyword argument.
            kwargs[keyword] = self

            #try:
            rv = callback(*args, **kwargs)
            #except StandardError, e:
            #    raise HTTPError(500, "SciDT Server Error", e)

            return rv

        # Replace the route callback with the wrapped one.
        return wrapper
