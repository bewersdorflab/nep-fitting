import jinja2
import os
import base64

env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.split(__file__)[0]))

from PYME.Analysis import graphing_filters #FIXME - move the filters somewhere better
#env.filters['movieplot'] = graphing_filters.movieplot2
env.filters['plot'] = graphing_filters.plot
env.filters['hist'] = graphing_filters.hist

#filter to encode base 64 encode data
env.filters['b64encode'] = base64.b64encode

def generate(context, template_name):
    template = env.get_template(template_name)
    return template.render(**context)

def generate_and_save(filename, context, template_name):
    with open(filename, 'w') as f:
        f.write(generate(context, template_name))