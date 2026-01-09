FROM node:10-buster

# Fix sources for archived Debian Buster
RUN echo "deb http://archive.debian.org/debian buster main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main" >> /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/* && \
    echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf.d/99no-check-valid-until

# Install build tools
RUN apt-get update && apt-get install -y python3 gcc g++ make libtool

# Suppress GCC 8+ warnings treated as errors in grpc 1.14
ENV CXXFLAGS="-Wno-error=class-memaccess -Wno-error=ignored-qualifiers -Wno-error=stringop-overflow -Wno-error=stringop-truncation"

# Install caliper-cli globally
RUN npm config set unsafe-perm true
RUN npm -g config set user root
RUN npm install -g @hyperledger/caliper-cli@0.3.2

# Patch @so-ric/colorspace in global node_modules
# 1. Fix ||= syntax (Node 15+)
RUN find /usr/local/lib/node_modules -name index.cjs.js -exec grep -l "limiters\[m\] ||=" {} + | xargs sed -i 's/(limiters\[m\] ||= \[\])\[channel\] = modifier;/(limiters[m] = limiters[m] || [])[channel] = modifier;/g'

# 2. Fix Object.hasOwn (Node 16.9+)
RUN find /usr/local/lib/node_modules -name index.cjs.js -exec grep -l "Object.hasOwn" {} + | xargs sed -i 's/Object.hasOwn(\([^,]*\), \([^)]*\))/Object.prototype.hasOwnProperty.call(\1, \2)/g'

# 3. Fix Numeric Separators (Node 12.5+)
# Example: 0.003_130_8 -> 0.0031308
RUN find /usr/local/lib/node_modules -name index.cjs.js -exec sed -i 's/\([0-9]\)_\([0-9]\)/\1\2/g' {} +

# Bind Fabric 1.4.4 globally
RUN caliper bind --caliper-bind-sut fabric:1.4.4 --caliper-bind-args="-g --unsafe-perm"
