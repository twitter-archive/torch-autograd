return {
   -- generic generator, from sequential list of layers:
   sequence = function(layers, layer2params)
      return function(params, input)
         for i,layer in ipairs(layers) do
            local paramsi = layer2params[i]
            if paramsi then
               input = layer(params[paramsi], input)
            else
               input = layer(input)
            end
         end
         return input
      end
   end
}
