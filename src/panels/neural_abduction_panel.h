#pragma once

// The NeuralAbduction panel: what the abduction network guesses is relevant to
// the task in context. One row per representation the model can vouch for, worn
// as its own item icon, ordered by how strongly it is abduced.

#include <flecs.h>

namespace panels {

struct NeuralAbduction
{
    NeuralAbduction(flecs::world& world);
};

} // namespace panels
