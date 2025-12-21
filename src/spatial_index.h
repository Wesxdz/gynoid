#pragma once

#include <flecs.h>
#include <spatialindex/SpatialIndex.h>
#include <lmdb.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <unistd.h>

namespace spatial {

// Visitor class to collect entity IDs from spatial queries
class EntityCollectorVisitor : public SpatialIndex::IVisitor {
public:
    std::vector<SpatialIndex::id_type> ids;

    void visitNode(const SpatialIndex::INode& n) override {}

    void visitData(const SpatialIndex::IData& d) override {
        ids.push_back(d.getIdentifier());
    }

    void visitData(std::vector<const SpatialIndex::IData*>& v) override {
        for (auto* data : v) {
            ids.push_back(data->getIdentifier());
        }
    }
};

// 1D spatial index using LMDB B+ tree
class SpatialIndex1D {
private:
    MDB_env* env;
    MDB_dbi dbi;
    std::unordered_map<uint64_t, float> entity_to_value;

    // Convert float to sortable byte key
    void float_to_key(float value, unsigned char* key) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(float));
        // Flip sign bit for sorting, flip all bits if negative
        if (bits & 0x80000000) {
            bits = ~bits;
        } else {
            bits |= 0x80000000;
        }
        // Convert to big endian for byte-wise sorting
        key[0] = (bits >> 24) & 0xFF;
        key[1] = (bits >> 16) & 0xFF;
        key[2] = (bits >> 8) & 0xFF;
        key[3] = bits & 0xFF;
    }

public:
    SpatialIndex1D(const std::string& db_name = "default") {
        // Create environment without subdirectory - use unique file per field
        std::string db_path = "/tmp/lmdb_1d_" + db_name + ".db";

        // Delete existing database file to ensure clean state on restart
        unlink(db_path.c_str());

        // Create LMDB environment
        int rc = mdb_env_create(&env);
        if (rc != 0) {
            fprintf(stderr, "mdb_env_create failed: %s\n", mdb_strerror(rc));
            throw std::runtime_error("Failed to create LMDB environment");
        }

        mdb_env_set_mapsize(env, 10485760); // 10MB

        rc = mdb_env_open(env, db_path.c_str(), MDB_NOSUBDIR | MDB_NOSYNC, 0664);
        if (rc != 0) {
            fprintf(stderr, "mdb_env_open failed: %s\n", mdb_strerror(rc));
            throw std::runtime_error("Failed to open LMDB environment");
        }

        // Open database
        MDB_txn* txn;
        rc = mdb_txn_begin(env, nullptr, 0, &txn);
        if (rc != 0) {
            fprintf(stderr, "mdb_txn_begin failed: %s\n", mdb_strerror(rc));
            throw std::runtime_error("Failed to begin LMDB transaction");
        }

        rc = mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi);
        if (rc != 0) {
            fprintf(stderr, "mdb_dbi_open failed: %s\n", mdb_strerror(rc));
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to open LMDB database");
        }

        rc = mdb_txn_commit(txn);
        if (rc != 0) {
            fprintf(stderr, "mdb_txn_commit failed: %s\n", mdb_strerror(rc));
            throw std::runtime_error("Failed to commit LMDB transaction");
        }
    }

    ~SpatialIndex1D() {
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
    }

    void insert(flecs::entity e, float value) {
        uint64_t eid = e.id();

        // Remove old value if exists
        auto it = entity_to_value.find(eid);
        if (it != entity_to_value.end()) {
            printf("DEBUG: 1D index updating entity %lu (was %.2f, now %.2f)\n", eid, it->second, value);
            remove_entity(e);
        } else {
            printf("DEBUG: 1D index inserting entity %lu with value %.2f\n", eid, value);
        }

        // Create composite key: float_bytes + entity_id
        unsigned char key_data[12];
        float_to_key(value, key_data);
        memcpy(key_data + 4, &eid, sizeof(uint64_t));

        printf("  Key bytes: [%02X %02X %02X %02X] + entity %lu\n",
               key_data[0], key_data[1], key_data[2], key_data[3], eid);

        MDB_val key = {12, key_data};
        MDB_val data = {0, nullptr};

        MDB_txn* txn;
        mdb_txn_begin(env, nullptr, 0, &txn);
        mdb_put(txn, dbi, &key, &data, 0);
        mdb_txn_commit(txn);

        entity_to_value[eid] = value;
    }

    void remove_entity(flecs::entity e) {
        uint64_t eid = e.id();
        auto it = entity_to_value.find(eid);
        if (it == entity_to_value.end()) return;

        float value = it->second;
        unsigned char key_data[12];
        float_to_key(value, key_data);
        memcpy(key_data + 4, &eid, sizeof(uint64_t));

        MDB_val key = {12, key_data};

        MDB_txn* txn;
        mdb_txn_begin(env, nullptr, 0, &txn);
        mdb_del(txn, dbi, &key, nullptr);
        mdb_txn_commit(txn);

        entity_to_value.erase(it);
    }

    std::unordered_set<uint64_t> range_query(float min, float max) {
        std::unordered_set<uint64_t> results;

        unsigned char min_key[12];
        float_to_key(min, min_key);
        memset(min_key + 4, 0, 8); // Start of range

        MDB_txn* txn;
        mdb_txn_begin(env, nullptr, MDB_RDONLY, &txn);

        MDB_cursor* cursor;
        mdb_cursor_open(txn, dbi, &cursor);

        MDB_val key = {12, min_key};
        MDB_val data;

        // Position cursor at first key >= min
        int rc = mdb_cursor_get(cursor, &key, &data, MDB_SET_RANGE);

        printf("DEBUG: range_query [%.2f, %.2f] starting scan\n", min, max);
        while (rc == 0) {
            // Extract float from key
            unsigned char* key_bytes = (unsigned char*)key.mv_data;

            printf("  Reading key bytes: [%02X %02X %02X %02X]\n",
                   key_bytes[0], key_bytes[1], key_bytes[2], key_bytes[3]);

            uint32_t bits = (key_bytes[0] << 24) | (key_bytes[1] << 16) |
                           (key_bytes[2] << 8) | key_bytes[3];
            // Reverse the transform
            if (bits & 0x80000000) {
                bits &= 0x7FFFFFFF;
            } else {
                bits = ~bits;
            }
            float key_value;
            memcpy(&key_value, &bits, sizeof(float));

            // Extract entity ID from key
            uint64_t eid;
            memcpy(&eid, key_bytes + 4, sizeof(uint64_t));

            printf("  Decoded: value=%.2f, entity=%lu\n", key_value, eid);

            // Check if still in range
            if (key_value > max) {
                printf("  value %.2f > max %.2f, stopping\n", key_value, max);
                break;
            }

            results.insert(eid);

            rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT);
        }

        mdb_cursor_close(cursor);
        mdb_txn_abort(txn);

        return results;
    }
};

// 2D spatial index using R-tree
class SpatialIndex2D {
private:
    std::unique_ptr<SpatialIndex::IStorageManager> storage;
    std::unique_ptr<SpatialIndex::ISpatialIndex> rtree;

    std::unordered_map<uint64_t, SpatialIndex::id_type> entity_to_spatial;
    std::unordered_map<SpatialIndex::id_type, uint64_t> spatial_to_entity;
    SpatialIndex::id_type next_id = 1;

public:
    SpatialIndex2D() {
        storage.reset(SpatialIndex::StorageManager::createNewMemoryStorageManager());
        SpatialIndex::id_type indexIdentifier;
        rtree.reset(SpatialIndex::RTree::createNewRTree(
            *storage,
            0.7,
            100,
            100,
            2,
            SpatialIndex::RTree::RV_RSTAR,
            indexIdentifier
        ));
    }

    void insert_bounds(flecs::entity e, float x, float y, float width, float height) {
        uint64_t eid = e.id();

        // Don't allow updates - entity must be removed first
        if (entity_to_spatial.find(eid) != entity_to_spatial.end()) {
            printf("DEBUG: 2D index - entity %lu already exists, skipping update\n", eid);
            return;
        }

        SpatialIndex::id_type sid = next_id++;
        entity_to_spatial[eid] = sid;
        spatial_to_entity[sid] = eid;

        double low[2] = {static_cast<double>(x), static_cast<double>(y)};
        double high[2] = {static_cast<double>(x + width), static_cast<double>(y + height)};
        SpatialIndex::Region region(low, high, 2);

        printf("DEBUG: 2D index inserting entity %lu with bounds (%.2f,%.2f,%.2f,%.2f) -> region [%.2f,%.2f] to [%.2f,%.2f]\n",
               eid, x, y, width, height, low[0], low[1], high[0], high[1]);

        rtree->insertData(0, nullptr, region, sid);
    }

    void remove_entity(flecs::entity e) {
        uint64_t eid = e.id();
        auto it = entity_to_spatial.find(eid);
        if (it != entity_to_spatial.end()) {
            spatial_to_entity.erase(it->second);
            entity_to_spatial.erase(it);
        }
    }

    std::unordered_set<uint64_t> intersects_query(float x, float y, float width, float height) {
        double low[2] = {static_cast<double>(x), static_cast<double>(y)};
        double high[2] = {static_cast<double>(x + width), static_cast<double>(y + height)};
        SpatialIndex::Region query_region(low, high, 2);

        EntityCollectorVisitor visitor;
        rtree->intersectsWithQuery(query_region, visitor);

        std::unordered_set<uint64_t> results;
        for (auto sid : visitor.ids) {
            auto it = spatial_to_entity.find(sid);
            if (it != spatial_to_entity.end()) {
                results.insert(it->second);
            }
        }
        return results;
    }

    std::unordered_set<uint64_t> contains_point_query(float x, float y) {
        printf("DEBUG: 2D contains_point_query for point (%.2f, %.2f)\n", x, y);
        printf("DEBUG: R-tree has %zu entities indexed\n", entity_to_spatial.size());

        // Use intersectsWithQuery with a point region (low == high)
        double coords[2] = {static_cast<double>(x), static_cast<double>(y)};
        SpatialIndex::Region point_region(coords, coords, 2);

        EntityCollectorVisitor visitor;
        rtree->intersectsWithQuery(point_region, visitor);

        printf("DEBUG: intersectsWithQuery (point) returned %zu spatial IDs\n", visitor.ids.size());

        std::unordered_set<uint64_t> results;
        for (auto sid : visitor.ids) {
            auto it = spatial_to_entity.find(sid);
            if (it != spatial_to_entity.end()) {
                printf("DEBUG: Converting spatial ID %ld to entity %lu\n", sid, it->second);
                results.insert(it->second);
            }
        }
        return results;
    }

    std::unordered_set<uint64_t> radius_query(float x, float y, float radius) {
        return intersects_query(x - radius, y - radius, radius * 2, radius * 2);
    }
};

// String index for exact string matching
class StringIndex {
private:
    std::unordered_map<std::string, std::unordered_set<uint64_t>> value_to_entities;
    std::unordered_map<uint64_t, std::string> entity_to_value;

public:
    void insert(flecs::entity e, const std::string& value) {
        uint64_t eid = e.id();

        // Remove old value if exists
        auto it = entity_to_value.find(eid);
        if (it != entity_to_value.end()) {
            printf("DEBUG: String index updating entity %lu (was '%s', now '%s')\n",
                   eid, it->second.c_str(), value.c_str());
            value_to_entities[it->second].erase(eid);
            if (value_to_entities[it->second].empty()) {
                value_to_entities.erase(it->second);
            }
        } else {
            printf("DEBUG: String index inserting entity %lu with value '%s'\n", eid, value.c_str());
        }

        value_to_entities[value].insert(eid);
        entity_to_value[eid] = value;
    }

    void remove_entity(flecs::entity e) {
        uint64_t eid = e.id();
        auto it = entity_to_value.find(eid);
        if (it != entity_to_value.end()) {
            const std::string& value = it->second;
            value_to_entities[value].erase(eid);
            if (value_to_entities[value].empty()) {
                value_to_entities.erase(value);
            }
            entity_to_value.erase(it);
        }
    }

    std::unordered_set<uint64_t> exact_match(const std::string& value) {
        auto it = value_to_entities.find(value);
        if (it != value_to_entities.end()) {
            printf("DEBUG: String exact_match for '%s' returned %zu entities\n",
                   value.c_str(), it->second.size());
            return it->second;
        }
        printf("DEBUG: String exact_match for '%s' returned 0 entities\n", value.c_str());
        return std::unordered_set<uint64_t>();
    }
};

// Manager class
class SpatialIndexManager {
private:
    flecs::world* world;
    std::unordered_map<std::string, std::shared_ptr<SpatialIndex1D>> indices_1d;
    std::unordered_map<flecs::id_t, std::shared_ptr<SpatialIndex2D>> indices_2d;
    std::unordered_map<std::string, std::shared_ptr<StringIndex>> indices_string;

public:
    SpatialIndexManager(flecs::world* w) : world(w) {}

    template<typename T, typename ExtractorFunc>
    void register_1d_component(const std::string& field_name, ExtractorFunc extractor) {
        auto index = std::make_shared<SpatialIndex1D>(field_name);
        indices_1d[field_name] = index;

        world->observer<T>()
            .event(flecs::OnSet)
            .each([index, extractor](flecs::entity e, const T& component) {
                // Debug: log what entity is triggering the observer
                const char* name = e.name();
                printf("DEBUG: 1D observer triggered for entity %lu (%s)\n",
                       (uint64_t)e.id(), name ? name : "<unnamed>");

                // Filter out flecs internal entities (metadata, modules, type descriptors, etc.)
                if (e.has(flecs::Module) || e.has(flecs::Prefab) ||
                    e.has<flecs::Member>() || e.has<flecs::Type>() ||
                    e.has<flecs::Component>()) {
                    printf("  -> Filtered out (flecs internal entity)\n");
                    return;  // Don't index metadata entities
                }
                float value = extractor(component);
                index->insert(e, value);
            });

        world->observer<T>()
            .event(flecs::OnRemove)
            .each([index](flecs::entity e, const T&) {
                index->remove_entity(e);
            });
    }

    template<typename T, typename ExtractorFunc>
    void register_2d_bounds_component(ExtractorFunc extractor) {
        flecs::id_t component_id = world->id<T>();
        auto index = std::make_shared<SpatialIndex2D>();
        indices_2d[component_id] = index;

        world->observer<T>()
            .event(flecs::OnSet)
            .each([index, extractor](flecs::entity e, const T& component) {
                // Filter out flecs internal entities (metadata, modules, type descriptors, etc.)
                if (e.has(flecs::Module) || e.has(flecs::Prefab) ||
                    e.has<flecs::Member>() || e.has<flecs::Type>() ||
                    e.has<flecs::Component>()) {
                    return;  // Don't index metadata entities
                }
                auto [x, y, w, h] = extractor(component);
                index->insert_bounds(e, x, y, w, h);
            });

        world->observer<T>()
            .event(flecs::OnRemove)
            .each([index](flecs::entity e, const T&) {
                index->remove_entity(e);
            });
    }

    std::shared_ptr<SpatialIndex1D> get_1d_index(const std::string& field_name) {
        auto it = indices_1d.find(field_name);
        return (it != indices_1d.end()) ? it->second : nullptr;
    }

    std::vector<std::string> get_registered_fields() const {
        std::vector<std::string> fields;
        for (const auto& pair : indices_1d) {
            fields.push_back(pair.first);
        }
        return fields;
    }

    template<typename T>
    std::shared_ptr<SpatialIndex2D> get_2d_index() {
        flecs::id_t component_id = world->id<T>();
        auto it = indices_2d.find(component_id);
        return (it != indices_2d.end()) ? it->second : nullptr;
    }

    template<typename T, typename ExtractorFunc>
    void register_string_component(const std::string& field_name, ExtractorFunc extractor) {
        auto index = std::make_shared<StringIndex>();
        indices_string[field_name] = index;

        world->observer<T>()
            .event(flecs::OnSet)
            .each([index, extractor](flecs::entity e, const T& component) {
                // Filter out flecs internal entities
                if (e.has(flecs::Module) || e.has(flecs::Prefab) ||
                    e.has<flecs::Member>() || e.has<flecs::Type>() ||
                    e.has<flecs::Component>()) {
                    return;
                }
                std::string value = extractor(component);
                index->insert(e, value);
            });

        world->observer<T>()
            .event(flecs::OnRemove)
            .each([index](flecs::entity e, const T&) {
                index->remove_entity(e);
            });
    }

    std::shared_ptr<StringIndex> get_string_index(const std::string& field_name) {
        auto it = indices_string.find(field_name);
        return (it != indices_string.end()) ? it->second : nullptr;
    }

    std::vector<std::string> get_registered_string_fields() const {
        std::vector<std::string> fields;
        for (const auto& pair : indices_string) {
            fields.push_back(pair.first);
        }
        return fields;
    }
};

} // namespace spatial
