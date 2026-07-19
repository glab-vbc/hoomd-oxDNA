// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "OxDNABondedForceCompute.h"
#include "OxDNABondedKernel.h"
#include "MixedPrecisionCompat.h"

#include <stdexcept>

using namespace std;

/*! \file OxDNABondedForceCompute.cc
    \brief CPU driver for the oxDNA bonded force. The per-bond physics (FENE +
    bonded excluded volume + stacking) lives in OxDNABondedKernel.h and is shared
    verbatim with the GPU kernel.
*/

namespace hoomd
    {
namespace md
    {

OxDNABondedForceCompute::OxDNABondedForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_fene_energy(0), m_bexc_energy(0), m_stack_energy(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing OxDNABondedForceCompute" << endl;

    m_bond_data = m_sysdef->getBondData();
    if (m_bond_data->getNTypes() == 0)
        throw runtime_error("OxDNABondedForceCompute: no bond types in the system.");
    m_params.resize(m_bond_data->getNTypes());
    }

OxDNABondedForceCompute::~OxDNABondedForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying OxDNABondedForceCompute" << endl;
    }

void OxDNABondedForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    unsigned int typ = m_bond_data->getTypeByName(type);
    if (typ >= m_bond_data->getNTypes())
        throw runtime_error("OxDNABondedForceCompute: invalid bond type.");
    m_params[typ] = oxdna_bonded_params(params);
    }

pybind11::dict OxDNABondedForceCompute::getParams(std::string type)
    {
    unsigned int typ = m_bond_data->getTypeByName(type);
    if (typ >= m_bond_data->getNTypes())
        throw runtime_error("OxDNABondedForceCompute: invalid bond type.");
    return m_params[typ].asDict();
    }

void OxDNABondedForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<ForceReal4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<ForceReal4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<ForceReal> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    m_force.zeroFill();
    m_torque.zeroFill();
    m_virial.zeroFill();

    const BoxDim& box = m_pdata->getGlobalBox();
    m_fene_energy = m_bexc_energy = m_stack_energy = Scalar(0.0);

    const unsigned int n_bonds = (unsigned int)m_bond_data->getN();
    for (unsigned int i = 0; i < n_bonds; i++)
        {
        const BondData::members_t bond = m_bond_data->getMembersByIndex(i);
        unsigned int idxA = h_rtag.data[bond.tag[0]];
        unsigned int idxB = h_rtag.data[bond.tag[1]];
        if (idxA == NOT_LOCAL || idxB == NOT_LOCAL)
            {
            m_exec_conf->msg->error() << "OxDNABondedForceCompute: incomplete bond "
                                      << bond.tag[0] << " " << bond.tag[1] << endl;
            throw runtime_error("Error in oxDNA bonded force calculation");
            }

        const oxdna_bonded_params& p = m_params[m_bond_data->getTypeByIndex(i)];

        Scalar3 posA = make_scalar3(h_pos.data[idxA].x, h_pos.data[idxA].y, h_pos.data[idxA].z);
        Scalar3 posB = make_scalar3(h_pos.data[idxB].x, h_pos.data[idxB].y, h_pos.data[idxB].z);
        Scalar3 Rc3 = box.minImage(make_scalar3(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z));
        vec3<Scalar> Rc(Rc3.x, Rc3.y, Rc3.z);

        quat<Scalar> qA(h_orientation.data[idxA]);
        quat<Scalar> qB(h_orientation.data[idxB]);
        unsigned int typeA = __scalar_as_int(h_pos.data[idxA].w);
        unsigned int typeB = __scalar_as_int(h_pos.data[idxB].w);

        oxdna::BondedResult r;
        oxdna::oxdna_bonded_pair(p, Rc, qA, qB, typeA, typeB, r);

        m_fene_energy += r.e_fene;
        m_bexc_energy += r.e_bexc;
        m_stack_energy += r.e_stack;
        Scalar e_half = Scalar(0.5) * (r.e_fene + r.e_bexc + r.e_stack);

        if (idxA < m_pdata->getN())
            {
            h_force.data[idxA].x += -r.FB.x;
            h_force.data[idxA].y += -r.FB.y;
            h_force.data[idxA].z += -r.FB.z;
            h_force.data[idxA].w += e_half;
            h_torque.data[idxA].x += r.tauA.x;
            h_torque.data[idxA].y += r.tauA.y;
            h_torque.data[idxA].z += r.tauA.z;
            for (int v = 0; v < 6; v++)
                h_virial.data[v * virial_pitch + idxA] += r.vir[v];
            }
        if (idxB < m_pdata->getN())
            {
            h_force.data[idxB].x += r.FB.x;
            h_force.data[idxB].y += r.FB.y;
            h_force.data[idxB].z += r.FB.z;
            h_force.data[idxB].w += e_half;
            h_torque.data[idxB].x += r.tauB.x;
            h_torque.data[idxB].y += r.tauB.y;
            h_torque.data[idxB].z += r.tauB.z;
            for (int v = 0; v < 6; v++)
                h_virial.data[v * virial_pitch + idxB] += r.vir[v];
            }
        }
    }

namespace detail
    {
void export_OxDNABondedForceCompute(pybind11::module& m)
    {
    pybind11::class_<OxDNABondedForceCompute,
                     ForceCompute,
                     std::shared_ptr<OxDNABondedForceCompute>>(m, "OxDNABondedForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &OxDNABondedForceCompute::setParamsPython)
        .def("getParams", &OxDNABondedForceCompute::getParams)
        .def("getFeneEnergy", &OxDNABondedForceCompute::getFeneEnergy)
        .def("getBondedExcVolEnergy", &OxDNABondedForceCompute::getBondedExcVolEnergy)
        .def("getStackingEnergy", &OxDNABondedForceCompute::getStackingEnergy);
    }
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
