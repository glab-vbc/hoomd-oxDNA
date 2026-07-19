// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABondedForceComputeGPU.cc
    \brief Defines OxDNABondedForceComputeGPU.
*/

#include "OxDNABondedForceComputeGPU.h"
#include "MixedPrecisionCompat.h"

using namespace std;

namespace hoomd
    {
namespace md
    {

OxDNABondedForceComputeGPU::OxDNABondedForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : OxDNABondedForceCompute(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating an OxDNABondedForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing OxDNABondedForceComputeGPU");
        }

    GPUArray<oxdna_bonded_params> params(m_bond_data->getNTypes(), m_exec_conf);
    m_params_gpu.swap(params);

    GPUArray<Scalar4> split(m_pdata->getN(), m_exec_conf);
    m_split_energy.swap(split);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf, "oxdna_bonded"));
    m_autotuners.push_back(m_tuner);
    }

OxDNABondedForceComputeGPU::~OxDNABondedForceComputeGPU() { }

void OxDNABondedForceComputeGPU::setParamsPython(std::string type, pybind11::dict params)
    {
    OxDNABondedForceCompute::setParamsPython(type, params);
    unsigned int typ = m_bond_data->getTypeByName(type);
    ArrayHandle<oxdna_bonded_params> h_params(m_params_gpu, access_location::host,
                                              access_mode::readwrite);
    h_params.data[typ] = m_params[typ];
    }

Scalar OxDNABondedForceComputeGPU::reduceSplit(int component)
    {
    ArrayHandle<Scalar4> h_split(m_split_energy, access_location::host, access_mode::read);
    Scalar total = Scalar(0.0);
    unsigned int n = m_pdata->getN();
    for (unsigned int i = 0; i < n; i++)
        {
        Scalar4 s = h_split.data[i];
        total += (component == 0) ? s.x : (component == 1) ? s.y : s.z;
        }
    return total;
    }

void OxDNABondedForceComputeGPU::computeForces(uint64_t timestep)
    {
    if (m_split_energy.getNumElements() != m_pdata->getN())
        {
        GPUArray<Scalar4> split(m_pdata->getN(), m_exec_conf);
        m_split_energy.swap(split);
        }

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device,
                                       access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<ForceReal4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<ForceReal4> d_torque(m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<ForceReal> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_split(m_split_energy, access_location::device, access_mode::overwrite);
    ArrayHandle<oxdna_bonded_params> d_params(m_params_gpu, access_location::device,
                                              access_mode::read);

    ArrayHandle<BondData::members_t> d_blist(m_bond_data->getGPUTable(), access_location::device,
                                             access_mode::read);
    ArrayHandle<unsigned int> d_bpos(m_bond_data->getGPUPosTable(), access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_n_bonds(m_bond_data->getNGroupsArray(), access_location::device,
                                        access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_oxdna_bonded_forces(d_force.data, d_torque.data, d_virial.data,
                                            m_virial.getPitch(), d_split.data, m_pdata->getN(),
                                            d_pos.data, d_orientation.data, box, d_blist.data,
                                            d_bpos.data, m_bond_data->getGPUTableIndexer().getW(),
                                            d_n_bonds.data, d_params.data,
                                            m_bond_data->getNTypes(), m_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_OxDNABondedForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<OxDNABondedForceComputeGPU,
                     OxDNABondedForceCompute,
                     std::shared_ptr<OxDNABondedForceComputeGPU>>(m, "OxDNABondedForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
