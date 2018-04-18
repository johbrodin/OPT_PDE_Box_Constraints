#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
//Added: minres solver
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>

#include "MiscUtilities.hh"

#include <sstream>
#include <string>
#include <typeinfo>
#include <list>
#include <iostream>
#include <fstream>
#include <cmath>

#define DIRICHLET_BC    0
#define NEUMANN_BC      1

#define INNER_SOLVER_CG     0
#define INNER_SOLVER_GMRES  1
#define INNER_SOLVER_DIRECT 2

//Added: linear solver types
#define LINEAR_SOLVER_MINRES 0
#define LINEAR_SOLVER_GMRES 1 //default

using namespace dealii;

namespace EquationData
{
double  beta=0;
double  epsilon=0;
double  sigma=0;
double  y_upper=0.0;
double  y_lower=-0.0;
int     bc=0;

//Added: variables for the choice of desired state
int desired_state = 1;
double k1 = 8;
double k2 = 5;

//Nonhomogeneous Dirichlet boundaryvalues
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    BoundaryValues(const unsigned int _components) : Function<dim>(_components) {}
    virtual double value (const Point<dim> &p, const unsigned int component) const;
};


template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int component) const
{
    double x1=p[0];
    double x2=p[1];
    double value;
    //    std::cout<<"component: " << component<<", x1="<< x1 <<", x2="<< x2 <<", return_val: ";
    if(component==0){
        //Added: different option for the function used as desired state
        if (desired_state == 1) //Original value
            value  = std::sin(2*M_PI*x1*x2);
        else
            value  = std::sin(k1*M_PI*x1-std::cos(k2*M_PI*x2));
        //Commented: Original value
        //value  = std::sin(2*M_PI*x1*x2);
        if(value>=y_upper)
            value=y_upper;
        if(value<=y_lower)
            value=y_lower;
//        std::cout<<value<<std::endl;
        return value;
    }
    else
        //        std::cout<< 0 << std::endl;
        return 0;
}

//Desired state
template <int dim>
class ControlValues : public Function<dim>
{
public:
    ControlValues(const unsigned int components) : Function<dim>(components) {}
    virtual double value (const Point<dim> &p, const unsigned int component) const;
};

template <int dim>
double ControlValues<dim>::value (const Point<dim> &p, const unsigned int /*n_components*/) const
{
    double x1=p[0];
    double x2=p[1];
    double return_value;//Commented: =std::sin(2*M_PI*x1*x2);
//    std::cout<< ", val: "<< return_value<<std::endl;
//    std::cout<<"x1:"<<x1<<", x2:"<<x2<<", val: "<< return_value<<std::endl;
    //Added: different option for the function used as desired state
    if (desired_state == 1) //Original return value
        return_value  = std::sin(2*M_PI*x1*x2);
    else
        return_value  = std::sin(k1*M_PI*x1-std::cos(k2*M_PI*x2));
    //Commented: Original value
    //return_value  = std::sin(2*M_PI*x1*x2);

    return return_value;
}

}//end of namespace EquationData


namespace InnerSolverData
{
unsigned int max_iter=400;
double rel_tolerance=1.0e-4;
unsigned int amg_iterations=0;
unsigned int solver_type = 1;
}

//Added: Linear solver data
namespace LinearSolverData
{
unsigned int max_iter=1000;
unsigned int solver_type = 1;
//std::string solver_name = "";
bool fixed_tolerance_on =false;
double fixed_tolerance = 0.0001;
double fixed_tolerance_reduction = 0.1;
}

//Added: Non-Linear solver data
namespace NonLinearSolverData
{
bool save_intermediate_solution = false;
}



template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
    InverseMatrix (const Matrix         &m,
                   const Preconditioner &preconditioner,
                   const unsigned int   iterations,
                   const double         rel_tolerance);

    template <typename VectorType>
    void vmult (VectorType          &dst,
                const VectorType    &src) const;
private:
    const SmartPointer<const Matrix>         matrix;
    const Preconditioner                     &preconditioner;
    const unsigned int                       iterations;
    const double                             rel_tolerance;
};


template <class Matrix, class Preconditioner>
InverseMatrix<Matrix,Preconditioner>::
InverseMatrix (const Matrix &m,
               const Preconditioner &preconditioner,
               const unsigned int   _iterations,
               const double         _rel_tolerance )
    :
      matrix (&m),
      preconditioner (preconditioner),
      iterations (_iterations),
      rel_tolerance(_rel_tolerance)
{}


template <class Matrix, class Preconditioner>
template <typename VectorType>
void InverseMatrix<Matrix,Preconditioner>::
vmult (VectorType       &dst,
       const VectorType &src) const
{
    double abs_tolerance = src.l2_norm()*rel_tolerance;
    //std::cout<< "Amg tol = " <<abs_tolerance << std::endl;
    //dst.reinit(src);
    dst = 0;
    SolverControl solver_control (iterations, abs_tolerance);
    if(InnerSolverData::solver_type == INNER_SOLVER_CG)
    {
        SolverCG<VectorType> cg(solver_control);
        try
        {
            //PreconditionIdentity identity; 
            cg.solve (*matrix, dst, src, preconditioner);
            InnerSolverData::amg_iterations += solver_control.last_step();
            //std::cout<<" cg iterations:  "<< solver_control.last_step();
        }

        catch (std::exception &e)
        {
            Assert (false, ExcMessage(e.what()));
        }
    }


    else if (InnerSolverData::solver_type == INNER_SOLVER_GMRES)
    {
        std::cout<<"H is spd, dont use gmres as inner solver!!"<<std::endl;

//        SolverFGMRES<VectorType> gmres(solver_control,
//                                       SolverFGMRES<TrilinosWrappers::Vector>::AdditionalData(1000)
//                                       );
//        try
//        {

//            gmres.solve (*matrix, dst, src, preconditioner);
//            InnerSolverData::amg_iterations += solver_control.last_step();
//            //std::cout<<solver_control.last_step()<<std::endl;
//        }

//        catch (std::exception &e)
//        {
//            Assert (false, ExcMessage(e.what()));
//        }
    }

    else //(InnerSolverData::solver_type == INNER_SOLVER_DIRECT)
    {
        TrilinosWrappers::SolverDirect  direct(solver_control);
        try
        {
            direct.solve (*matrix, dst, src);
        }
        catch (std::exception &e)
        {
            Assert (false, ExcMessage(e.what()));
        }
    }
}


template <class Preconditioner>
class NewtonSystemPreconditioner : public Subscriptor
{

public:
    NewtonSystemPreconditioner (const TrilinosWrappers::SparseMatrix  &sbetaK,
                                const InverseMatrix<TrilinosWrappers::SparseMatrix, Preconditioner> &amg1 );

    void vmult (TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const;

private:
    const SmartPointer<const TrilinosWrappers::SparseMatrix>         stiff_matrix;
    const SmartPointer<const InverseMatrix<TrilinosWrappers::SparseMatrix,
    Preconditioner> >                                                amg1;

    mutable TrilinosWrappers::MPI::Vector       tmp1,w,z;
};



template <class Preconditioner>
NewtonSystemPreconditioner<Preconditioner>::
NewtonSystemPreconditioner(const TrilinosWrappers::SparseMatrix  &sbetaK,
                           const InverseMatrix<TrilinosWrappers::SparseMatrix, Preconditioner> &amg1)
    :
      stiff_matrix          (&sbetaK),
      amg1                  (&amg1),
      tmp1                  (complete_index_set(stiff_matrix->m())),
      w                     (complete_index_set(stiff_matrix->m())),
      z                     (complete_index_set(stiff_matrix->m()))
{}


template <class Preconditioner>
void NewtonSystemPreconditioner<Preconditioner>::vmult(TrilinosWrappers::MPI::BlockVector &dst,
                                                       const TrilinosWrappers::MPI::BlockVector &src) const
{
    //1: solve Hz = x1 + x2
    tmp1 = src.block(0);
    tmp1 -= src.block(1);
    amg1->vmult(z,tmp1);
    //2: w=-sbeta*K*z + x2
    stiff_matrix->vmult(w,z);
    w*=-1;
    w+=src.block(1);
    //3: solve H*y2 = w
    amg1->vmult(dst.block(1), w);
    //4: compute y1 = z+y2
    dst.block(0)=z;
    dst.block(0)+=dst.block(1);
}


class ParameterReader : public Subscriptor
{
public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string);
private:
    void declare_parameters();
    ParameterHandler &prm;
};


ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    :
      prm(paramhandler)
{}


void ParameterReader::declare_parameters()
{
    prm.enter_subsection("Mesh");
    {
        prm.declare_entry("Refinement steps","1",
                          Patterns::Integer(1),
                          "Number of refinement steps");
        prm.declare_entry("Initial refinements", "4",
                          Patterns::Integer(0),
                          "Number of initial global refinements");
    }
    prm.leave_subsection();

    prm.enter_subsection("Equation parameters");
    {
        prm.declare_entry("Beta", "1e-2",
                          Patterns::Double(0),
                          "Beta");

        prm.declare_entry("Epsilon", "1e-6",
                          Patterns::Double(0),
                          "Epsilon");

        prm.declare_entry("Num epsilon", "1",
                          Patterns::Integer(1),
                          "Number of different epsilon values");

        prm.declare_entry("Num beta", "1",
                          Patterns::Integer(1),
                          "Number of different beta values");

        prm.declare_entry("Y upper","0.1",
                          Patterns::Double(),
                          "Upper constraint on state");

        prm.declare_entry("Y lower","-1000",
                          Patterns::Double(),
                          "Lower constraint on state");
        prm.declare_entry("Boundary type", "0",
                          Patterns::Integer(0,1),
                          "Type of boundary condition, Dirichlet=0, Neumann=1");
        prm.declare_entry("Sigma", "1",
                          Patterns::Double(0),
                          "Parameter sigma used in preconditioner");

        //Added: entries related to picking a function for the desired state
        prm.declare_entry("Problem type for desired state", "1",
                          Patterns::Integer(1),
                          "desired state, 1 for sin(pi*x1*x2), 2 for sin(k1*pi*x1-cos(k2*pi*x2))");
        prm.declare_entry("k1","8",
                          Patterns::Double(),
                          "factor 1");
        prm.declare_entry("k2","5",
                          Patterns::Double(),
                          "factor 2");

    }
    prm.leave_subsection();

    prm.enter_subsection("Inner solver");
    {
        prm.declare_entry("Inner solver type", "0",
                          Patterns::Integer(0,2),
                          "type of inner solver, CG=0,GMRES=1,DIRECT=2");
    }
    prm.leave_subsection();

    //Added: subsection and entries for linear solver
    prm.enter_subsection("Linear solver");
    {
        prm.declare_entry("Linear solver type", "1",
                          Patterns::Integer(0,1),
                          "type of linear solver, MINRES=0, GMRES=1");
        prm.declare_entry("Linear solver max it", "1000",
                          Patterns::Integer(0),
                          "the maximum number of iterations for the linear solver");
        prm.declare_entry("Linear solver fixed tolerance on", "false",
                          Patterns::Bool(),
                          "use a supplied tol for the linear solver, true/false (if false, a proportional tol is used)");
        prm.declare_entry("Linear solver fixed tolerance", "0.0001",
                          Patterns::Double(0),
                          "constant tolerance for the linear solver");
        prm.declare_entry("Linear solver fixed tolerance reduction", "0.1",
                          Patterns::Double(0),
                          "linear solver tolerance reduction used when the number of linear iterations were insufficient (<=2), value");
    }
    prm.leave_subsection();

    prm.enter_subsection("Nonlinear solver");
    {
        prm.declare_entry("Max newton iterations", "50",
                          Patterns::Integer(1,500),
                          "Maximum number of nonlinear iterations allowed");

        //Added: choice to output results for each nonlinear iteration
        prm.declare_entry("Save intermediate solution on", "false",
                         Patterns::Bool(),
                         "Save the intermediate solutions of the nonlinear solver (true = save solution every nonlinear step");
    }
    prm.leave_subsection();
}

void ParameterReader::read_parameters (const std::string parameter_file)
{
    declare_parameters();
    prm.read_input (parameter_file);
}

template<int dim>
class PoissonBox
{
public:
    PoissonBox(ParameterHandler &);
    void run();

private:
    void make_grid_and_dofs();
    void assemble_constant_system();
    void update_active_set_and_newton_system();
    void initialize_preconditioner();
    void solve_newton_system();     // linear solver
    void solve_newton();            // nonlinear solver
    void output_results() const;
    void push_table_data();
    void write_matrix(TrilinosWrappers::SparseMatrix &M, std::string filename);
    void write_vector(const TrilinosWrappers::MPI::Vector &V, std::string filename );

    //Added: output intermediate results
    void output_intermediate_results (int nonlin_step) const;

    ParameterHandler                            &prm;
    Triangulation<dim>                          triangulation;
    FESystem<dim>                               fe;
    DoFHandler<dim>                             dof_handler;
    TrilinosWrappers::BlockSparsityPattern      sparsity_pattern;
    TrilinosWrappers::BlockSparseMatrix         constant_system_matrix;
    TrilinosWrappers::BlockSparseMatrix         newton_system_matrix;
    TrilinosWrappers::BlockSparseMatrix         newton_preconditioner_matrix;
    TrilinosWrappers::MPI::BlockVector          solution;
    TrilinosWrappers::MPI::BlockVector          residual_vec;
    //Commented: unused variable
    //TrilinosWrappers::MPI::BlockVector          prec_residual_vec;
    TrilinosWrappers::MPI::BlockVector          constant_system_rhs;
    TrilinosWrappers::MPI::BlockVector          newton_system_rhs;
    TrilinosWrappers::MPI::BlockVector          desired_state;
    TrilinosWrappers::MPI::Vector               D_a;

    ConstraintMatrix                            constraints;
    ConstraintMatrix                            pre_constraints;
    IndexSet                                    active_set;
    IndexSet                                    active_set_upper;
    IndexSet                                    active_set_lower;

    //Added: variable for the additional NL stopping criterion
    bool                                        maybe_nl_conv;

    std::vector<std::string>                    input_file_names;

    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>  amg_preconditioner1;

    std::vector<types::global_dof_index>        system_block_sizes;

    unsigned int                                newton_iter;
    unsigned int                                gmres_iter;
    unsigned int                                refs;

    std::vector< MiscUtilities::TableData>      table_data_list;

    Timer                                       timer;
    double                                      solve_time;
};


template<int dim>
PoissonBox<dim>::PoissonBox(ParameterHandler &param)
    :
      prm(param),
      fe(FE_Q<dim>(1),2),
      dof_handler(triangulation)
{}


// Write matrices to matlab with double precision
template <int dim>
void
PoissonBox<dim>::write_matrix(TrilinosWrappers::SparseMatrix &M, std::string filename )
{

    std::string name = "data_" + filename + ".dat";
    FILE *fp;
    fp = fopen(name.c_str(),"wb");

    int n   = M.n();
    int m   = M.m();
    int nnz = M.n_nonzero_elements();

    Epetra_CrsMatrix mtr = M.trilinos_matrix();

    fwrite(&n, sizeof(int), 1, fp);
    fwrite(&m, sizeof(int), 1, fp);
    fwrite(&nnz, sizeof(int), 1, fp);

    fwrite(mtr.ExpertExtractIndexOffset().Values(), sizeof(int), m+1, fp);
    fwrite(mtr.ExpertExtractIndices().Values(), sizeof(int), nnz, fp);
    fwrite(mtr.ExpertExtractValues(), sizeof(double), nnz, fp);

    fclose(fp);
}

//write vectors to matlab with double precision
template<int dim>
void
PoissonBox<dim>::write_vector(const TrilinosWrappers::MPI::Vector &V, std::string filename ){
    std::string name = "data_" + filename + ".dat";
    FILE *fp;
    fp = fopen(name.c_str(),"wb");

    int n   = V.size();

    Epetra_MultiVector vtr = V.trilinos_vector();

    fwrite(&n, sizeof(int), 1, fp);

    fwrite(vtr.Values(), sizeof(double), n, fp);

    fclose(fp);
}


template <int dim>
void PoissonBox<dim>::make_grid_and_dofs()
{
    std::cout<<"making grid and dofs"<<std::endl;
    std::cout<<"# active cells "<< triangulation.n_active_cells()<< std::endl;
    dof_handler.distribute_dofs(fe);
    std::cout<<"dofs: "<<dof_handler.n_dofs()<<std::endl;
    active_set.clear();
    active_set_upper.clear();
    active_set_lower.clear();
    active_set_upper.set_size (dof_handler.n_dofs());    //initialize active set
    active_set_lower.set_size (dof_handler.n_dofs());    //initialize active set

    /*----------------------------- Set boundary indicator ------------------------------*/
    // shouldn't need to do this? Boundary function is zero on where b.c set to zero
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->center()(0) < 1e-12 || cell->face(f)->center()(1)< 1e-12){
                cell->face(f)->set_boundary_id(1);
            }
    /*-----------------------------------------------------------------------------------*/

    /*--------------renumber system after components----------------------*/
    std::vector<unsigned int> pde_sub_blocks (2,0);
    pde_sub_blocks[0]=0;
    pde_sub_blocks[1]=1;
    DoFRenumbering::component_wise (dof_handler, pde_sub_blocks);
    std::vector<types::global_dof_index> dofs_per_component (2);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

    const unsigned int n_y = dofs_per_component[0];
    const unsigned int n_u = dofs_per_component[1];
    std::cout<<"dofs per component: y-"<<n_y<<" u-"<<n_u<<std::endl;

    system_block_sizes.resize (2);
    system_block_sizes[0] = n_y;
    system_block_sizes[1] = n_u;
    /*--------------------------------------------------------------------*/

    /*-----------------------Dirichlet Boundary conditions----------------------------*/
    const FEValuesExtractors::Scalar ys(0);
    const FEValuesExtractors::Scalar us(1);

    constraints.clear ();
    if(EquationData::bc==DIRICHLET_BC)
    {
        VectorTools::interpolate_boundary_values (dof_handler,0,
                                                  EquationData::BoundaryValues<dim>(2) ,
                                                  constraints,
                                                  fe.component_mask(ys));

        //Added: dirichlet BC for desired_state version 2 on boundary 1, else {} contains boundary 1 DBC for the original (desired state version 1)
        if(EquationData::desired_state==2)
            VectorTools::interpolate_boundary_values (dof_handler,1,
                                                      EquationData::BoundaryValues<dim>(2),
                                                      constraints,
                                                      fe.component_mask(ys));
        else //Original version: (now desired_state 1)
            VectorTools::interpolate_boundary_values (dof_handler,1,
                                                      ZeroFunction<dim>(2),
                                                      constraints,
                                                      fe.component_mask(ys));
    }

    VectorTools::interpolate_boundary_values (dof_handler,0,
                                              ZeroFunction<dim>(2),
                                              constraints,
                                              fe.component_mask(us));

    VectorTools::interpolate_boundary_values (dof_handler,1,
                                              ZeroFunction<dim>(2),
                                              constraints,
                                              fe.component_mask(us));
    constraints.close ();
    /*--------------------------------------------------------------------------------*/


    /*-------------------------make sparsity pattern---------------------------*/
    constant_system_matrix.clear ();
    sparsity_pattern.reinit (2,2);
    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();
    std::cout<<"couplings: "<< n_couplings << std::endl;
    sparsity_pattern.block(0,0).reinit (n_y, n_y, n_couplings);
    sparsity_pattern.block(1,0).reinit (n_u, n_y, n_couplings);
    sparsity_pattern.block(0,1).reinit (n_y, n_u, n_couplings);
    sparsity_pattern.block(1,1).reinit (n_u, n_u, n_couplings);
    sparsity_pattern.collect_sizes();
    DoFTools::make_sparsity_pattern (dof_handler,
                                     sparsity_pattern,
                                     constraints, true);
    sparsity_pattern.compress();
    /*-------------------------------------------------------------------------*/

    /*---------------init vectors and matrices---------------*/
    constant_system_matrix.reinit(sparsity_pattern);
    newton_system_matrix.reinit(sparsity_pattern);
    newton_preconditioner_matrix.reinit(sparsity_pattern);
    D_a.reinit(complete_index_set(n_y));

    constant_system_rhs.reinit(2);
    constant_system_rhs.block(0).reinit (complete_index_set(n_y));
    constant_system_rhs.block(1).reinit (complete_index_set(n_u));
    constant_system_rhs.collect_sizes ();
    newton_system_rhs.reinit(2);
    newton_system_rhs.block(0).reinit (complete_index_set(n_y));
    newton_system_rhs.block(1).reinit (complete_index_set(n_u));
    newton_system_rhs.collect_sizes ();
    solution.reinit (2);
    solution.block(0).reinit (complete_index_set(n_y));
    solution.block(1).reinit (complete_index_set(n_u));
    solution.collect_sizes ();
    residual_vec.reinit(2);
    residual_vec.block(0).reinit(complete_index_set(n_y));
    residual_vec.block(1).reinit(complete_index_set(n_u));
    residual_vec.collect_sizes();
    //Commented: unused variable
//    prec_residual_vec.reinit(2);
//    prec_residual_vec.block(0).reinit(complete_index_set(n_y));
//    prec_residual_vec.block(1).reinit(complete_index_set(n_u));
//    prec_residual_vec.collect_sizes();
    /*-------------------------------------------------------*/

    /*---------- construct desired state vector (only used for output) -----------------*/
    desired_state.reinit(2);
    desired_state.block(0).reinit (complete_index_set(n_y));
    desired_state.block(1).reinit (complete_index_set(n_u));
    desired_state.collect_sizes();
    const EquationData::ControlValues<dim> control_values(2);
    std::vector<bool> dof_touched (dof_handler.n_dofs(), false);
    //loop over all cells
    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell)
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const unsigned int dof_index = cell->vertex_dof_index (v,0);
            //make sure dofs are only touched once
            if (dof_touched[dof_index] == false)
            {
                dof_touched[dof_index] = true;
            }
            else{ continue; }
            desired_state(dof_index) = control_values.value(cell->vertex(v),0);
        }
    /*--------------------------------------------------------------------------*/
}



template<int dim>
void PoissonBox<dim>::assemble_constant_system()
{
    std::cout<<"assembling system"<<std::endl;
    double sbeta = std::sqrt(EquationData::beta);
    constant_system_matrix = 0;
    constant_system_rhs = 0;
    D_a = 0;
    const FEValuesExtractors::Scalar ys(0);
    const FEValuesExtractors::Scalar us(1);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;

    FullMatrix<double>
            local_00_matrix(dofs_per_cell, dofs_per_cell),

            local_11_matrix(dofs_per_cell, dofs_per_cell),

            local_10_matrix(dofs_per_cell, dofs_per_cell),
            local_01_matrix(dofs_per_cell, dofs_per_cell),
            local_matrix(dofs_per_cell, dofs_per_cell),
            local_pre_matrix(dofs_per_cell,dofs_per_cell);

    Vector<double>      local_rhs (dofs_per_cell);

    QGauss<dim>     quadrature_formula_stiffness (3); // 9 point/cell for stiffness matrix
    QTrapez<dim>    quadrature_formula_mass;    // Trapezoidal quadrature rule for lumped mass matrix
    FEValues<dim>   fe_values_mass (fe, quadrature_formula_mass,
                                    update_values    | update_gradients |
                                    update_quadrature_points |update_JxW_values);

    FEValues<dim>   fe_values_stiffness(fe, quadrature_formula_stiffness,
                                        update_values    | update_gradients |
                                        update_quadrature_points |update_JxW_values);

    const unsigned int   n_q_points_mass        = quadrature_formula_mass.size();
    const unsigned int   n_q_points_stiffness   = quadrature_formula_stiffness.size();
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    double y_phi_i, u_phi_i, y_phi_j, u_phi_j;

    Tensor<1,dim>    y_grad_phi_i, u_grad_phi_i,
            y_grad_phi_j, u_grad_phi_j;

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    const EquationData::ControlValues<dim> control_values(2);
    std::vector<double>     local_control_values (n_q_points_stiffness);

    for (; cell!=endc; ++cell)
    {
        fe_values_mass.reinit (cell);
        fe_values_stiffness.reinit(cell);
        std::vector<Point<dim>> p_list=fe_values_stiffness.get_quadrature_points();
        control_values.value_list (p_list, local_control_values);

        local_00_matrix=0;
        local_01_matrix=0;
        local_11_matrix=0;
        local_10_matrix=0;
        local_matrix=0;
        local_rhs = 0;


        /*------------------- assemble stiffness  matrix--------------------------- */
        for(unsigned int q=0; q<n_q_points_stiffness; ++q)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                y_grad_phi_i         = fe_values_stiffness[ys].gradient (i, q);
                u_grad_phi_i         = fe_values_stiffness[us].gradient (i, q);
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    y_grad_phi_j = fe_values_stiffness[ys].gradient(j, q);
                    u_grad_phi_j = fe_values_stiffness[us].gradient(j, q);

                    local_01_matrix(i, j) =
                            (y_grad_phi_i * u_grad_phi_j)*fe_values_stiffness.JxW(q);
                    local_10_matrix(i, j) =
                            (u_grad_phi_i * y_grad_phi_j)*fe_values_stiffness.JxW(q);
                    local_matrix(i,j) +=
                            sbeta*local_10_matrix(i, j) - sbeta*local_01_matrix(i, j);
                }
            }
        }
        /*--------------------------------------------------------------------------*/

        /*----------------------- assemble lumped mass matrix -----------------------*/
        for (unsigned int q=0; q<n_q_points_mass; ++q)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                y_phi_i         = fe_values_mass[ys].value (i, q);
                u_phi_i         = fe_values_mass[us].value (i, q);
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    y_phi_j      = fe_values_mass[ys].value (j, q);
                    u_phi_j      = fe_values_mass[us].value (j, q);

                    local_00_matrix(i, j) = (y_phi_i * y_phi_j)*fe_values_mass.JxW(q);
                    local_11_matrix(i, j) = (u_phi_i * u_phi_j)*fe_values_mass.JxW(q);

                    if( EquationData::bc==NEUMANN_BC )
                    {
                        // with Neumann conditions K=K+M
                        local_01_matrix (i, j) +=
                                y_phi_i * u_phi_j * fe_values_mass.JxW(q);
                        local_10_matrix(i, j) +=
                                u_phi_i * y_phi_j * fe_values_mass.JxW(q);
                    }
                    local_matrix(i,j) += local_00_matrix(i, j)+local_11_matrix(i, j);
                }
                local_rhs(i) += (local_00_matrix(i,i)*local_control_values[q]);
            }
        }
        /*-----------------------------------------------------------------------------*/

        /*-------- Distribute from local to global matrix -------------*/
        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (local_matrix,
                                                local_rhs,
                                                local_dof_indices,
                                                constant_system_matrix,
                                                constant_system_rhs
                                                );
        /*-------------------------------------------------------------*/
    }

    constant_system_matrix.compress(VectorOperation::add); // is this only for MPI?

}


template<int dim>
void PoissonBox<dim>::update_active_set_and_newton_system()
{
    std::cout << "   Updating active set..." << std::endl;
    active_set_upper.clear ();
    active_set_lower.clear ();
    std::vector<bool> dof_touched (dof_handler.n_dofs(), false);

    // reinitialize the newton system to the constant system
    newton_system_matrix.copy_from(constant_system_matrix);
    newton_system_rhs = constant_system_rhs;

    // reinitialize preconditioner matrix H = M+sbetaK
    newton_preconditioner_matrix.block(0,0).copy_from(constant_system_matrix.block(0,0));
    newton_preconditioner_matrix.block(0,0).add(1,constant_system_matrix.block(1,0));

    // loop over all cells touching each dof once
    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell)
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            //get dof index of state variable
            const unsigned int dof_index = cell->vertex_dof_index (v,0);

            //make sure dofs are only touched once
            if (dof_touched[dof_index] == false)
            {
                dof_touched[dof_index] = true;
            }
            else{ continue; }

            if (solution.block(0)[dof_index] > EquationData::y_upper)
            {
                active_set_upper.add_index(dof_index);
                // set rhs of newton system
                newton_system_rhs(dof_index)+=constant_system_matrix(dof_index,dof_index)*EquationData::y_upper*1/EquationData::epsilon;
                //update L
                newton_system_matrix.add(dof_index, dof_index, constant_system_matrix(dof_index,dof_index)*1/EquationData::epsilon);
                newton_preconditioner_matrix.add(dof_index,dof_index, constant_system_matrix(dof_index,dof_index)*EquationData::sigma*1/EquationData::epsilon);
                D_a(dof_index)=1;
            }

            // check lower constraint
            else if (solution.block(0)[dof_index] < EquationData::y_lower)
            {
                active_set_lower.add_index(dof_index);
                //set rhs
                newton_system_rhs(dof_index)+= constant_system_matrix(dof_index,dof_index)*EquationData::y_lower*1/EquationData::epsilon;
                //update L
                newton_system_matrix.add(dof_index,dof_index, constant_system_matrix(dof_index,dof_index)*1/EquationData::epsilon);
                newton_preconditioner_matrix.add(dof_index,dof_index, constant_system_matrix(dof_index,dof_index)*EquationData::sigma*1/EquationData::epsilon);
                D_a(dof_index)=1;
            }
            else
            {
                D_a(dof_index)=0;
            }
        }

    std::cout<<"size of (upper)active set:                 "<< active_set_upper.n_elements() <<std::endl;


    /*------------------write matrices to file------------------------*/
//        std::string stiffName="sbetaK"+ std::to_string(newton_iter);
//        write_matrix(newton_system_matrix.block(1,0),stiffName);

//        std::string HName="H"+ std::to_string(newton_iter);
//        write_matrix(newton_preconditioner_matrix.block(0,0),HName);

//        std::string H2Name="H2"+ std::to_string(newton_iter);
//        write_matrix(newton_preconditioner_matrix.block(1,1),H2Name);

//        std::string MName="M"+std::to_string(newton_iter);
//        write_matrix(newton_system_matrix.block(1,1),MName);

//        std::string LName="L"+std::to_string(newton_iter);
//        write_matrix(newton_system_matrix.block(0,0),LName);

//        std::string DName="Dh"+std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(D_a,DName);

//        std::string rhsName1="1rhsh"+std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(newton_system_rhs.block(0),rhsName1);

//        std::string rhsName2="2rhsh"+std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(newton_system_rhs.block(1),rhsName2);
    /*----------------------------------------------------------------*/
}

template <int dim>
void PoissonBox<dim>::initialize_preconditioner()
{
    amg_preconditioner1 = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> (new TrilinosWrappers::PreconditionAMG());
    //amg_preconditioner2 = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> (new TrilinosWrappers::PreconditionAMG());
    TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data1;
    Amg_data1.elliptic = true;
    Amg_data1.n_cycles=4;
    Amg_data1.smoother_sweeps =2 ;
    Amg_data1.aggregation_threshold = 1e-4; // this may need tuning
    Amg_data1.smoother_type ="ML Gauss-Seidel";
    amg_preconditioner1->initialize(newton_preconditioner_matrix.block(0,0), Amg_data1);
}

template<int dim>
void PoissonBox<dim>::solve_newton_system()
{
    std::cout<<"solving linear system"<<std::endl;
    //initialize_preconditioner();
    PreconditionIdentity                identity; // = no preconditioner

    //Added: to be able to use the same initial guess if we end up in solve again
    TrilinosWrappers::MPI::BlockVector          solution_old(solution);

    //Added: case for minres
    if (LinearSolverData::solver_type == LINEAR_SOLVER_MINRES)
    {

//        /*-------------------Create preconditioner but just to check time difference------------------------------*/
//        initialize_preconditioner();
//        // approximate inverse for preconditioner matrix H
//        const InverseMatrix<TrilinosWrappers::SparseMatrix, TrilinosWrappers::PreconditionAMG> h_inverse (newton_preconditioner_matrix.block(0,0),
//                                                                                                          *amg_preconditioner1,
//                                                                                                          InnerSolverData::max_iter,
//                                                                                                          InnerSolverData::rel_tolerance);
//        // Create the preconditioner
//        const NewtonSystemPreconditioner<TrilinosWrappers::PreconditionAMG> precondi (newton_system_matrix.block(1,0),
//                                                                                            h_inverse);

        /*-------------------Create preconditioner------------------------------*/
        PreconditionIdentity preconditioner;
        /*----------------------------------------------------------------------*/

        /*-------calculate absolute tolerance for MINRES solver--------------*/

        newton_system_matrix.vmult(residual_vec, solution);
        residual_vec -= newton_system_rhs;
        //Commented: unnecessary variable
//        preconditioner.vmult(prec_residual_vec, residual_vec);
        double absolute_tolerance = residual_vec.l2_norm()*1e-6;
        //Commented:
        //std::cout<<"MINRES tolerance - "<<absolute_tolerance<<std::endl;
        //std::cout<<"rhs l2norm - "<<newton_system_rhs.l2_norm()<<std::endl;

        //Added:
        std::cout<<"residual l2norm: "<<absolute_tolerance*1e6<<std::endl;
        std::cout<<"rhs l2norm: "<<newton_system_rhs.l2_norm()<<std::endl;


        //Added: case for fixed tolerance
        double tol = 0;
        if (LinearSolverData::fixed_tolerance_on)
        {
            tol =LinearSolverData::fixed_tolerance;
            std::cout<<"using fix MINRES tolerance: "<<tol<<std::endl;
        }
        else
        {
            tol =absolute_tolerance;
            std::cout<<"calculated MINRES absolute tolerance: "<<absolute_tolerance<<std::endl;
        }

        /*-------------------------------------------------------------------*/


        //Added: max iter for minres
        LinearSolverData::max_iter = dof_handler.n_dofs()*2;
        
        /*------------------------------------setup solver and solve -----------------------------------------*/
        //Commented: solver control
        //SolverControl                                   solver_control (1000, absolute_tolerance, true, true);
        //Added: solver control
        SolverControl                                   solver_control (LinearSolverData::max_iter, tol, true, true);
        SolverMinRes<TrilinosWrappers::MPI::BlockVector>
                solver(solver_control);
        solver_control.enable_history_data();

        solver.solve(newton_system_matrix, solution, newton_system_rhs, preconditioner);
        gmres_iter += solver_control.last_step();

        /*------------------ Write residual reduction to file----------------------------------*/
//        std::ofstream residual_file;
//        std::string residual_file_name = "residualsPII/res_b"
//                +std::to_string((int) std::log10(EquationData::beta))
//                +"_e"+std::to_string((int) std::log10(EquationData::epsilon))
//                +"_s"+std::to_string(EquationData::sigma)
//                +"_h"+std::to_string(refs)
//                +"_it"+std::to_string(newton_iter);
//        residual_file.open(residual_file_name);
//        residual_file << residual_vec.l2_norm() << "\n";
//        for(unsigned int i=1; i<=solver_control.last_step(); i++)
//        {
//            residual_file << solver_control.step_reduction(i) << " ";
//        }
        /*-------------------------------------------------------------------------------------*/


        // check if sufficient number of iterations. If not, solve again.
        if (solver_control.last_step() <= 2)
        {
            std::cout<<"insufficient number of iterations, solve again!"<<std::endl;
            //Commented: calculating new absolute tolerance not used anymore
            /*-------calculate new absolute tolerance for MINRES solver--------------*/
            //newton_system_matrix.vmult(residual_vec, solution);
            //residual_vec -= newton_system_rhs;
            //preconditioner.vmult(prec_residual_vec, residual_vec);
            //absolute_tolerance = residual_vec.l2_norm()*1e-3;

            //Added: reduce tolerance with the factor specified in fixed_tolerance_reduction
            tol*=LinearSolverData::fixed_tolerance_reduction;
            std::cout<<"minres rerun using tol: "<<tol<<std::endl;

            //Added: To get the same initial guess for this linear solve
            solution_old.swap(solution);

            //solver_control.set_tolerance(absolute_tolerance);
            solver_control.set_tolerance(tol);
            solver.solve(newton_system_matrix, solution, newton_system_rhs, preconditioner);
            gmres_iter += solver_control.last_step();

            //Added: if solver has <=2it again, check if we're actually done before starting something that oscillates
            if(solver_control.last_step() <= 2)
                maybe_nl_conv = true;

            //Added: print "error" reduction
            std::cout<<"average 'error' reduction: "<<solver_control.average_reduction()<<std::endl;


            /*---------- write residual to file-------------------------- */
//            for(unsigned int i=1; i<=solver_control.last_step(); i++)
//            {
//                residual_file << solver_control.step_reduction(i) << " ";
//            }
            /*----------------------------------------------------------- */
        }
//        residual_file.close();

        constraints.distribute (solution);

        std::cout<<"                             MINRES iterations: "<<solver_control.last_step()<<std::endl;
        /*----------------------------------------------------------------------------------------------------*/
    }
    else //GMRES (original)
    {
        /*-------------------Create preconditioner------------------------------*/
        initialize_preconditioner();
        // approximate inverse for preconditioner matrix H
        const InverseMatrix<TrilinosWrappers::SparseMatrix, TrilinosWrappers::PreconditionAMG> h_inverse (newton_preconditioner_matrix.block(0,0),
                                                                                                          *amg_preconditioner1,
                                                                                                          InnerSolverData::max_iter,
                                                                                                          InnerSolverData::rel_tolerance);
        // Create the preconditioner
        const NewtonSystemPreconditioner<TrilinosWrappers::PreconditionAMG> preconditioner (newton_system_matrix.block(1,0),
                                                                                            h_inverse);
        /*----------------------------------------------------------------------*/


        /*-------calculate absolute tolerance for FGMRES solver--------------*/

        newton_system_matrix.vmult(residual_vec, solution);
        residual_vec -= newton_system_rhs;
        //Commented: unnecessary variable
//        preconditioner.vmult(prec_residual_vec, residual_vec);
        double absolute_tolerance = residual_vec.l2_norm()*1e-6;
        //Commented:
        //std::cout<<"fgmres tolerance - "<<absolute_tolerance<<std::endl;
        //std::cout<<"rhs l2norm - "<<newton_system_rhs.l2_norm()<<std::endl;
        /*-------------------------------------------------------------------*/

        //Added:
        std::cout<<"residual l2norm: "<<absolute_tolerance*1e6<<std::endl;
        std::cout<<"rhs l2norm: "<<newton_system_rhs.l2_norm()<<std::endl;


        //Added: case for fixed tolerance
        double tol;
        if (LinearSolverData::fixed_tolerance_on)
        {
            tol =LinearSolverData::fixed_tolerance;
            std::cout<<"using fix FGMRES tolerance: "<<tol<<std::endl;
        }
        else
        {
            tol =absolute_tolerance;
            std::cout<<"using calculated FGMRES absolute tolerance: "<<absolute_tolerance<<std::endl;
        }

        /*------------------------------------setup solver and solve -----------------------------------------*/
        //Commented: solver control
        //SolverControl                                   solver_control (1000, absolute_tolerance, true, true);
        //Added: solver control
        SolverControl                                   solver_control (1000, tol, true, true);
        SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
                solver(solver_control,
                       SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(1000));
        solver_control.enable_history_data();

        solver.solve(newton_system_matrix, solution, newton_system_rhs, preconditioner);
        gmres_iter += solver_control.last_step();

        //Added: cout avg error reduction
        std::cout<<"average 'error' reduction: "<<solver_control.average_reduction()<<std::endl;


        /*------------------ Write residual reduction to file----------------------------------*/
//        std::ofstream residual_file;
//        std::string residual_file_name = "residualsPII/res_b"
//                +std::to_string((int) std::log10(EquationData::beta))
//                +"_e"+std::to_string((int) std::log10(EquationData::epsilon))
//                +"_s"+std::to_string(EquationData::sigma)
//                +"_h"+std::to_string(refs)
//                +"_it"+std::to_string(newton_iter);
//        residual_file.open(residual_file_name);
//        residual_file << residual_vec.l2_norm() << "\n";
//        for(unsigned int i=1; i<=solver_control.last_step(); i++)
//        {
//            residual_file << solver_control.step_reduction(i) << " ";
//        }
        /*-------------------------------------------------------------------------------------*/


        // check if sufficient number of iterations. If not, solve again.
        if (solver_control.last_step() <= 2)
        {
            //Commented:
            //std::cout<<"solve again!"<<std::endl;

            //Added:
            std::cout<<"insufficient number of iterations, solve again!"<<std::endl;

            /*-------calculate new absolute tolerance for FGMRES solver--------------*/
            //Commented:
//            newton_system_matrix.vmult(residual_vec, solution);
//            residual_vec -= newton_system_rhs;
//            preconditioner.vmult(prec_residual_vec, residual_vec);
//            absolute_tolerance = residual_vec.l2_norm()*1e-3;   // Kolla 1e-3
//            solver_control.set_tolerance(absolute_tolerance);

            //Added: reduce tolerance with the factor specified in fixed_tolerance_reduction
            tol*=LinearSolverData::fixed_tolerance_reduction;
            std::cout<<"fgmres rerun using tol: "<<tol<<std::endl;
            solver_control.set_tolerance(tol);

            //Added: To get the same initial guess for this linear solve
            solution_old.swap(solution);

            solver.solve(newton_system_matrix, solution, newton_system_rhs, preconditioner);
            gmres_iter += solver_control.last_step();
            
            //Added: if solver has <=2it again, check if we're actually done before starting soething that oscillates
            if(solver_control.last_step() <= 2)
                maybe_nl_conv = true;

            //Added: print "error" reduction
            std::cout<<"average 'error' reduction: "<<solver_control.average_reduction()<<std::endl;

            /*---------- write residual to file-------------------------- */
//            for(unsigned int i=1; i<=solver_control.last_step(); i++)
//            {
//                residual_file << solver_control.step_reduction(i) << " ";
//            }
            /*----------------------------------------------------------- */
        }
//        residual_file.close();

        constraints.distribute (solution);

        std::cout<<"                             fgmres iterations: "<<solver_control.last_step()<<std::endl;
        /*----------------------------------------------------------------------------------------------------*/
    }
}

template<int dim>
void PoissonBox<dim>::solve_newton()
{
    gmres_iter = 0;
    InnerSolverData::amg_iterations = 0;
    prm.enter_subsection("Nonlinear solver");
    unsigned int max_newton_steps = prm.get_integer("Max newton iterations");
    prm.leave_subsection();
    IndexSet active_set_upper_old (active_set_upper);
    IndexSet active_set_lower_old (active_set_lower);

    //Added: variables used for the additional NL stopping criterion
    double h_quad =1; //just h^2
    for (unsigned int t = 1 ; t<=refs*2; t++)
        h_quad=h_quad/2;
    double dmin = h_quad+1; //to avoid additional NL stopping criterion first NL iter
    maybe_nl_conv=false;

    // Iterate Newton solver until active set has not changed
    timer.restart();
    for(unsigned int newton_step=1; newton_step <= max_newton_steps; ++newton_step)
    {
        newton_iter=newton_step;
        std::cout<<"Newton step "<<newton_step<<std::endl;
        update_active_set_and_newton_system();

        //check convergence by comparing active sets
        if(active_set_upper==active_set_upper_old &&
                active_set_lower==active_set_lower_old
                && newton_step!=1)
        {
            std::cout<<"active set did not change, convergence in newton step: "<< newton_step-1 <<std::endl;
            newton_iter = newton_step-1;
            //Added: break
            break;
        }
        else if(newton_step==max_newton_steps)
        {
            std::cout<<"newton method did not converge :("<<std::endl;
            newton_iter=newton_step;
            break;
        }
        //Added: additional stopping criterion for upper bound (not implemented for lower bound)
        else if(maybe_nl_conv && dmin<h_quad)
        {
            std::cout<<"Additional criterion met (dmin<h^2) in newton step: "<< newton_step-1<<")"<<std::endl;
            newton_iter = newton_step-1;
            break;
        }

        /*-----------------print initial guess to file--------------------------*/
//        std::string solfinName1="1solh" +std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(solution.block(0),solfinName1);

//        std::string solfinName2="2solh"+std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(solution.block(1),solfinName2);
        /*----------------------------------------------------------------------*/


        //solve newton system
        solve_newton_system();

        //Added: for the additional NL stopping criterion, measure the minimal distance between solution and upper bound in active points and calculate dmin
        if (newton_step != 1) //not to calculate unnecessary stuff
        {
            std::vector<unsigned int> active_ind;
            std::vector<double> active_val; //(active_set_upper_diff.n_elements());
            active_ind.resize(active_set_upper.n_elements());
            active_set_upper.fill_index_vector(active_ind);
            active_val.resize(active_set_upper.n_elements());
            solution.block(0).extract_subvector_to(active_ind, active_val);
            //active_val-=EquationData::y_upper;
            dmin = *std::min_element(active_val.begin(), active_val.end());
            //std::cout<<"-----------------------------------------min distance: "<< dmin <<std::endl;
            dmin -=EquationData::y_upper;
            double k=2;
            dmin = std::abs(dmin)/(k*h_quad);

            std::cout<<"--------------------------------------------------h^2: "<< h_quad <<std::endl;
            std::cout<<"-------------------------------------------------dmin: "<< dmin <<std::endl;

        }

        //Added: save intermediate solution
        if (NonLinearSolverData::save_intermediate_solution)
            output_intermediate_results(newton_step);

        /*----------------print final solution to file--------------------------*/
//        std::string solfinName1="1solh" +std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(solution.block(0),solfinName1);

//        std::string solfinName2="2solh"+std::to_string(refs)+"_"+std::to_string(newton_iter);
//        write_vector(solution.block(1),solfinName2);
        /*----------------------------------------------------------------------*/
        active_set_upper_old=active_set_upper;
        active_set_lower_old=active_set_lower;
    }

    solve_time = timer.wall_time();
    std::cout<<"solved in "<< solve_time << " seconds "<< std::endl;
}


//Added: output intermediate results
template <int dim>
void PoissonBox<dim>::output_intermediate_results (int nonlin_step) const
{
    std::cout<<"outputting results, nonlinear iteration "<<nonlin_step<<std::endl;
    std::vector<std::string> solution_names;
    solution_names.push_back("y_"+std::to_string(nonlin_step));
    solution_names.push_back("u_"+std::to_string(nonlin_step));
    std::vector<std::string> solution_names2;
    solution_names2.push_back("state_"+std::to_string(nonlin_step));
    solution_names2.push_back("nothing_"+std::to_string(nonlin_step));

    DataOut<dim>  data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector (solution,solution_names);
    data_out.add_data_vector(active_set_upper,"active_set_"+std::to_string(nonlin_step));

    data_out.add_data_vector(desired_state,solution_names2);
    data_out.build_patches ();
    std::ostringstream ssb;
    std::ostringstream sse;
    ssb << std::scientific << std::setprecision(2) << EquationData::beta;
    sse << std::scientific << std::setprecision(2) << EquationData::epsilon;
    std::string st;
    if (LinearSolverData::solver_type == 1)
        st="GMRES";
    else
        st="MINRES";
    std::ofstream output("solution_"+st+"_beta"+ssb.str()+"_epsilon"+sse.str()
                         +"_dofs"+std::to_string(dof_handler.n_dofs())+"_nonlin_step_"+std::to_string(nonlin_step)+".vtk");
    //std::ofstream output("solution_beta"+std::to_string(EquationData::beta)+"_epsilon"+std::to_string(EquationData::epsilon)
    //                     +"_dofs"+std::to_string(dof_handler.n_dofs())+"_nonlin_step_"+std::to_string(nonlin_step)+".vtk");
    data_out.write_vtk(output);
}

template <int dim>
void PoissonBox<dim>::output_results () const
{
    std::cout<<"outputting results"<<std::endl;
    std::vector<std::string> solution_names;
    solution_names.push_back("y");
    solution_names.push_back("u");
    std::vector<std::string> solution_names2;
    solution_names2.push_back("state");
    solution_names2.push_back("nothing");

    DataOut<dim>  data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector (solution,solution_names);
    data_out.add_data_vector(active_set_upper,"active_set");

    data_out.add_data_vector(desired_state,solution_names2);
    data_out.build_patches ();
    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);
}

template <int dim>
void PoissonBox<dim>::push_table_data()
{
    //Added: case for different linear solver
    if (LinearSolverData::solver_type==LINEAR_SOLVER_MINRES)
    {
        std::cout<<"total minres iterations: "<<gmres_iter<<std::endl;
        std::cout<<"average minres iterations: "<<gmres_iter/newton_iter<<std::endl;
    }
    else //Original
    {
    std::cout<<"total gmres iterations: "<<gmres_iter<<std::endl;
    std::cout<<"average gmres iterations: "<<gmres_iter/newton_iter<<std::endl;
    }
    //Commented:
    //std::cout<<"total gmres iterations: "<<gmres_iter<<std::endl;
    //std::cout<<"average gmres iterations: "<<gmres_iter/newton_iter<<std::endl;

    MiscUtilities::TableData td;
    td.dofs=dof_handler.n_dofs();
    td.beta=EquationData::beta;
    td.epsilon = EquationData::epsilon;
    td.h1_iterations     = InnerSolverData::amg_iterations/gmres_iter;
    std::cout<<"-------------------------------------------------------avg amg"<< td.h1_iterations << std::endl;
    std::cout<<"-------------------------------------------------------tot amg"<< InnerSolverData::amg_iterations << std::endl;
    std::cout<<"-------------------------------------------------------tot lin"<<gmres_iter << std::endl;
    td.gmres_iterations  = gmres_iter/newton_iter;
    td.newton_iterations = newton_iter;
    td.refinements = refs;
    td.time=solve_time;
    table_data_list.push_back(td);
}



template<int dim>
void PoissonBox<dim>::run()
{
    //Added: sections Linear and Nonlinear solver
    prm.enter_subsection("Linear solver");
    LinearSolverData::solver_type= prm.get_integer("Linear solver type");
    LinearSolverData::max_iter= prm.get_integer("Linear solver max it");
    LinearSolverData::fixed_tolerance_on= prm.get_bool("Linear solver fixed tolerance on");
    LinearSolverData::fixed_tolerance= prm.get_double("Linear solver fixed tolerance");
    LinearSolverData::fixed_tolerance_reduction= prm.get_double("Linear solver fixed tolerance reduction");
    prm.leave_subsection();
    prm.enter_subsection("Nonlinear solver");
    NonLinearSolverData::save_intermediate_solution = prm.get_bool("Save intermediate solution on");
    prm.leave_subsection();


    /*--------------------get some parameters here----------------------*/
    prm.enter_subsection("Inner solver");
    InnerSolverData::solver_type= prm.get_integer("Inner solver type");
    prm.leave_subsection();

    prm.enter_subsection("Mesh");
    unsigned int n_refinements = prm.get_integer("Refinement steps") ;
    unsigned int initial_refinements = prm.get_integer("Initial refinements");
    prm.leave_subsection();
    /*-------------------------------------------------------------------*/

    prm.enter_subsection("Equation parameters");
    unsigned int n_epsilon  = prm.get_double("Num epsilon");
    unsigned int n_beta     = prm.get_double("Num beta");
    EquationData::y_upper   = prm.get_double("Y upper");
    EquationData::y_lower   = prm.get_double("Y lower");
    EquationData::bc        = prm.get_integer("Boundary type");

    //Added: set parameters for desired_state with and coefficients
    EquationData::desired_state        = prm.get_integer("Problem type for desired state");
    EquationData::k1   = prm.get_double("k1");
    EquationData::k2   = prm.get_double("k2");

    prm.leave_subsection();
    // solve system for all values of beta and epsilon and all grid sizes
    for(unsigned int i=0; i<n_beta; ++i)
    {
        if(i==0)
        {
            prm.enter_subsection("Equation parameters");
            EquationData::beta = prm.get_double("Beta");
            EquationData::sigma = prm.get_double("Sigma");
            prm.leave_subsection();

        }
        else
            EquationData::beta = EquationData::beta/10.0;

        for(unsigned int j=0; j<n_epsilon; ++j)
        {
            //Vectors to store solution for each grid
            //std::vector<TrilinosWrappers::MPI::BlockVector *> solution_lvl;

            if(j==0){
                prm.enter_subsection("Equation parameters");
                EquationData::epsilon = prm.get_double("Epsilon");
                prm.leave_subsection();
            }
            else
                EquationData::epsilon = EquationData::epsilon/10.0;
            
            for(unsigned int k=0; k<n_refinements; ++k)
            {
                refs = initial_refinements+k;
                std::cout<<"beta: " <<EquationData::beta<<std::endl;
                std::cout<<"epsilon: " <<EquationData::epsilon<<std::endl;
                prm.enter_subsection("Equation parameters");
                EquationData::sigma = prm.get_double("Sigma");
                prm.leave_subsection();
                if (EquationData::sigma == 0)
                {
                    double h = std::pow(2,-1.0*refs);
                    double h2 = std::pow(h,2);
                    double seps = std::sqrt(EquationData::epsilon);
                    EquationData::sigma = 1/(h2*seps);
                }
                std::cout<<"sigma: " << EquationData::sigma<<std::endl;
                //transfer solution from previous mesh(if previous solution exists)
                if(k>0)
                {
                    SolutionTransfer<dim, TrilinosWrappers::MPI::BlockVector> soltrans(dof_handler);
                    triangulation.prepare_coarsening_and_refinement(); // seems this is not needed when doing global refinement?
                    soltrans.prepare_for_pure_refinement();
                    TrilinosWrappers::MPI::BlockVector solution_old;
                    solution_old.reinit(solution);
                    solution_old = solution;
                    triangulation.refine_global(1);
                    make_grid_and_dofs();
                    soltrans.refine_interpolate(solution_old, solution);

                    //save coarse solution vector
//                    std::cout <<" solution_lvl size " << solution_lvl.size() <<std::endl;
//                    solution_lvl.push_back(new TrilinosWrappers::MPI::BlockVector);
//                    solution_lvl.back()->reinit(solution);
//                    *(solution_lvl.back()) = solution;
//                    std::cout <<" solution_lvl size " << solution_lvl.size() <<std::endl;

                    //interpolate coarser solutions to current grid lvl
//                    for (unsigned int l=0; l<k-1; ++l)
//                    {
//                        solution_old.reinit(*(solution_lvl[l]));
//                        solution_old=*(solution_lvl[l]);
//                        solution_lvl[l]->reinit(solution);
//                        soltrans.refine_interpolate(solution_old,*(solution_lvl[l]));
//                    }

                }
                else //start solving on coarsest grid
                {
                    /*------------------Initialize grid---------------------*/
                    if(triangulation.n_cells()!=0)
                        triangulation.clear();

                    GridGenerator::hyper_cube(triangulation,0,1);
                    triangulation.refine_global(initial_refinements);
                    /*------------------------------------------------------*/
                    make_grid_and_dofs();
                }
                assemble_constant_system();


                //Added: adapt system matrix to be fit for MINRES
                if (LinearSolverData::solver_type == LINEAR_SOLVER_MINRES)
                {  //change sign of block 01 and 11 to get a symmetric system for MINRES compatibility
                    constant_system_matrix.block(0,1) *= -1.0;
                    constant_system_matrix.block(1,1) *= -1.0;
                }


                //Added: try  { } catch (std::exception e) for solve_newton() to continue the run of other parameters even if one b e h combination fails
                try  {
                solve_newton();
                push_table_data();
                }
                catch (std::exception e) {
                    std::cout<<"Exception: "<<e.what()<<std::endl;
                }

                //Added: change sign of solution.block(1) to be able to compare more easily to fgmres results
                if (LinearSolverData::solver_type == LINEAR_SOLVER_MINRES)
                {
                    solution.block(1)*=-1.0;
                }

                /*-----------------------------------------------------------------------------------*/
                /*                          write solution to file                                   */
                /* ----------------------------------------------------------------------------------*/
//                std::string sol_u_name = "u_b"+std::to_string((int) std::log10(EquationData::beta))+
//                        "e"+std::to_string((int) std::log10(EquationData::epsilon))+
//                        "h"+std::to_string(refs);
//                std::string sol_y_name = "y_b"+std::to_string((int) std::log10(EquationData::beta))+
//                        "e"+std::to_string((int) std::log10(EquationData::epsilon))+
//                        "h"+std::to_string(refs);
//                write_vector(solution.block(0),sol_y_name);
//                write_vector(solution.block(1),sol_u_name);
                /* ----------------------------------------------------------------------------------*/

//                std::string MName="Mh"+std::to_string(refs);
//                write_matrix(constant_system_matrix.block(0,0),MName);
//                std::string KName="sbetaKh"+std::to_string(refs);
//                write_matrix(constant_system_matrix.block(1,0),KName);
            }


            /* ---------------------------------------------------------------------------------*/
            /*  Create reference solution by restricting finest solution to each coarser level. */
            /* ---------------------------------------------------------------------------------*/

//            for(unsigned int k=initial_refinements+n_refinements-2; k>=initial_refinements; k--)
//            {
//                SolutionTransfer<dim, TrilinosWrappers::MPI::BlockVector> soltrans(dof_handler);
//                //Coarsen mesh uniformly
//                //restrict reference solution to previous mesh
//                TrilinosWrappers::MPI::BlockVector solution_fine;
//                solution_fine.reinit(solution);
//                solution_fine = solution;
//                Vector<float> coarsening_marker (triangulation.n_active_cells());
//                coarsening_marker = 1.0;
//                GridRefinement::coarsen(triangulation,coarsening_marker,2.0);
//                triangulation.prepare_coarsening_and_refinement();
//                soltrans.prepare_for_coarsening_and_refinement(solution_fine);
//                triangulation.execute_coarsening_and_refinement();
//                make_grid_and_dofs();
//                soltrans.interpolate(solution_fine,solution);
//                //write solution to file
//                std::string ref_u_name = "ref_u_b"+std::to_string((int) std::log10(EquationData::beta))+
//                        "e"+std::to_string((int) std::log10(EquationData::epsilon))+
//                        "h"+std::to_string(k);
//                std::string ref_y_name = "ref_y_b"+std::to_string((int) std::log10(EquationData::beta))+
//                        "e"+std::to_string((int) std::log10(EquationData::epsilon))+
//                        "h"+std::to_string(k);
//                write_vector(solution.block(0),ref_y_name);
//                write_vector(solution.block(1),ref_u_name);
//            }

            /* ---------------------------------------------------------------------------------*/


            //Added: write solution for the finest level with scaling for the control
            /*-----------------------------------------------------------------------------------*/
            /*                          write solution to file                                   */
            /* ----------------------------------------------------------------------------------*/

//            //scaling of the lagrange multiplier in block 1 to get the control of the original system
//            TrilinosWrappers::MPI::BlockVector solution_scaled_control(solution);
//            solution_scaled_control = solution;
//            solution_scaled_control.block(1)*=-1.0/std::sqrt(EquationData::beta);

//            std::string sol_u_name = "u(control)scaled_b"+std::to_string((int) std::log10(EquationData::beta))+
//                    "e"+std::to_string((int) std::log10(EquationData::epsilon))+
//                    "h"+std::to_string(refs);
//            std::string sol_y_name = "y_b"+std::to_string((int) std::log10(EquationData::beta))+
//                    "e"+std::to_string((int) std::log10(EquationData::epsilon))+
//                    "h"+std::to_string(refs);
//            write_vector(solution_scaled_control.block(0),sol_y_name);
//            write_vector(solution_scaled_control.block(1),sol_u_name);
            /* ----------------------------------------------------------------------------------*/
        }


        std::string TableName = "table_data_beta"
                +std::to_string(std::log10(EquationData::beta))
                +"sig"+std::to_string(EquationData::sigma);

        //Added: specify linear solver in filename
        if (LinearSolverData::solver_type == 1)
                TableName += "GMRES";
            else // == 0
                TableName += "MINRES";

        MiscUtilities::print_table_data_ref_epsilon(TableName,
                                                    n_refinements,
                                                    n_epsilon,
                                                    table_data_list);
        table_data_list.clear();
    }

    output_results();

}

//Added: for mpi finalize, need input int argc, char *argv[]
int main(int argc, char *argv[])//(/*int argc, char **argv*/)
{
    //Added: turn off mpi (needed in my version of deal.ii)
    Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

    //Added: residual output from solvers (remember to check so that history is activated in solver control)
//    std::string logfilename = "large_residual_log.log";
//    std::ofstream logFile(logfilename.c_str());
//    deallog.attach(logFile);
//    deallog.depth_console (0);

    ParameterHandler prm;
    ParameterReader     param(prm);
    param.read_parameters("parametersPII.prm");

    deallog.depth_console (0);

    PoissonBox<2> poissonbox(prm);
    poissonbox.run();

}

