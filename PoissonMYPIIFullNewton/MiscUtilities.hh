
#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>
#include <string>
/*For Tensors */
#include <deal.II/base/tensor_function.h>

using namespace dealii;

namespace MiscUtilities{

class TableData{
    static const unsigned int size=9;
public : double beta;
    double epsilon;
    unsigned int dofs;
    double refinements;
    unsigned int newton_iterations;
    unsigned int gmres_iterations;
    unsigned int h1_iterations;
    unsigned int h2_iterations;
    unsigned int m_iterations;
    double time;
    unsigned int index;
    TableData(){}

public : friend std::ofstream& operator <<(std::ofstream& os, const TableData& dt);
};
//std::ofstream& operator <<(std::ofstream& os, const TableData& dt)
//{
//    os <<dt.iterations << "("  <<dt.k_iterations << "+" << dt.m_iterations << ")";

//    return os;
//}


void print_table_data_beta_epsilon (std::string file_name, unsigned int beta_values_length, unsigned int epsilon_values_length, std::vector<TableData> table_data_list)
{
    //std::vector<char*>  ref(refinements_length);
    //ref.push_back("agc");
    //ref.push_back("$2^{-5}$");  ref.push_back("$2^{-6}$");  ref.push_back("$2^{-7}$");  ref.push_back("abc");
    std::ofstream output;
    file_name +=".txt";
    output.open (file_name);
    for(unsigned int i=0;i<epsilon_values_length;i++){
        for(unsigned int j=0;j<beta_values_length;j++){
            MiscUtilities::TableData p=(MiscUtilities::TableData)table_data_list[i+j*epsilon_values_length];
            if(j==0)
                output<<p.epsilon << " & ";
            //newton iterations then total gmres
            output<<"{"<< p.newton_iterations << "}("  <<p.gmres_iterations << ")" << "(" << p.h1_iterations << ")";
            //put & sign
            if(j<beta_values_length-1)
                output<<" & ";

        }

        output <<"\\\\ \n";
        for(unsigned int j=0;j<beta_values_length;j++){
            MiscUtilities::TableData p=( MiscUtilities::TableData)table_data_list[i+j*epsilon_values_length];
            if(j==0)
                output <<" & ";

            output << round(p.time*1000.0)/1000.0;
            if(j<beta_values_length-1)
                output<<" & ";
        }
        output <<"\\\\";
        output <<"\n";
    }
    output.close();
}

void print_table_data_ref_epsilon (std::string file_name, unsigned int refinements_length, unsigned int epsilon_values_length, std::vector<TableData> table_data_list)
{
    //std::vector<char*>  ref(refinements_length);
    //ref.push_back("agc");
    //ref.push_back("$2^{-5}$");  ref.push_back("$2^{-6}$");  ref.push_back("$2^{-7}$");  ref.push_back("abc");
    std::ofstream output;
    file_name +=".txt";
    output.open (file_name);
    for(unsigned int i=0;i<epsilon_values_length;i++){
        for(unsigned int j=0;j<refinements_length;j++){
            MiscUtilities::TableData p=(MiscUtilities::TableData)table_data_list[i*refinements_length+j];
            if(j==0)
                output<<p.epsilon << " & ";
            //newton iterations then total gmres
            output<<"{"<< p.newton_iterations << "}("  <<p.gmres_iterations << ")" << "(" << p.h1_iterations << ")";
            //put & sign
            if(j<refinements_length-1)
                output<<" & ";

        }

        output <<"\\\\ \n";
        for(unsigned int j=0;j<refinements_length;j++){
            MiscUtilities::TableData p=( MiscUtilities::TableData)table_data_list[i*refinements_length+j];
            if(j==0)
                output <<" & ";

            output << round(p.time*1000.0)/1000.0;
            if(j<refinements_length-1)
                output<<" & ";
        }
        output <<"\\\\";
        output <<"\n";
    }
    output.close();
}

//void print_table_data_for_h_blocks (std::string file_name, unsigned int beta_values_length, unsigned int refinements_length, std::vector<TableData> table_data_list)
//{
//    //std::vector<char*>  ref(refinements_length);
//    //ref.push_back("agc");
//    //ref.push_back("$2^{-5}$");  ref.push_back("$2^{-6}$");  ref.push_back("$2^{-7}$");  ref.push_back("abc");
//    std::ofstream output;
//    file_name +=".txt";
//    output.open (file_name);
//    for(unsigned int i=0;i<refinements_length;i++){
//        for(unsigned int j=0;j<beta_values_length;j++){
//            MiscUtilities::TableData p=(MiscUtilities::TableData)table_data_list[i+j*refinements_length];
//            if(j==0)
//                output<<p.dofs << " & ";
//            output<<"{"<< p.iterations << "}("  <<p.h1_iterations <<")";
//            //put & sign
//            if(j<beta_values_length-1)
//                output<<" & ";

//        }
//        output <<"\\\\ \n";
//        for(unsigned int j=0;j<beta_values_length;j++){
//            MiscUtilities::TableData p=(MiscUtilities::TableData)table_data_list[i+j*refinements_length];
//            if(j==0)
//                output<<" & ";
//            //output << p;

//            std::string str=std::to_string(p.iterations);
//            str.replace(str.begin(),str.end(), " " );
//            output << str<< "("  <<p.k_iterations << "+" << p.m_iterations << ")";

//            if(j<beta_values_length-1)
//                output<<" & ";

//        }

//        output <<"\\\\ \n";
//        //This places the time
//        for(unsigned int j=0;j<beta_values_length;j++){

//            MiscUtilities::TableData p=( MiscUtilities::TableData)table_data_list[i+j*refinements_length];
//            if(j==0)
//                output << " & " << " & ";

//            output << round(p.time*1000.0)/1000.0;
//            if(j<beta_values_length-1)
//                output<<" & ";
//        }
//        output <<"\\\\";
//        output <<"\n";
//    }
//    output.close();
//}







//void print_tex_tables(std::string gmv_filename , /*SolverType s,*/ TableHandler table)
//{

////    if(s==SolverType::GMRES)
////        gmv_filename += "-fgmres";
////    else
////        gmv_filename += "-minres";


//    table.set_precision("$\\beta$",0 );
//    table.set_precision("$\\widetilde{\\beta}$",2 );

//    table.set_precision("time",3 );
//    table.set_precision("ref", 0);
//    table.set_precision("Tol", 2);

//    table.set_precision("$|u|_{2}$",2);
//    table.set_precision("$\|u\|_{\\infty}$", 2);
//    table.set_precision("$\|y-\\hat{y}\|$",2);
//    table.set_precision("$\|y-\\hat{y}\|/|\\hat{y}\|$",2);
//    table.set_precision("$J$",2);

//    table.set_scientific("$|u|_{2}$",true);
//    table.set_scientific("$\|u\|_{\\infty}$", true);

//    table.set_scientific("$\\beta$", true);
//    table.set_scientific("$\\widetilde{\\beta}$",true );
//    table.set_scientific("$\|y-\\hat{y}\|$",true);
//    table.set_scientific("$\|y-\\hat{y}\|/|\\hat{y}\|$",true);
//    table.set_scientific("$J$",true);
//    table.set_scientific("Tol", true);


//    std::cout << std::endl;
//    table.write_text(std::cout);
//    std::ofstream table_file(gmv_filename.c_str());
//    table.write_tex(table_file);

//}

//void print_tex_tables_for_solution_data(TableHandler table, std::string file_name)
//{

//    file_name += ".tex";


//    table.set_precision("$p_{max}$", 2);
//    table.set_precision("$p_{min}$", 2);
//    table.set_precision("$\|p\|_{\\infty}$", 2);
//    table.set_precision("$m_{max}$", 2);
//    table.set_precision("$m_{min}$", 2);
//    table.set_precision("$\|m\|_{\\infty}$", 2);
//    table.set_precision("$|u|_{2}$", 2);
//    table.set_precision("$\|u\|_{\\infty}$", 2);
//    table.set_precision("$|y|_{2}$", 2);
//    table.set_precision("$\|y\|_{\\infty}$",2);

//    table.set_scientific("$p_{max}$", false);
//    table.set_scientific("$p_{min}$", false);
//    table.set_scientific("$\|p\|_{\\infty}$", true);
//    table.set_scientific("$m_{max}$", true);
//    table.set_scientific("$m_{min}$", true);
//    table.set_scientific("$\|m\|_{\\infty}$", true);
//    table.set_scientific("$|u|_{2}$", 2);
//    table.set_scientific("$\|u\|_{\\infty}$", true);
//    table.set_scientific("$|y|_{2}$", true);
//    table.set_scientific("$\|y\|_{\\infty}$",true);

//    std::cout << std::endl;
//    table.write_text(std::cout);
//    std::ofstream table_file(file_name.c_str());
//    table.write_tex(table_file);

//}

//void print_inner_solver_errors(){
//    unsigned int error_list=LinearSolvers::error_list.size();
//    if( error_list>0){
//        std::cout<<std::endl;
//        for(int i=0;i<error_list;i++){
//            PreconditionerError p=((PreconditionerError)LinearSolvers::error_list[i]);
//            std::cout<<p.n_refs<<" "<<p.beta_value<<" "<<p.preconditioner_name<<" "<<p.solver_last_value<<" "<<p.target_tolerance<<std::endl;
//        }
//    }
//}

template <int dim>
void interp_solution_f_to_c(unsigned int block_id,
                            DoFHandler<dim> &dof_handler_VH,
                            DoFHandler<dim> &dof_handler_Vh,
                            FEValues<dim> &fe_values_VH,
                            FEValues<dim> &fe_values_Vh,
                            TrilinosWrappers::Vector solution_Vh,
                            TrilinosWrappers::BlockVector intrp_solution_VH
                            )

{
    InterGridMap<DoFHandler<dim> > intergrid_map;
    //make mapping from coarse grid to fine grid
    intergrid_map.make_mapping(dof_handler_Vh, dof_handler_VH);
    Vector<double> local_fine_values(fe_values_Vh.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_Vh.begin_active(),
            endc = dof_handler_Vh.end();
    for (; cell!=endc; ++cell)
    {
        cell->get_dof_values (solution_Vh,
                              local_fine_values);

        intergrid_map[cell]->set_dof_values_by_interpolation(local_fine_values, intrp_solution_VH.block(block_id));

    }
}

template <int dim>
void interp_solution_c_to_f(unsigned int block_id,
                            DoFHandler<dim> &dof_handler_VH,
                            DoFHandler<dim> &dof_handler_Vh,
                            FEValues<dim> &fe_values_VH,
                            FEValues<dim> &fe_values_Vh,
                            TrilinosWrappers::BlockVector solution_VH,
                            TrilinosWrappers::BlockVector intrp_solution_Vh

                            )
{

    InterGridMap<DoFHandler<dim> > intergrid_map;
    //make mapping from coarse grid to fine grid
    intergrid_map.make_mapping(dof_handler_VH, dof_handler_Vh);

    Vector<double> local_coarse_values(4);

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_VH.begin_active(),
            endc = dof_handler_VH.end();
    for (; cell!=endc; ++cell)
    {
        cell->get_dof_values (solution_VH.block(block_id),
                              local_coarse_values);
        intergrid_map[cell]->set_dof_values_by_interpolation(local_coarse_values, intrp_solution_Vh.block(block_id));
    }

}

//template <int dim>
//void assemble_Ph(unsigned int block_id,
//                 DoFHandler<dim> &dof_handler_Vh,
//                 FEValues<dim> &fe_values_Vh,
//                 TrilinosWrappers::SparseMatrix Ph_matrix_Vh,
//                 TrilinosWrappers::Vector system_rhs_Vh,
//                 TrilinosWrappers::BlockVector intrp_solution_Vh,
//                 ConstraintMatrix proj_constraints)

//{
//    //FEValuesExtractors::Vector vec(0);
//    Ph_matrix_Vh=0;
//    system_rhs_Vh=0;

//    const unsigned int dofs_per_cell = 4;//proj_fe_Vh.dofs_per_cell;


//    const unsigned int n_q_points = 4;//quadrature_formula.size();
//    std::cout<<"n_q_points: "<<n_q_points <<std::endl;
//    std::cout<<"dofs per cell Vh: "<<dofs_per_cell<<std::endl;


//    const ConvectionDiffusion::AdvectionField<dim> advection_field;

//    std::vector<Tensor<1,dim> > advection_directions(n_q_points);
//    std::vector< Tensor<1, dim>> grad_solution(n_q_points);


//    //Local matrix
//    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
//    //local rhs
//    Vector<double> cell_rhs (dofs_per_cell);

//    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

//    const ConvectionDiffusion::ControlValues<dim> poisson_control_values(1);
//    std::vector<double> control_values (n_q_points);


//    typename DoFHandler<dim>::active_cell_iterator
//            cell = dof_handler_Vh.begin_active(),
//            endc = dof_handler_Vh.end();
//    for (; cell!=endc; ++cell)
//    {

//        fe_values_Vh.reinit (cell);

//        fe_values_Vh.get_function_gradients(intrp_solution_Vh.block(block_id), grad_solution);


//        advection_field.value_list (fe_values_Vh.get_quadrature_points(),advection_directions);


//        std::vector<Point<dim>> p_list=fe_values_Vh.get_quadrature_points();
//        poisson_control_values.value_list (p_list, control_values);



//        cell_matrix = 0;
//        cell_rhs    = 0;

//        for (unsigned int q_index=0; q_index<n_q_points; ++q_index){
//            /*double advection_directions_norm=advection_directions[q_index].norm();
//            double pe_lps=(cell->diameter()*advection_directions_norm)/EquationData::epsilon;
//            double delta=0;
//            if(pe_lps>=1)
//                delta=(cell->diameter())/(advection_directions_norm);
//            else
//                delta=0;*/

//            for (unsigned int i=0; i<dofs_per_cell; ++i)
//            {
//                for (unsigned int j=0; j<dofs_per_cell; ++j){

//                    //Projection Mass Matrix */
//                    cell_matrix(i, j) +=(fe_values_Vh.shape_value (j, q_index) * fe_values_Vh.shape_value (i, q_index))*
//                            fe_values_Vh.JxW (q_index);


//                }
//                cell_rhs(i) += (advection_directions[q_index]*grad_solution[q_index]*fe_values_Vh.shape_value(i, q_index))*fe_values_Vh.JxW (q_index);


//            }
//        }

//        cell->get_dof_indices (local_dof_indices);
//        proj_constraints.distribute_local_to_global (cell_matrix,
//                                                     cell_rhs,
//                                                     local_dof_indices,
//                                                     Ph_matrix_Vh,
//                                                     system_rhs_Vh);


//    }


//    std::cout<<"projection mass matrix assembled "<<std::endl;
//}




}
