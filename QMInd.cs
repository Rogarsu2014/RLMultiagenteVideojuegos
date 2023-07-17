using Assets.Scripts.DataStructures;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;
//using System;
//using System.Runtime.Remoting.Messaging;

namespace Assets.Scripts.SampleMind
{
    public class QMInd : AbstractPathMind
    {
        // declarar Stack de Locomotion.MoveDirection de los movimientos hasta llegar al objetivo
        private Stack<Locomotion.MoveDirection> currentPlan = new Stack<Locomotion.MoveDirection>();

        private float[,] TablaQ, TablaRewards;

        public float alpha = 0.3f;
        public float gamma = 0.8f;

        private int numRows, numCols;

        public override void Repath()
        {
            // limpiar Stack 
        }

        public override Locomotion.MoveDirection GetNextMove(BoardInfo board, CellInfo currentPos, CellInfo[] goals)
        {
            // si la Stack no está vacía, hacer siguiente movimiento
            if (!currentPlan.Any())
            {
                numRows = board.NumRows;
                numCols = board.NumColumns;
                createTable(board, goals);
                explorar(board, goals);
                var searchResult = explotar(currentPos, board, goals);
                // recorre searchResult and copia el camino a currentPlan
                while (searchResult.nodoPadre != null)
                {
                    currentPlan.Push(searchResult.movPadre);
                    searchResult = searchResult.nodoPadre;
                }

                return currentPlan.Pop();
            }
            else
            {
                return currentPlan.Pop();
            }

            return Locomotion.MoveDirection.None;
        }

        private void explorar(BoardInfo board, CellInfo[] goals)
        {
            int maxEpi = 100; // número de episodios como máximo.
            int maxIter = 100; // número de iteraciones por cada episodio como máximo.

            for (int k = 0; k < maxEpi; k++)
            {
                int iter = 0;
                var next_cell = Get_random_cell(board);
                bool stop_condition = false;
                while (stop_condition == false)
                {
                    var current_cell = next_cell;
                    var current_action = Get_random_action();
                    do
                    {
                        current_action = Get_random_action();
                        next_cell = runFsm(board, current_cell, current_action);
                    } while (next_cell == current_cell);
                    float current_Q = getQ(current_cell, current_action);
                    float reward = getReward(next_cell);
                    float next_Qmax = getMaxQ(board, next_cell);
                    float next_Q = updateRule(current_Q, reward, next_Qmax);
                    updateTable(current_cell, current_action, next_Q);
                    iter = iter + 1;
                    stop_condition = stopEvaluation(iter, maxIter, next_cell, goals);
                }
            }

            board.setSeed();
            var seed = board.seed1;

            int rewardsLength = -100, qLenght = -100;
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    int num = TablaRewards[i, j].ToString().Length;
                    if (num > rewardsLength) rewardsLength = num;
                }
            }

            for (int i = 0; i < numRows * numCols - 1; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int num = GetNumDigitsInFloat(TablaQ[i, j]);
                    if (num > qLenght) qLenght = num;
                }
            }


            System.IO.StreamWriter streamWriter2 = new System.IO.StreamWriter("TablaRewards" + seed + ".txt"); //Codigo sacado parcialmente de https://social.msdn.microsoft.com/Forums/en-US/c3c37507-4166-4e15-97ff-4f5ddb89abcc/writing-an-2d-array-to-a-txt?forum=csharpgeneral
            string output2 = "";
            for (int i = numRows - 1; i >= 0; i--)
            {
                for (int j = 0; j < numCols; j++)
                {
                    output2 += (TablaRewards[i, j].ToString()).PadLeft(rewardsLength) + " ";
                }
                streamWriter2.WriteLine(output2);
                output2 = "";
            }
            streamWriter2.Close();

            System.IO.StreamWriter streamWriter = new System.IO.StreamWriter("TablaQSemilla"+seed+".txt");
            string output = "";
            for (int i = numRows * numCols - 1; i >= 0; i--)
            {
                for (int j = 0; j < 4; j++)
                {
                    output += (TablaQ[i, j].ToString()).PadLeft(qLenght) + " ";
                }
                streamWriter.WriteLine(output);
                output = "";
            }
            streamWriter.Close();

        }


        private Nodo explotar(CellInfo start, BoardInfo board, CellInfo[] goals)
        {
            var next_cell = new Nodo(start);
            bool stop_condition = false;
            int contadorRepeticiones = 0;
            while (stop_condition == false && contadorRepeticiones < 4000)
            {
                var current_cell = next_cell;
                var decision_made = Get_best_action(current_cell);
                next_cell = runFsm(board, current_cell, decision_made);
                if(next_cell == current_cell)
                {
                    do
                    {
                        var current_action = Get_random_action();
                        next_cell = runFsm(board, current_cell, current_action);
                    } while (next_cell == current_cell);
                }
                stop_condition = stopEvaluation(0, 10, next_cell, goals);
                contadorRepeticiones++;
            }
            if(stopEvaluation(0, 10, next_cell, goals) == true)
            {
                return next_cell;
            }
            else
            {
                Debug.Log("Lo más probable es que no exista una solución para esta semilla");
                Debug.Log("Intentelo de nuevo con otra semilla");
                return next_cell;
            }    
        }

        public Locomotion.MoveDirection Get_best_action(Nodo currentCel)
        {
            float best_Q = -10000000;
            int action = -1;
            int best_action = action;
            for(int i = 0; i < 4; i++)
            {
                float Q = TablaQ[numCols * currentCel.RowId + currentCel.ColumnId, i];
                action = action + 1;
                if (Q > best_Q)
                {
                    best_Q = Q;
                    best_action = action;
                }
            }
            return parseAction(best_action);
        }

        public Locomotion.MoveDirection parseAction(int action)
        {
            switch (action)
            {
                case 0:
                    return Locomotion.MoveDirection.Up;
                case 1:
                    return Locomotion.MoveDirection.Down;
                case 2:
                    return Locomotion.MoveDirection.Left;
                case 3:
                    return Locomotion.MoveDirection.Right;
                default:
                    return Locomotion.MoveDirection.None;                    
            }   
        }

        public int parseLocomotion(Locomotion.MoveDirection action)
        {
            switch (action)
            {
                case Locomotion.MoveDirection.Up:
                    return 0;
                case Locomotion.MoveDirection.Down:
                    return 1;
                case Locomotion.MoveDirection.Left:
                    return 2;
                case Locomotion.MoveDirection.Right:
                    return 3;
                default:
                    return -1;
            }
        }

        public Nodo runFsm(BoardInfo board, Nodo cel, Locomotion.MoveDirection action)
        {
            switch (action)
            {
                case Locomotion.MoveDirection.Up:
                    if (cel.RowId < numRows - 1 && board.CellInfos[cel.ColumnId, cel.RowId + 1].Walkable == true)
                    {
                        return new Nodo(board.CellInfos[cel.ColumnId, cel.RowId + 1], cel);
                    }
                    return cel;

                case Locomotion.MoveDirection.Down:
                    if (cel.RowId > 0 && board.CellInfos[cel.ColumnId, cel.RowId - 1].Walkable == true)
                    {
                        return new Nodo(board.CellInfos[cel.ColumnId, cel.RowId - 1], cel);
                    }
                    return cel;

                case Locomotion.MoveDirection.Left:
                    if (cel.ColumnId > 0 && board.CellInfos[cel.ColumnId - 1, cel.RowId].Walkable == true)
                    {
                        return new Nodo(board.CellInfos[cel.ColumnId - 1, cel.RowId], cel);
                    }
                    return cel;

                case Locomotion.MoveDirection.Right:
                    if (cel.ColumnId < numCols - 1 && board.CellInfos[cel.ColumnId + 1, cel.RowId].Walkable == true)
                    {
                        return new Nodo(board.CellInfos[cel.ColumnId + 1, cel.RowId], cel);
                    }
                    return cel;

                default:
                    return cel;
            }
        }

        public float getQ(Nodo currentCel, Locomotion.MoveDirection action)
        {
            switch (action)
            {
                case Locomotion.MoveDirection.Up:
                    return TablaQ[numCols * currentCel.RowId + currentCel.ColumnId, 0];

                case Locomotion.MoveDirection.Down:
                    return TablaQ[numCols * currentCel.RowId + currentCel.ColumnId, 1];

                case Locomotion.MoveDirection.Left:
                    return TablaQ[numCols * currentCel.RowId + currentCel.ColumnId, 2];

                case Locomotion.MoveDirection.Right:
                    return TablaQ[numCols * currentCel.RowId + currentCel.ColumnId, 3];

                default:
                    return 0;
            }

        }

        public float getReward(Nodo nextCel)
        {
            return TablaRewards[nextCel.RowId, nextCel.ColumnId];
        }

        public float getMaxQ(BoardInfo board, Nodo nextCel)
        {
            float maxQ = -100000000;

            for (int i = 0; i < 4; i++)
            {
                Locomotion.MoveDirection action = parseAction(i);
                float tempQ = getQ(nextCel, action);
                if (tempQ > maxQ) maxQ = tempQ;
            }

            return maxQ;
        }

        public float updateRule(float current_Q, float reward, float next_Qmax)
        {
            return (1 - alpha) * current_Q + alpha * (reward + gamma * next_Qmax);
        }

        public void updateTable(Nodo currentCel, Locomotion.MoveDirection current_action, float next_Q)
        {
            int accion = parseLocomotion(current_action);
            if(accion != -1)
            {
                TablaQ[numCols * currentCel.RowId + currentCel.ColumnId, accion] = next_Q;
            }
        }
            

        public bool stopEvaluation(int iter, int maxIter, Nodo nextCell, CellInfo[] goals)
        {
            if(iter > maxIter)
            {
                return true;
            }
            if(nextCell.RowId == goals[0].RowId && nextCell.ColumnId == goals[0].ColumnId)
            {
                return true;
            }

            return false;
        }

        public Locomotion.MoveDirection Get_random_action() //Usa el mismo código que RandomMind
        {
            var val = Random.Range(0, 4);
            if (val == 0) return Locomotion.MoveDirection.Up;
            if (val == 1) return Locomotion.MoveDirection.Down;
            if (val == 2) return Locomotion.MoveDirection.Left;
            return Locomotion.MoveDirection.Right;
        }

        public Nodo Get_random_cell(BoardInfo board)
        {
            int val1, val2;
            do
            {
                val1 = Random.Range(0, board.NumRows);
                val2 = Random.Range(0, board.NumColumns);

            } while (board.CellInfos[val2, val1].Walkable != true);

            return new Nodo(board.CellInfos[val2, val1]);
        }

        public void createTable(BoardInfo board, CellInfo[] goals)
        {
            float[,] todash1 = new float[board.NumRows, board.NumColumns];
            for (int i = 0; i < board.NumRows; i++)
            {
                for (int j = 0; j < board.NumColumns; j++)
                {
                    if(i == goals[0].RowId && j == goals[0].ColumnId)
                    {
                        todash1[i, j] = 100;
                    }
                    else if (board.CellInfos[j, i].Walkable != true)
                    {
                        todash1[i, j] = -1;
                    }
                    else
                    {
                        todash1[i, j] = 0;
                    }
                    
                }
            }

            TablaRewards = todash1;

            float[,] todash2 = new float[board.NumRows * board.NumColumns , 4];
            for (int i = 0; i < board.NumRows * board.NumColumns; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    todash2[i, j] = 0;
                }
            }

            TablaQ = todash2;
        }

        static private int GetNumDigitsInFloat(float n) // Código cogido de https://stackoverflow.com/questions/24481813/how-to-get-the-length-of-a-float-c
        {
            string s = n.ToString();
            return s.Length - 1;
        }

    }
}
