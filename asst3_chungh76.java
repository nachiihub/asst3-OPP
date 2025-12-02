import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class asst3_chungh76 {

    //Small number to treat as 0
    private static final double EPS = 1e-12;

    //Simple container for L and U
    private static class LUResult {
        final double[][] L;
        final double[][] U;

        LUResult(double[][] L, double[][] U) {
            this.L = L;
            this.U = U;
        }
    }


    //Reads a matrix of doubles from a file.
    private static double[][] readMatrix(String filename) throws IOException {
        List<double[]> rows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            int expectedCols = -1;

            //read text file and valide it and conver to numb ers
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue; // skip blank lines
                }

                //Splits row into individual numbers
                String[] parts = line.split("\\s+");
                if (expectedCols == -1) {
                    expectedCols = parts.length;
                } 
                else if (parts.length != expectedCols) {
                    throw new IllegalArgumentException("Matrix rows must have the same number of columns.");
                }

                //Converts strings into doubles
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                rows.add(row);
            }
        }

        if (rows.isEmpty()) {
            throw new IllegalArgumentException("Input matrix is empty.");
        }

        int nRows = rows.size();
        //Checks first row for number of columns
        int nCols = rows.get(0).length;

        if (nRows != nCols) {
            throw new IllegalArgumentException("Matrix must be square.");
        }

        //Creates a matrix of the correct size and fills it
        double[][] A = new double[nRows][nCols];
        for (int i = 0; i < nRows; i++) {
            A[i] = rows.get(i);
        }
        return A;
    }

    /**
     * Doolittle LU decomposition with optional parallelization.
     * A = L * U, where L has unit diagonal.
     *
     * Throws:
     *  - ArithmeticException if the matrix is singular.
     */
    private static LUResult decompose(double[][] A, boolean parallel) {
        int n = A.length;
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        // Initialize L as identity
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0;
        }

        for (int i = 0; i < n; i++) {
            final int row = i;

            //Compute U[row][j]values using Parallel
            if (parallel) {
                IntStream.range(row, n).parallel().forEach(j -> {
                    double sum = 0.0;
                    for (int k = 0; k < row; k++) {
                        sum += L[row][k] * U[k][j];
                    }
                    U[row][j] = A[row][j] - sum;
                });
            } 

            //Compute U[row][j]values using Sequential
            else {
                for (int j = row; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < row; k++) {
                        sum += L[row][k] * U[k][j];
                    }
                    U[row][j] = A[row][j] - sum;
                }
            }

            //Check for singularity: U[row][row] must not be zero
            if (Math.abs(U[row][row]) < EPS) {
                throw new ArithmeticException("Matrix is singular, cannot perform decomposition.");
            }

            //Compute L[j][row] using Parrralel
            if (parallel) {
                final int col = row;
                IntStream.range(row + 1, n).parallel().forEach(j -> {
                    double sum = 0.0;
                    for (int k = 0; k < col; k++) {
                        sum += L[j][k] * U[k][col];
                    }
                    L[j][col] = (A[j][col] - sum) / U[col][col];
                });
            } 
            //Compute L[j][row] using Sequential
            else {
                for (int j = row + 1; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < row; k++) {
                        sum += L[j][k] * U[k][row];
                    }
                    L[j][row] = (A[j][row] - sum) / U[row][row];
                }
            }
        }

        return new LUResult(L, U);
    }


    //Matrix multiplication: C 
    private static double[][] multiply(double[][] A, double[][] B, boolean parallel) {
        int n = A.length;
        double[][] C = new double[n][n];

        if (parallel) {
            IntStream.range(0, n).parallel().forEach(i -> {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            });
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
        return C;
    }

    //Difference matrix: D 
    private static double[][] difference(double[][] A, double[][] B, boolean parallel) {
        int n = A.length;
        double[][] D = new double[n][n];

        if (parallel) {
            IntStream.range(0, n).parallel().forEach(i -> {
                for (int j = 0; j < n; j++) {
                    double raw = A[i][j] - B[i][j];
                    //Rounds to 4 digit to match output
                    D[i][j] = Math.round(raw * 10000.0) / 10000.0; 
                }
            });
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double raw = A[i][j] - B[i][j];
                    //Rounds to 4 digit to match output
                    D[i][j] = Math.round(raw * 10000.0) / 10000.0;
                }
            }
        }

        return D;
    }

    //Frobenius norm of a matrix
    private static double frobeniusNorm(double[][] D, boolean parallel) {
        int n = D.length;

        DoubleStream stream = IntStream.range(0, n).mapToDouble(i -> {
            double rowSum = 0.0;
            for (int j = 0; j < n; j++) {
                double v = D[i][j];
                rowSum += v * v;
            }
            return rowSum;
        });

        double sumSquares = parallel ? stream.parallel().sum() : stream.sum();
        return Math.sqrt(sumSquares);
    }


    //Writes A matrix to the output file with 1 decimal places per entry
    private static void writeMatrix(BufferedWriter writer, double[][] M) throws IOException {
        int n = M.length;
        for (int i = 0; i < n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                sb.append(String.format("%.1f ", M[i][j]));
            }
            writer.write(sb.toString().trim());
            writer.newLine();
        }
    }

    //Overloaded method to print Difference in 4 decimal place 
    private static void writeMatrix(BufferedWriter writer, double[][] M, boolean four_decimal) throws IOException {
        int n = M.length;
        for (int i = 0; i < n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                sb.append(String.format("%.4f ", M[i][j]));
            }
            writer.write(sb.toString().trim());
            writer.newLine();
        }
    }

    //Reads config to determine if we use parallel or seqeuential
    private static boolean readExecutionMode(String configFileName) {
        File f = new File(configFileName);
        //Checks if a file name was entered, if not default to parallel
        if (!f.exists()) {
            return true;
        }

        try{
            BufferedReader br = new BufferedReader(new FileReader(f));
            String line;
            while ((line = br.readLine()) != null) {
                //Remove leading/trailing spaces
                line = line.trim();
                //Skip empty lines & coimments
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                //Split the line at =
                String[] parts = line.split("=");
                if (parts.length == 2) {
                    //Trim both sides of the =
                    String key = parts[0].trim();
                    String value = parts[1].trim();
                    if (key == "parallel_execution") {
                        //returns true for true and false for false
                        return value.equalsIgnoreCase("true");
                    }
                }
            }
        } catch (IOException e) {
            // If config can't be read, default to parallel
            return true;
        }

        // If config didn't specify the key, default to parallel
        return true;
    }

    public static void main(String[] args) {
        String inputFileName;
        //Checks if user ran command line with input file txt
        boolean noInputArg = (args.length == 0);

        if (noInputArg) {
            inputFileName = "input.txt";
        } else {
            inputFileName = args[0];
        }

        String outputFileName = "output.txt";
        boolean parallel = readExecutionMode("config.txt");
        //If else statemente
        String executionMode = parallel ? "parallel" : "sequential";

        BufferedWriter writer = null;
        try {
            //Creates the writer object
            writer = new BufferedWriter(new FileWriter(outputFileName));

            // If no argument, we must note this in the output file
            if (noInputArg) {
                writer.write("No input file specified. Using default: input.txt");
                writer.newLine();
            }

            // Header lines
            writer.write("Input file: " + inputFileName);
            writer.newLine();
            writer.write("Output file: " + outputFileName);
            writer.newLine();
            writer.write("Execution mode: " + executionMode);
            writer.newLine();
            writer.newLine();

            // --- Read matrix A ---
            double[][] A;
            try {
                A = readMatrix(inputFileName);
            } 
            catch (IllegalArgumentException e) {
                // Non-square matrix or malformed input
                writer.write("Error: " + e.getMessage());
                writer.newLine();
                return;
            } 
            catch (IOException e) {
                writer.write("Error: Unable to read input file: " + e.getMessage());
                writer.newLine();
                return;
            }

            //Print A Matrix
            writer.write("Matrix A:");
            writer.newLine();
            writeMatrix(writer, A);
            writer.newLine();

            //Decompose A into L and U 
            LUResult lu;
            try {
                lu = decompose(A, parallel);
            } catch (ArithmeticException e) {
                // Singular matrix
                writer.write("Error: " + e.getMessage());
                writer.newLine();
                return;
            }

            //Reassign matrix L and U
            double[][] L = lu.L;
            double[][] U = lu.U;

            //Print L and U
            writer.write("Final Matrix L:");
            writer.newLine();
            writeMatrix(writer, L);
            writer.newLine();

            writer.write("Final Matrix U:");
            writer.newLine();
            writeMatrix(writer, U);
            writer.newLine();

            //Compute A'
            double[][] reconstructed = multiply(L, U, parallel);

            //Difference matrix D 
            double[][] diff = difference(A, reconstructed, parallel);
            boolean four_decimal = true;

            writer.write("Difference Matrix (A - LU):");
            writer.newLine();
            writeMatrix(writer, diff, four_decimal);
            writer.newLine();

            //Frobenius norm of D
            double tolerance = frobeniusNorm(diff, parallel);
            writer.write(String.format("Tolerance (difference between A and LU): %.4f", tolerance));
            writer.newLine();
            writer.newLine();
            writer.write("Decomposition complete. Results written to " + outputFileName);
        } 
        catch (IOException e) {
            // If we can't open/write the output file, print to stderr
            System.err.println("I/O error writing to " + outputFileName + ": " + e.getMessage());
        } 
        finally {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException ignore) {
                    // ignore
                }
            }
        }
    }
}