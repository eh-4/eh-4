// Emerson Ham - Good Racks LLC
// This file contains the class used for the machines used by Good Racks LLC
// Seperate classes were created for the UI window, the geometries of the racks, and 
// the worker which generates the G-Code sent to the machines via this class.

using System;
using System.Windows;
using System.IO;
using System.IO.Ports;
using System.Text.RegularExpressions;

namespace GR_CNC
{
    public class CNC_Machine
    {
        //Constant Fields
        public readonly double cutAreaWidth = 33.08;
        public readonly double[] bitRadius = new double[] { 0.125, 0.0234375, 0.25, 0.05, 0.0625 };     //radius of the bit

        //Private Fields
        private int state = 2;         //Disconnected at start
        private SerialPort serial = new SerialPort();
        private int machineNum = 2;     //Stores which machine we are using. 2=Right, 3=Left. 0 and 1 were stolen
        private MainWindow wnd = (MainWindow)Application.Current.MainWindow; //To output to textbox use: wnd.TextBox_output.AppendText("Stuff");

        //Constructors
        public CNC_Machine()
        {

        }

        //Properties/Encapsulation
        public int State
        {
            //State values: all good = 0, emergency stop = 1, disconnected = 2
            get => state;
            private set
            {
                if (value == 0 || value == 1 || value == 2) { state = value; }
                else { throw new ArgumentException("CNC Object Error: Incorrect value set to 'State' property."); }
            }
            //public set must be done by calling the CNC_Machine.EmergencyStop() function
        }

        public int MachineNum
        { 
            //We have had 5 machines, however machines #0 and #1 were stolen, leaving 2, 3, and 4 as valid options
            get => machineNum;
            set
            {
                if (value == 2 || value == 3 || value == 4) { machineNum = value; }
                else { throw new ArgumentException("CNC object error: Incorrect value set to 'MachineNum' property."); }
            }
        }

        public int COMport { get; set; } = 0;

        //Methods
        public void setup()
        {
            //Tasks:
            //  - Sets up the serial connection to the machine and homes the machine
            //Inputs: none
            //Returns: none

            wnd = (MainWindow)Application.Current.MainWindow; //In case it didn't connect properly when initialized
            wnd.TextBox_output.Text = "--SETUP--\n";
            wnd.GrayButtons("Connecting");

            State = 0;
            wnd.Button_Connect.IsEnabled = false;
            wnd.Delay(100);

            //Close serial port if already open
            try
            {
                serial.Close();
            }
            catch { }

            //Start serial communication
            try
            {
                serial.PortName = ("COM" + COMport);
                serial.BaudRate = 115200;
                //serial.Handshake = System.IO.Ports.Handshake.None;
                serial.Parity = Parity.None;
                serial.DataBits = 8;
                serial.StopBits = StopBits.One;
                serial.ReadTimeout = 5000; //3000
                serial.WriteTimeout = 200; //100
                serial.DtrEnable = true;
                serial.RtsEnable = false;

                //Creates event handler for incomming data
                serial.Open();

                wnd.Delay(50);
                serial.DiscardInBuffer();
                wnd.TextBox_output.AppendText("Serial port opened successfully.\n");
            }
            catch
            {
                wnd.TextBox_output.AppendText("ERROR - Serial port failed to open. Check power, USB, and COM port. state => 2");
                State = 2;
                wnd.GrayButtons("Disconnected");
                return;
            }

            //Home twice, the first with a large pulloff, distance and faster feed rate
            // the second and all consecutive are standard more accurate ones
            if (Home("Setup1"))
            {
                //Check that the sizes are correct
                wnd.GR.updateSizes();
                MessageBox.Show("MACHINE #" + MachineNum.ToString() + ":\n" +
                    "Check that these types/sizes match the racks in the machine");

                //Enable buttons
                write("S24000"); //Set spindle speed to max in case any future speed settings don't work for some reason
                wnd.TextBox_output.AppendText("Ready!\n");
                wnd.GrayButtons("Idle");
                return;
            }
            else
            {
                wnd.TextBox_output.AppendText("--ERROR-- FAILED TO HOME\n");
                wnd.GrayButtons("Disconnected");
                State = 2;
                return;
            }

        }

        public void shutdown()
        {
            //Tasks:
            //  - Tries to move the machine home and disconnects the serial connection
            //Inputs: none
            //Returns: none

            try
            { 
                wnd.TextBox_output.AppendText("Shutting down...");
                wnd.GrayButtons("Disconnected");
                wnd.Refresh();
            }
            catch { }
            try
            {
                //Move machine out of way
                write("M5"); //Stop the spindle
                write("G0 G53", Z:-0.25); // Return to machine coordinate 0,0,0 in back right corner
                write("G53", X:-0.25, Y:-0.25);

                //Close serial port
                waitForCompletion();
                serial.Close();
            }
            catch { }
            try
            {
                wnd.TextBox_output.AppendText("Done.\n");
            }
            catch { }
            State = 2;
        }

        public void write(String command, double X = Double.NaN, double Y = Double.NaN, double Z = Double.NaN, double R = Double.NaN, double I = Double.NaN, double J = Double.NaN)
        {
            //Tasks:
            //  - Sends a gCode command to the machine
            //Inputs:
            //  command - String containing either manually written out gCode, or the G number (G00, G01, G02, etc.)
            //  X - (optional) the x value for the command (inches)
            //  Y - (optional) the y value for the command (inches)
            //  Z - (optional) the z value for the command (inches)
            //  R - (optional) the R value for the command (inches)
            //  I - (optional) the i value for the command (inches)
            //  J - (optional) the j value for the command (inches)
            //Returns: none

            if (State != 0 && command[0] != '$') { return; }
            //Assemble gCode
            String gCode = "";
            switch (command)
            {
                case "G00":
                    gCode = "G00 ";
                    if (!Double.IsNaN(X)) { gCode += " X" + X.ToString("F4"); }
                    if (!Double.IsNaN(Y)) { gCode += " Y" + Y.ToString("F4"); }
                    if (!Double.IsNaN(Z)) { gCode += " Z" + Z.ToString("F4"); }
                    break;
                case "G01":
                    gCode = "G01 ";
                    if (!Double.IsNaN(X)) { gCode += " X" + X.ToString("F4"); }
                    if (!Double.IsNaN(Y)) { gCode += " Y" + Y.ToString("F4"); }
                    if (!Double.IsNaN(Z)) { gCode += " Z" + Z.ToString("F4"); }
                    break;
                case "G02":
                    gCode = "G02 ";
                    if (!Double.IsNaN(X)) { gCode += " X" + X.ToString("F6"); }     //Since CNC machines check the validity of an arc, we have to meet that precision
                    if (!Double.IsNaN(Y)) { gCode += " Y" + Y.ToString("F6"); }
                    if (!Double.IsNaN(Z)) { gCode += " Z" + Z.ToString("F6"); }
                    if (!Double.IsNaN(R)) { gCode += " R" + (R + 0.0000005).ToString("F6"); }     //Prevents rounding errors
                    if (!Double.IsNaN(I)) { gCode += " I" + I.ToString("F6"); }
                    if (!Double.IsNaN(J)) { gCode += " J" + J.ToString("F6"); }
                    break;
                case "G03":
                    gCode = "G03 ";
                    if (!Double.IsNaN(X)) { gCode += " X" + X.ToString("F6"); }
                    if (!Double.IsNaN(Y)) { gCode += " Y" + Y.ToString("F6"); }
                    if (!Double.IsNaN(Z)) { gCode += " Z" + Z.ToString("F6"); }
                    if (!Double.IsNaN(R)) { gCode += " R" + (R + 0.0000005).ToString("F6"); }
                    if (!Double.IsNaN(I)) { gCode += " I" + I.ToString("F6"); }
                    if (!Double.IsNaN(J)) { gCode += " J" + J.ToString("F6"); }
                    break;
                case "G53":
                    gCode = "G53 ";
                    if (!Double.IsNaN(X)) { gCode += " X" + X.ToString("F4"); }
                    if (!Double.IsNaN(Y)) { gCode += " Y" + Y.ToString("F4"); }
                    if (!Double.IsNaN(Z)) { gCode += " Z" + Z.ToString("F4"); }
                    break;
                case "G00 G53":
                    gCode = "G00 G53 ";
                    if (!Double.IsNaN(X)) { gCode += " X" + X.ToString("F4"); }
                    if (!Double.IsNaN(Y)) { gCode += " Y" + Y.ToString("F4"); }
                    if (!Double.IsNaN(Z)) { gCode += " Z" + Z.ToString("F4"); }
                    break;
                default:
                    gCode = command;
                    break;
            }

            //Send gCode
            try { serial.WriteLine(gCode); }            //Send codes
            catch
            {
                State = 2;
                wnd.TextBox_output.AppendText("write. Error writing to serial.\n   state => 2\n");
                //bool resetResult = hardReset();
                return;
            }

            //Get machine response
            String tmpString;
            tmpString = read();
            if (tmpString != "ok\r" && tmpString != "\r" && !tmpString.Contains("Idle"))
            {
                wnd.TextBox_output.AppendText("GCODE: '" + gCode + "'   " + tmpString + "\n");
            }
        }
        
        public String read()
        {
            //Tasks:
            //  - Waits for a response from the machine
            //  - Reads the response
            //  - Identifies, expands upon, and outputs known error messages
            //Inputs: none
            //Returns:
            //  tmpstring - the message received from the machine, expanded upon if it is a known error

            if (State != 0) { return "ok\r"; }
            String tmpString = "ok\r";
            DateTime startTime = DateTime.Now; // will give the date for today
            Random rnd = new Random();
            if (rnd.Next(1, 20) == 2) { wnd.Refresh(); } //1/20th chance of a refresh

            try
            {
                while (serial.BytesToRead < 2)
                {
                    if (State != 0) { return "ok\r"; }
                    TimeSpan duration = DateTime.Now.Subtract(startTime); // will give the date for today
                    if (duration.TotalSeconds > 0.2) { wnd.Refresh(); }
                    if (duration.TotalSeconds > 40)
                    {
                        State = 2;
                        wnd.TextBox_output.AppendText("CNCread ERROR - 40 Second Timeout Reached. Probably zapped myself. state => 2\n");
                        return "CNCread ERROR - 40 Second Timeout Reached. Probably zapped myself. state => 2";
                    }
                }
                tmpString = serial.ReadLine();
                serial.DiscardInBuffer();

                if (tmpString != "ok\r" && tmpString != "\r" && !tmpString.Contains("Idle"))
                {
                    bool shouldStop = true;
                    switch (tmpString)
                    {
                        case "error:1\r": { tmpString = "GRBL ERROR 1: G-code words consist of a letter and a value. Letter was not found."; break; }
                        case "error:2\r": { tmpString = "GRBL ERROR 2: Numeric value format is not valid or missing an expected value."; break; }
                        case "error:3\r": { tmpString = "GRBL ERROR 3: Grbl ‘$’ system command was not recognized or supported."; break; }
                        case "error:4\r": { tmpString = "GRBL ERROR 4: Negative value received for an expected positive value."; break; }
                        case "error:5\r": { tmpString = "GRBL ERROR 5: Homing cycle is not enabled via settings."; break; }
                        case "error:6\r": { tmpString = "GRBL ERROR 6: Minimum step pulse time must be greater than 3usec"; break; }
                        case "error:7\r": { tmpString = "GRBL ERROR 7: EEPROM read failed. Reset and restored to default values."; break; }
                        case "error:8\r": { tmpString = "GRBL ERROR 8: Grbl ‘$’ command cannot be used unless Grbl is IDLE. Ensures smooth operation during a job."; break; }
                        case "error:9\r": { tmpString = "GRBL ERROR 9: G-code locked out during alarm or jog state"; break; }
                        case "error:10\r": { tmpString = "GRBL ERROR 10: Soft limits cannot be enabled without homing also enabled."; break; }
                        case "error:11\r": { tmpString = "GRBL ERROR 11: Max characters per line exceeded. Line was not processed and executed."; break; }
                        case "error:12\r": { tmpString = "GRBL ERROR 12: (Compile Option) Grbl ‘$’ setting value exceeds the maximum step rate supported."; break; }
                        case "error:13\r": { tmpString = "GRBL ERROR 13: Safety door detected as opened and door state initiated."; break; }
                        case "error:14\r": { tmpString = "GRBL ERROR 14: (Grbl-Mega Only) Build info or startup line exceeded EEPROM line length limit."; break; }
                        case "error:15\r": { tmpString = "GRBL ERROR 15: Jog target exceeds machine travel. Command ignored."; break; }
                        case "error:16\r": { tmpString = "GRBL ERROR 16: Jog command with no ‘=’ or contains prohibited g-code."; break; }
                        case "error:20\r": { tmpString = "GRBL ERROR 20: Unsupported or invalid g-code command found in block."; break; }
                        case "error:21\r": { tmpString = "GRBL ERROR 21: More than one g-code command from same modal group found in block."; break; }
                        case "error:22\r": { tmpString = "GRBL ERROR 22: Feed rate has not yet been set or is undefined."; break; }
                        case "error:23\r": { tmpString = "GRBL ERROR 23: G-code command in block requires an integer value."; break; }
                        case "error:24\r": { tmpString = "GRBL ERROR 24: Two G-code commands that both require the use of the XYZ axis words were detected in the block."; break; }
                        case "error:25\r": { tmpString = "GRBL ERROR 25: A G-code word was repeated in the block."; break; }
                        case "error:26\r": { tmpString = "GRBL ERROR 26: A G-code command implicitly or explicitly requires XYZ axis words in the block, but none were detected."; break; }
                        case "error:27\r": { tmpString = "GRBL ERROR 27: N line number value is not within the valid range of 1 – 9,999,999."; break; }
                        case "error:28\r": { tmpString = "GRBL ERROR 28: A G-code command was sent, but is missing some required P or L value words in the line."; break; }
                        case "error:29\r": { tmpString = "GRBL ERROR 29: Grbl supports six work coordinate systems G54-G59. G59.1, G59.2, and G59.3 are not supported."; break; }
                        case "error:30\r": { tmpString = "GRBL ERROR 30: The G53 G-code command requires either a G0 seek or G1 feed motion mode to be active. A different motion was active."; break; }
                        case "error:31\r": { tmpString = "GRBL ERROR 31: There are unused axis words in the block and G80 motion mode cancel is active."; break; }
                        case "error:32\r": { tmpString = "GRBL ERROR 32: A G2 or G3 arc was commanded but there are no XYZ axis words in the selected plane to trace the arc."; break; }
                        case "error:33\r": { tmpString = "GRBL ERROR 33: The motion command has an invalid target. G2, G3, and G38.2 generates this error, if the arc is impossible to generate or if the probe target is the current position."; break; }
                        case "error:34\r": { tmpString = "GRBL ERROR 34: A G2 or G3 arc, traced with the radius definition, had a mathematical error when computing the arc geometry. Try either breaking up the arc into semi-circles or quadrants, or redefine them with the arc offset definition."; break; }
                        case "error:35\r": { tmpString = "GRBL ERROR 35: A G2 or G3 arc, traced with the offset definition, is missing the IJK offset word in the selected plane to trace the arc."; break; }
                        case "error:36\r": { tmpString = "GRBL ERROR 36: There are unused, leftover G-code words that aren’t used by any command in the block."; break; }
                        case "error:37\r": { tmpString = "GRBL ERROR 37: The G43.1 dynamic tool length offset command cannot apply an offset to an axis other than its configured axis. The Grbl default axis is the Z-axis."; break; }
                        case "error:38\r": { tmpString = "GRBL ERROR 38: An invalid tool number sent to the parser"; break; }
                        case "ALARM:1\r": { tmpString = "GRBL ALARM 1: Hard limit triggered. Machine position is likely lost due to sudden and immediate halt. Re-homing is highly recommended."; break; }
                        case "ALARM:2\r": { tmpString = "GRBL ALARM 2: G-code motion target exceeds machine travel. Machine position safely retained. Alarm may be unlocked."; break; }
                        case "ALARM:3\r": { tmpString = "GRBL ALARM 3: Reset while in motion. Grbl cannot guarantee position. Lost steps are likely. Re-homing is highly recommended."; break; }
                        case "ALARM:4\r": { tmpString = "GRBL ALARM 4: Probe fail. The probe is not in the expected initial state before starting probe cycle, where G38.2 and G38.3 is not triggered and G38.4 and G38.5 is triggered."; break; }
                        case "ALARM:5\r": { tmpString = "GRBL ALARM 5: Probe fail. Probe did not contact the workpiece within the programmed travel for G38.2 and G38.4."; break; }
                        case "ALARM:6\r": { tmpString = "GRBL ALARM 6: Homing fail. Reset during active homing cycle."; break; }
                        case "ALARM:7\r": { tmpString = "GRBL ALARM 7: Homing fail. Safety door was opened during active homing cycle."; break; }
                        case "ALARM:8\r": { tmpString = "GRBL ALARM 8: Homing fail. Cycle failed to clear limit switch when pulling off. Try increasing pull-off setting or check wiring."; break; }
                        case "ALARM:9\r": { tmpString = "GRBL ALARM 9: Homing fail. Could not find limit switch within search distance. Search distance is defined as 1.5 * max_travel on search and 5 * pulloff on locate phases."; break; }
                        case "Hold:0\r": { tmpString = "GRBL HOLD 0: Hold complete. Ready to resume."; break; }
                        case "Hold:1\r": { tmpString = "GRBL HOLD 1: Hold in-progress. Reset will throw an alarm."; break; }
                        case "Door:0\r": { tmpString = "GRBL DOOR 0: Door closed. Ready to resume."; break; }
                        case "Door:1\r": { tmpString = "GRBL DOOR 1: Machine stopped. Door still ajar. Can’t resume until closed."; break; }
                        case "Door:2\r": { tmpString = "GRBL DOOR 2: Door opened. Hold (or parking retract) in-progress. Reset will throw an alarm."; break; }
                        case "Door:3\r": { tmpString = "GRBL DOOR 3: Door closed and resuming. Restoring from park, if applicable. Reset will throw an alarm."; break; }
                        default:
                            {
                                shouldStop = false;
                                tmpString = "ERROR: " + tmpString;
                                break;
                            }
                    }
                    if (shouldStop) { Stop(); }
                    wnd.Refresh();
                }
            }
            catch
            {
                State = 2;
                tmpString = "CNCread ERROR - state => 2";
                wnd.TextBox_output.AppendText("CNCread. Exception while reading serial. state => 2\n");
            }

            return tmpString;
        }

        public bool waitForCompletion()
        {
            //Tasks:
            //  - Waits for the machine to finish all commands in its buffer
            //  - Refreshes the window while it waits
            //Inputs: none
            //Returns:
            //  true if completion occurred, false in all other cases, including errors and emergency stops

            bool done = false;
            string response = "";
            DateTime startTime = DateTime.Now; // will give the date for today
            try
            {
                while (!done)
                {
                    if (State != 0) { return false; }
                    TimeSpan duration = DateTime.Now.Subtract(startTime); // will give the date for today
                    if (duration.TotalSeconds > 0.2) { wnd.Refresh(); }
                    if (duration.TotalSeconds > 60)
                    {
                        State = 2;
                        wnd.TextBox_output.AppendText("waitForCompletion() ERROR: 60 Second Timeout Reached. state => 2\n");
                        return false;
                    }

                    wnd.Delay(200);
                    if (serial.IsOpen == false) { return false; }
                    serial.WriteLine("?");
                    wnd.Delay(50);
                    response = read();
                    done = response.Contains("Idle");
                    if (response.Contains("Alarm")) { return false; }
                }
            }
            catch
            {
                State = 2;
                wnd.TextBox_output.AppendText("waitForCompletion() ERROR: Machine isn't responding. state => 2");
                return false;
            }
            return true;

        }

        public bool Home(String homeType = "")
        {
            //Tasks:
            //  - Sets all the GRBL settings
            //  - Homes the machine
            //  - Squares and rehomes if this is a setup home
            //Inputs:
            //  homeType - Which version of home to perform (null, "", "Setup1", or "Setup2") since there are slight differences between the options
            //Returns:
            //  true if homed successfully, false otherwise

            if (State == 0)
            {
                try
                {
                    double homeDebounce = 1.5; //Distance the home is allowed to search from current position(mm)
                                               //homeDebounce is inconsistant at lower values

                    //Set Shapeoko Settings
                    write("$0=10");      //Step Pulse (microseconds)
                    write("$1=255");     //Step Idle Delay, delay after motion stops before motor power cut to 25%(milliseconds, or 255=never reduce power)
                    write("$2=0");       //Step Port Invert, Inverts the pulse signals between normally high and normally low (binary Mask)
                    if (MachineNum == 4)
                    {
                        write("$3=0");   //Positive direction Mask (This specific value sets No axes to be inverted)
                    }
                    else
                    {
                        write("$3=2");    //Positive direction Mask (This specific value inverts the Y axis)
                    }
                    write("$4=0");       //Stepper Enable Pin, Sets if high or low to enable steppers
                    write("$5=0");       //Limit Switch pins normal state invert (0 normally high, 1 normally low)
                    write("$6=0");       //Probe Pin Invert (0 normally high, 1 normally low)
                    write("$10=255");    //Status report mask(sets what info to return when machine receives "?" command (Mask)
                    write("$11=0.015");  //Junction Deviation, basically sets how fast it will go through corners (mm)
                    write("$12=0.01");   //Arc Tolerance (mm)
                    write("$13=0");      //Report inches (boolean, 0=mm, 1=inches)

                    write("$20=0");      //Soft Limits (boolean, 1=on)
                    write("$21=0");      //Hard Limits (boolean)
                    write("$22=1");      //Homing Cycle (boolean)
                    write("$23=0");      //Homing Dir Invert (Mask)
                    if (homeType == "Setup1")
                    {
                        write("$24=500");    //Homing feed rate(mm/min)
                        write("$25=2500");   //Homing seek rate(mm/min)
                    }
                    else
                    {
                        write("$24=80");
                        write("$25=800");   //Homing seek rate(mm/min)
                    }
                    write("$26=25");     //Homing debounce (msec)
                    if (homeType == "Setup1") { write("$27=7"); } //Homing Pull Off. Large enough in case y axis is against back wall at start
                    else { write("$27=" + (homeDebounce)); }         //Homing pull off used for all consecutive homes(mm)
                    //The distance it moves after the pulloff is somewhere around 5x the pulloff distance plus the homeDebounce. I could not figure out the exact formula

                    write("$30=24000");  //Max Spindle Speed (RPM)
                    write("$31=0");      //Min Spindle Speed (RPM, spindle still stops at zero volts)
                    write("$32=0");      //Laser Mode (Boolean, Enabling Varies PWM as it ramps into and out of each motion)
                    write("$100=40");    //steps/mm for x axis
                    write("$101=40");    //steps/mm for y axis
                    write("$102=200");   //steps/mm for Z-Plus Z axis
                    write("$110=12000"); //Max feed rate(sprint) in x direction
                    write("$111=12000"); //Max feed rate(sprint) in y direction
                    write("$112=3500");  //Max feed rate(sprint) in z direction
                    write("$120=900");   //Max acceleration in x direction
                    write("$121=900");   //Max acceleration in y direction
                    write("$122=600");   //Max acceleration in z direction
                    if (homeType == "Setup1" || homeType == "Setup2")
                    {
                        write("$130=" + (cutAreaWidth * 25.4));   //Max X travel (mm)
                        write("$131=850");   //Max Y travel (mm)
                        write("$132=95");    //Max Z travel (mm)
                    }
                    else
                    {
                        write("$130=1");   //Max X travel (mm)
                        write("$131=1");   //Max Y travel (mm)
                        write("$132=1");    //Max Z travel (mm)
                    }

                    //Home
                    if (homeType == "Setup1") { wnd.TextBox_output.AppendText("Initial Homing.."); }
                    else if (homeType == "Setup2") { wnd.TextBox_output.AppendText("Detail Homing.."); }
                    else { wnd.TextBox_output.AppendText("Homing.."); }
                    String tmpString = "\r";
                    int tmpCounter = 0;
                    while (tmpString == "\r" && tmpCounter < 3)
                    {
                        //Exits once homing is completed
                        wnd.TextBox_output.AppendText(".");
                        wnd.Delay(100);
                        serial.WriteLine("$H");
                        wnd.Delay(500);
                        tmpString = read();
                        tmpCounter = tmpCounter + 1;
                    }
                    if (tmpString != "ok\r")
                    {
                        wnd.TextBox_output.AppendText("HOME OUT OF ALIGNMENT " + tmpString + "\n");
                        shutdown();
                        return false;
                    }

                    //Make sure settings are default in case they don't register in the next one for some reason
                    write("$24=80");
                    write("$27=" + (homeDebounce));  //Pull off(mm)
                    write("$130=1");   //Max X travel (mm)
                    write("$131=1");   //Max Y travel (mm)
                    write("$132=1");    //Max Z travel (mm)


                    //Set Origin Location(measured at tool change)
                    //This is the one that determines how deep in the spindle chuck the bit goes
                    write("G90 G20;");   //Absolute positioning, inch units
                    write("G10 P1 L2 X" + (-33.06 + homeDebounce / 25.4)
                        + " Y" + (-29.32 + homeDebounce / 25.4)
                        + " Z" + (-2.92 + homeDebounce / 25.4));
                    write("G01 F30");

                    //Square Machine
                    if (homeType == "Setup1")
                    {
                        bool result = Square();
                        return result;
                    }
                    else
                    {
                        write("G00", X: (cutAreaWidth));

                        //Done?
                        if (State != 0)
                        {
                            //shutdown(null,null);   //Might be causing crash during setup
                            wnd.TextBox_output.AppendText("Home ERROR: Machine State no longer 0. Shutting Down...");
                            shutdown();
                            return false;
                        }
                        else
                        {
                            waitForCompletion();
                            wnd.TextBox_output.AppendText("Done\n");
                        }
                    }
                }
                catch
                {
                    //Something went wrong during homing, usually a serial error
                    wnd.TextBox_output.AppendText("Home ERROR: Machine isn't responding." +
                        "Try Arduino Serial Monitor? state => 2");
                    State = 2;
                    shutdown();
                    return false;
                }
                wnd.Refresh();
            }
            else
            {
                return false;
            }

            return true;
        }

        public bool Square()
        {
            //Tasks:
            //  - Allows the user to square the machine
            //Inputs: none
            //Returns:
            //  whether or not the square was successful or not

            //Shift the machine a bit in the y direction so it's not always being done at the same spot
            wnd.TextBox_output.AppendText("Squaring...");
            Random random = new Random();
            write("G00 G53", X: -(7 / 25.4),
                Y: -0.1 - random.NextDouble() * 0.2);

            //Ask user to square machine
            MessageBox.Show("MACHINE #" + MachineNum.ToString() + ":\nSquare the Machine\n" +
            "    Push the machine all the way to the back on both sides to square it.\n" +
            "    Press Okay when ready to continue.");
            write("G00 G53", 
                X: -(2 / 25.4),
                Y: -(9 / 25.4),
                Z: -(2 / 25.4));
            waitForCompletion();
            bool result = Home("Setup2");
            return result;
        }

        public bool Stop()
        {
            //Tasks:
            //  - Stops the machine immediately. Can lose steps
            //  - Set the machine state to 1 for "connected but stopped"
            //Inputs: none
            //Returns:
            //  whether or not the stop worked

            if (State != 0) { return true; }
            try { wnd.Button_EmergencyStop.IsEnabled = false; }
            catch { }

            //stop
            try
            {
                State = 1;
                serial.DiscardOutBuffer();
                serial.WriteLine("!");
                wnd.Delay(100);
                serial.WriteLine("!");
                return true;
            }
            catch
            {
                State = 2;
                return false;
            }
            //emergency stop is deactivated at the end of the currently operating movement by the change tool function
        }

        public bool hardReset()
        {
            //Tasks:
            //  - Attempts to get the machine to do a hard reset
            //  - Reconnectes the serial connection
            //Inputs: none
            //Returns:
            //  Whether the hardreset worked or not
            wnd.TextBox_output.AppendText("Attempting Hard Reset...\n");
            wnd.Delay(300);

            //Close serial port if already open
            try
            {
                serial.Close();
            }
            catch { }

            //Re-establish serial communication
            try
            {
                serial.PortName = ("COM" + COMport);
                serial.BaudRate = 115200;
                //serial.Handshake = System.IO.Ports.Handshake.None;
                serial.Parity = Parity.None;
                serial.DataBits = 8;
                serial.StopBits = StopBits.One;
                serial.ReadTimeout = 3000; //200
                serial.WriteTimeout = 500; //50
                serial.DtrEnable = true;
                serial.RtsEnable = false;

                //Creates event handler for incomming data
                serial.Open();

                wnd.TextBox_output.AppendText("   serial opened...");
            }
            catch
            {
                wnd.TextBox_output.AppendText("   ERROR: Serial port failed to re-open. " +
                    "Check power, USB, and COM port. state => 2");
                State = 2;
                wnd.GrayButtons("Disconnected");
                return false;
            }

            // Rapid go home if possible
            write("G53", Z: -1);
            write("G00 G53", X: 0, Y: 0, Z: 0);
            waitForCompletion();

            //Hard reset
            byte[] resetCode = new byte[] { 0x18 };
            serial.Write(resetCode, 0, 1);
            wnd.Delay(2000);
            serial.Write(resetCode, 0, 1);
            wnd.Delay(2000);

            //Try to get a response from the machine
            String response = "";
            try
            {
                serial.WriteLine("?");
                response = read();
                wnd.TextBox_output.AppendText("   Response after 0x18 command: " + response);
            }
            catch
            {
                wnd.TextBox_output.AppendText("   Serial Sepmaphore Timeout!\n");
                State = 2;
            }

            //Try to restart the machine
            if (response.Contains("Idle") || response.Contains("ok"))
            {
                wnd.Delay(2000);
                setup();
            }
            else
            {
                State = 2;
            }

            //User output
            if (State == 0)
            {
                wnd.TextBox_output.AppendText("Hard reset worked!\n");
                return true;
            }
            else
            {
                wnd.TextBox_output.AppendText("Hard reset did not work.\n\n" +
                    "   Please re-start the machine. ");
                shutdown();
                return false;
            }
        }

        public bool CheckTool(int i = 5)
        {
            //Tasks:
            //  - Moves the machine to the left edge of the cutting area, along the racks to be milled
            //  - Askes the user to check everything is good before continuing
            //Inputs:
            //  i - the row at which to check the tool
            //Returns:
            //  true if good, false if bad

            wnd.TextBox_output.AppendText("User Confirmation...");
            write("S24000 M3"); //Start the spindle rotating clockwise at 24000 rpm to start
            write("G00", X: 0, Y: wnd.GR.Racks[i].YOrigin + bitRadius[0]);

            waitForCompletion();
            if (State == 0)
            {
                String messageString = "MACHINE #" + MachineNum.ToString() + ": Please Confirm:\n"
                    + "   1/16th-1/8th an inch from the bumper?\n"
                    + "   Correct tool?\n"
                    + "   Spindle spinning?\n";
                MessageBoxResult dialogResult = MessageBox.Show(messageString, "User Confirmation", MessageBoxButton.YesNo);
                if (dialogResult == MessageBoxResult.No)
                {
                    wnd.TextBox_output.AppendText("BAD X ALIGNMENT: Possible Fixes:\n" +
                        "   If the machine is cold, let it warm up for an hour.\n" +
                        "   If the sensor side's alignment is off too, then its a sensor problem.\n" +
                        "   If the sensor side is fine then it is a belt problem.\n");
                    State = 1;
                    return false;
                }
            }
            return true;
        }

        public void ChangeTool()
        {
            //Tasks:
            //  - If state == 0, performs the tool change process
            //  - If state == 1, Sets machine state to 0 and gjust goes home
            //  - Ungray all the buttons
            //Inputs: none
            //Returns: none
            if (State == 0)
            {
                //Re-Home
                wnd.GrayButtons("Running");
                write("M5"); //Stop the spindle
                write("G53", Z: -0.25);
                write("G00 G53", X: 0, Y: 0, Z: 0);
                waitForCompletion();
                if (Home() == false ||  wnd.CheckBox_ToolUse.IsChecked == false || State != 0)
                {
                    wnd.GrayButtons("Idle");
                    return;
                }

                //Normal Use - Do the tool change
                wnd.TextBox_output.AppendText("Tool Change...");
                write("F" + (30));
                write("G00 G53", Z:-0.01);
                write("G53", X:-17.05, Y:-17, Z:-0.01);
                MessageBox.Show("MACHINE #" + MachineNum.ToString() + "\n" +
                    "Remove vac attachment and insert new tool. \n\n" +
                    "Press okay when ready to continue");

                write("G53", X:-16.31, Y:-0.25, Z:-0.01);
                write("G00", Z:0.8);
                write("G01", Z:0.21);
                wnd.Delay(1500);
                MessageBox.Show("MACHINE #" + MachineNum.ToString() + ":\n" +
                    "Adjust tool height so it touches the table and tighten and tighten the collet.\n\n" +
                    "Press okay when ready to continue");

                /*
                //Probe
                wnd.TextBox_output.AppendText("First Probe...");
                write("G38.2 Z-1 F10"); //First quick probe
                wnd.TextBox_output.AppendText("Done ");
                write("G92 Z0");
                write("G00", Z:0.1);
                wnd.TextBox_output.AppendText("Second Probe...");
                write("G38.2 Z-1 F3");
                wnd.TextBox_output.AppendText("Done\n");
                write("G92 Z0"); //################CHANGE THIS TO CORRECT HEIGHT
                //write("G00", Z:1.25);
                */

                write("G00 G53", Z:-0.01);

                //Re-Home
                write("G53", Z: -0.25);
                write("G00 G53", X:0, Y:0, Z:0);
                waitForCompletion();
                if (Home() == false || State != 0) { return; }

                MessageBox.Show("MACHINE #" + MachineNum.ToString() + ":\n" +
                    "   Replace vac attachment.\n" +
                    "   Clean rails if needed(top and bottom)\n\n" +
                    "Press okay when ready to continue");
                wnd.GrayButtons("Idle");
            }
            else if (State == 1)
            {
                //Emergency stop is thrown - Stop machine and go home
                State = 0;
                wnd.Delay(300);
                byte[] resetCommand = new byte[] { Convert.ToByte(0x18) }; //Force stop the machine
                serial.Write(resetCommand, 0, 1);
                wnd.Delay(300);
                wnd.Refresh();
                write("M5"); //Stop the spindle
                write("G90 G20;");   //Absolute positioning, inch units
                write("G00 G53", Z:0); //Lift router
                wnd.Delay(2700);
                write("M5"); //Stop the spindle
                write("G90 G20;");   //Absolute positioning, inch units
                write("G00 G53", Z:0); //Lift router
                MessageBox.Show("Machine #" + MachineNum.ToString() + ":\n" +
                    "Emergency Stop Activated. \n\n" +
                    "Press okay when ready to go home");
                write("G00 G53", X:0, Y:0, Z:0);

                //Re-Home
                waitForCompletion();
                Home();
                MessageBox.Show("Machine #" + MachineNum.ToString() + ":\n" +
                    "Emergency Stop Complete. \n" +
                    "    If Y alignment is bad: Turn the machine off and square it\n" +
                    "    If Z alignment is bad: Perform tool change");
                wnd.GrayButtons("Idle");
            }
            else
            {
                //E-State = 2, Machine is unresponsive
                //shutdown(null, null); is performed at the end of the hard reset if unsuccessful
                write("M5"); //Stop the spindle
                wnd.GrayButtons("Disconnected");
            }
             wnd.Label_TimeLeft.Content = "Idle";
        }

        public void NewType(string content, double centerX, double centerY, double centerZ, double height)
        {
            //Tasks:
            //  - Engraves the input text into the rack at the set location
            //Inputs:
            //  content - the string containing the text to be input
            //  centerX - the x location of the center of the engraving
            //  centerY - the y location of the center of the engraving
            //  centerZ - the z location of the center of the engraving
            //  height - the dimension of the engraving in the y dimension. Used to scale the text
            //Returns: none

            Tuple<double, double, double, double, double, double> values;
            String filename;
            double spacing = 0.085;
            double hScale = 0.7;
            double xLeft = centerX - newPredictTypeWidth(content, spacing, hScale, height) / 2; //Position of left side of current letter
            double yBottom = centerY - height / 2;

            write("F" + (20));
            write("G00", Z:1);
            write("G00", X: xLeft, Y: yBottom);
            double scale = height / 0.42;

            double typeWidth = xLeft;

            for (int i = 0; i < content.Length; i++)
            {
                filename =  wnd.mainFolder + "Logo Carving\\Type\\" + content[i] + ".nc";
                if (File.Exists(filename) == false) { return; }
                values = getDimensions(filename);
                engraveFromFile(filename, xLeft, yBottom, centerZ, scale);
                xLeft = xLeft + values.Item2 * scale + spacing;
            }

            typeWidth = xLeft - typeWidth;
            write("G00", Z:1);
        }

        private double newPredictTypeWidth(string content, double spacing, double hScale, double height)
        {
            //Tasks:
            //  - Calculate the width of all the letters in the input text and figure out how wide the entire thing will be
            //Inputs:
            //  content - the string to be engraved
            //  spacing - the spacing between the letters (inches)
            //  hScale - the horizontal scale of the letters (unused)
            //  height - the dimension of the engraving in the y dimension. Used to scale the text
            //Returns:
            //  width - the width of the text toolpath(inches, does not include bitRadius)

            Tuple<double, double, double, double, double, double> letterDims;
            String filename;
            double width = 0;
            double scale = height / 0.42;

            for (int i = 0; i < content.Length; i++)
            {
                if (content[i] == ' ') { width = width + 0.1 + spacing; }
                else
                {
                    filename =  wnd.mainFolder + "Logo Carving\\Type\\" + content[i] + ".nc";
                    letterDims = getDimensions(filename);
                    width = width + letterDims.Item2 * scale + spacing;
                }
            }
            width = width - spacing;
            return width;
        }

        public double ExtractNumber(string original)
        {
            //Tasks:
            //  - Extracts a double from a string
            //Inputs:
            //  original - the string from which to extract the double
            //Returns:
            //  result - the double extracted from the string
            double result = Double.NaN;
            try
            {
                string numberChars = Regex.Replace(original, "[^0-9.-]", "");
                if (numberChars != "") { result = Convert.ToDouble(numberChars); }
            }
            catch { }
            return result;
        }

    }
}
