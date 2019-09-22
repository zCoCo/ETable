classdef ETable < dynamicprops & matlab.mixin.SetGet
    properties
        data; % Core Data Table
        shortNames; % Short Names for Each Valid Column
        unitsList; % Cosmetically Styled Units for Each Short Name (using latex)
    end
    methods
        % Instantiates table from the given spreadsheet or table (make sure 
        % to upgrade xls to xlsx) with the given short names for columns.
        % Note, 'source' can also be another ETable to copy (if only one 
        % argument given) or either a table source or the URI to an excel 
        % file.
        function obj = ETable(source, shortNames)
            % Creates a New ETable with the same contents as the given ETable
            % if only one argument is given and that argument is the 
            % ETable
            if nargin < 2
                other = source; 
                obj.data = other.data;
                obj.shortNames = other.shortNames;
                obj.unitsList = other.unitsList;
                
                % Copy all custom properties over
                for name = other.shortNames
                    obj.addprop(char(name));
                    obj.set(char(name), other.get(char(name)));
                end
            else
                obj.shortNames = shortNames;
                
                if istable(source)
                    obj.data = source;
                else
                    obj.data = readtable(source, 'ReadVariableNames',false);
                    % Prune Columns that are Empty or Contain NaN from Table:
                    w = width(obj.data);
                    c = 1;
                    while c<=w
                        if (...
                            ~iscell(obj.data{:,c}) && ~prod(~isnan(obj.data{:,c})) || ... % contains NaN
                            isequal(obj.data{2,c}, {''}) && isempty(strtrim(strjoin(cellstr(obj.data{2:end,c})))) ... %is empty
                        )
                            obj.data(:,c) = [];
                            w = w - 1; % Readjust width
                        else
                            c = c+1;
                        end
                    end

                    % Scoop up Unaltered Full Names into Variable Descriptions, for
                    % plotting labels:
                    obj.data.Properties.VariableDescriptions = obj.data{1,:};
                    % Dump them into Variable Names as well, for command-line #head
                    % display:
                    obj.data.Properties.VariableNames = matlab.lang.makeValidName(obj.data{1,:});
                    % Remove Header Row from Data:
                    obj.data(1,:) = [];
                end

                % Prune Any Rows which are All Empty (eg. due to equations in
                % excel which returned '').
                r = 1;
                h = height(obj.data);
                while r<=h
                    row = strtrim(join(obj.data{r,:}));
                    if isequal(row, {''})
                        obj.data(r,:) = [];
                        h = h-1;
                    else
                        r = r+1;
                    end
                end

                for c = 1:width(obj.data)
                    % Convert Strings to Numbers i/a:
                    nums = str2double(obj.data{:,c});
                    numericData = prod(~isnan(nums)); % nums is NaN free
                    if numericData
                        obj.data{:,c} = num2cell(nums);
                    end

                    % Create Object Properties Based on Short Name:
                    try
                        obj.addprop(char(shortNames(c)));
                        if numericData
                            tabData = cell2mat(obj.data{:,c});
                        else
                            tabData = string(obj.data{:,c});
                        end
                        obj.set(char(shortNames(c)), tabData);
                    catch e
                        warning('Possibly Wrong Number of Short Names Supplied');
                    end
                end
            end
            obj.unitsList = repmat("", 1,width(obj.data));
        end % ctor
        
        % Helper Function which Returns the Full Variable Name, as a Valid 
        % Variable Name, Associated with the Given shortName:
        function vfn = validFullName(obj, shortName)
            vfn = obj.data.Properties.VariableNames{obj.shortNames == shortName};
        end
        % Helper Function which Returns the Cosmetic (user-facing) Full 
        % Variable Name, Associated with the Given shortName:
        function cfn = cosmeticFullName(obj, shortName)
            cfn = obj.data.Properties.VariableDescriptions{obj.shortNames == shortName};
        end
        
        % Adds a Column with the Given Name, ShortName, and Values:
        function add(obj, n, sn, vs)
            % Add Parameter:
            obj.addprop(sn);
            obj.set(char(sn), vs);
            if numel(obj.shortNames)
                obj.shortNames(end+1) = sn;
            else
                obj.shortNames = string(sn); % Must be first entry being added
            end
            % Add to Core Data Table:
            if size(vs,1) == 1
                vs = vs'; % Ensure data is column-vector
            end
            obj.data{:, end+1} = num2cell(vs); % use full name for table headers
            obj.rename(sn, n); % Set all names
        end

        % Adds a Column to This Table where Each Entry is Interpolated
        % As a Value from colX -> colY in the src Table where Column X in
        % This Table is used as the Reference Value.
        % N.B.: All columns given as shortNames.
        % ex. T1.interp('Pressure in Valve B', 'Pb', PvsT_Table, 'T', 'P', 'Tb');
        function interp(obj, n, sn, src, colX, colY, x)
            % TODO: add 'extrap' to interp1 or strict boundary cutoffs  
            % (however that would be implemented for arbitrary datasets 
            % which might not be monotonic... or is it ideal for this to 
            % spit out NaN for OOB issues? 
            obj.add(n,sn, interp1(src.get(colX), src.get(colY), obj.get(x), 'linear'));
        end
        % Same as interp but Steals name and short name from colY of source
        % table.
        function interpS(obj, src, colX, colY, x)
            obj.interp(src.cosmeticFullName(colY),colY, colX, colY, x);
        end
        
        % Same as #interp but in quasi-2D (ie. stacked tables as in the
        % Thermodynamics textbook).
        function interpQ2(obj, n, sn, src, colX,colY,colV, x,y)
            xs = obj.get(x);
            ys = obj.get(y);
            vs = zeros(size(ys));
            for i = 1:numel(ys)
                Xs = src.get(colX);
                Ys = src.get(colY);
                Vs = src.get(colV);
                
                y_low = max(Ys(Ys < ys(i)));
                low = Ys == y_low;
                v_low = interp1(Xs(low), Vs(low), xs(i), 'linear');
                
                y_high = min(Ys(Ys >= ys(i)));
                high = Ys == y_high;
                v_high = interp1(Xs(high), Vs(high), xs(i), 'linear');
                
                vs(i) = (ys(i) - y_low)*(v_high - v_low)/(y_high - y_low) + v_low;
            end
            
            obj.add(n,sn, vs);
        end
        
        % Performs Logarithmic Decrement for a Signal Experiencing
        % Free-Vibration.
        % Returns the Damping Ratio, z, for the Data in the Column with the
        % Given Short Name, colY, as a Function of the Column with the
        % Given Short Name, colX, over the given range. Range must only
        % include one section of free-oscillation and nothing else.
        % Returns damping ratio, z, the natural frequency wn, equilibrium 
        % position (steady-state value), and the location of all the peaks 
        % identified as a struct with parameters X and Y.
        function [z, wn, peaks, equilibrium]  = logdec(obj, colX,colY, range)
            xs = obj.get(colX); 
            ys = obj.get(colY);
            
            if nargin > 3
                xs = xs(range);
                ys = ys(range);
            end
            
            peaks = struct('X',[],'Y',[]);
            % Perform a basic first pass to assess the data:
            [peaks.Y, peaks.X] = findpeaks(ys, xs); % Find all local maxima
            
            % Only select peaks which have gone down and back up again by
            % a selected prominence value (to avoid detecting noise at the
            % peaks as multiple separate peaks).
            equilibrium = ys(end); % Steady-state value.
            peaks.Y = peaks.Y(peaks.Y > equilibrium); % Filter out noise peaks at near minima
            prominence = peaks.Y - equilibrium; % Half-Prominence of all peaks
            % Take limit prominence as that of the half-prominence of the
            % middle peak or the fourth (want at least 4 peaks):
            medianPeak = max(floor(numel(peaks.Y)/2),4);
            prominence = prominence(medianPeak);
            % Reassess Peaks:
            [peaks.Y, peaks.X] = findpeaks(ys, xs, 'MinPeakProminence',prominence);
            
            % Perform Logarithmic Decrement:
            % Average across all possible spans with at least 3 peaks to 
            % try to eliminate effects of any errant peaks:
            if(numel(peaks.Y) < 3)
                error('Not enough peaks to perform logarithmic decrement.');
            else
                zs = [];
                peaksRel = peaks.Y - equilibrium;
                for i = 2:numel(peaks.Y)
                    d = log(peaksRel(1)/peaksRel(i)) / (i-1);
                    zs(end+1) = d / sqrt(4*pi^2 + d^2);
                end
                z = mean(zs);
                
                % Collect Associated Values:
                Td = mean(diff(peaks.X)); % Average Damped Period
                wd = 2*pi/Td; % Damped Natural Frequency
                wn = wd / sqrt(1-z^2); % Natural Frequency
            end
        end
            
        
        % Edits the Given Column with the Given Short Name by replacing its
        % values with the given new values:
        function edit(obj, sn, newVals)
            % Update Parameter:
            obj.set(char(sn), newVals);
            % Update Core Data Table:
            obj.data{:, obj.validFullName(sn)} = num2cell(newVals);
        end
        
        % Set the value of the first given variable to its average across 
        % alls rows where the second given variable has one of the given
        % values for each of the given values.
        % Rows where varB is not (within 5% of) any of the given values
        % remain unchanged.
        %{
        ex.: table.bin('A', 'B', 10,20)
        A | B            A | B
        1 | 10           2 | 10
        2 | 10           2 | 10
        3 | 10           2 | 10
        3 | 13     ->    3 | 13
        4 | 20           5 | 20
        5 | 20           5 | 20
        6 | 20           5 | 20
        %}
        function bin(obj, varA, varB, varargin)
            As = obj.get(char(varA));
            Bs = obj.get(char(varB));
            binned = As;
            for i=1:numel(varargin)
                cond = ETable.is(Bs,varargin{i});
                binned = binned.*~cond + mean(As(cond)).*cond;
            end
            obj.edit(varA,binned);
        end
        
        % Edits the Full Name Associated with the Given Short Name:
        function rename(obj, sn, newFullName)
            % Set Name:
            idx = obj.shortNames == sn;
            obj.data.Properties.VariableNames{idx} = matlab.lang.makeValidName(newFullName);
            obj.data.Properties.VariableDescriptions{idx} = newFullName;
            
            % Try to Extract Units from Name:
            units = regexp(newFullName, '(?<=\[).*(?=\])', 'match');
            if ~isempty(units)
                obj.unitsList(idx) = units(1);
            elseif numel(obj.unitsList) < find(idx,1,'last') || ismissing(obj.unitsList(idx))
                obj.unitsList(idx) = ""; % Add blank units if none exist yet
            end
        end
        
        % Sets the Cosmetic Units Associated with the Given ShortName
        function setUnits(obj, sn, us)
            obj.unitsList(obj.shortNames == char(sn)) = us;
        end
        
        % Returns the Units Associated with the Given Short Name
        function u = units(obj, sn)
            u = obj.unitsList(obj.shortNames == char(sn));
        end
        
        % Prints the Top of the Table in the Command Line:
        function head(obj)
            disp(head(obj.data,5));
        end
        
        % Returns a copy of this object as a new ETable
        function copy = copy(this)
            copy = ETable(this);
        end
        
        % Returns subsection of the current ETable as a Table containing 
        % all the columns between the columns with short names: colA, colB. 
        % If only colA is needed, just use table.get(col)
        function sub = cols(obj, colA, colB)
            idxA = find(obj.shortNames == colA, 1);
            idxB = find(obj.shortNames == colB, 1);
            sub = obj.data{:, idxA:idxB};
        end
        
        % Returns subsection of the current ETable as a Matrix containing 
        % the columns with the given indicies in the desired order
        function sub = selectColumns(obj, varargin)
            sz = size(obj.get(char(varargin{1})));
            sz(2) = length(varargin);
            sub = zeros(sz);
            for i = 1:numel(varargin)
                sub(:,i)= obj.get(char(varargin{i}));
            end
        end
        
        % Returns a Table Containing the Columns with the Given Short
        % Names.
        function tab = subColTable(obj, varargin)
            dat = obj.get(varargin);
            tab = array2table(cell2mat(dat));
            tab.Properties.VariableNames = varargin;
            desc = varargin;
            for i = 1:numel(varargin)
                desc{i} = obj.cosmeticFullName(varargin{i});
            end
            tab.Properties.VariableDescriptions = desc;
        end
        
        % Exports a Table Containing the Columns with the Given Short
        % Names to an Excel file with the Given Filename.
        function subColToExcel(obj, filename, varargin)
            tab = obj.subColTable(varargin{:});
            writetable(tab, filename);
        end
        
        % Returns a ETable which is a subtable of the given table where
        % each row is the average of all values that meet the conditions
        % given by each element of varargin, where varargin is a list of
        % conditional vectors obtained by performing, say,
        % ETable.is(table.parameterA, parameterValue) & table.parameterB>5
        function ST = subtable(obj, varargin)
            ST = obj.copy();
            % Summarize Data for Each Range:
            subdata = zeros(length(varargin), length(obj.shortNames));
            for c = 1:width(obj.data)
                col = obj.get(char(obj.shortNames(c))); % Fetch Column Data
                for r = 1:length(varargin)
                    subdata(r,c) = mean(col([varargin{r}]));
                    ST.set(char(obj.shortNames(c)), subdata(r,c));
                end
            end
            sub = array2table(subdata);
            sub.Properties.VariableNames = obj.data.Properties.VariableNames;
            sub.Properties.VariableDescriptions = obj.data.Properties.VariableDescriptions;
            ST.data = sub;
        end
        
        % Function Summary, displays and returns a summary table of the 
        % mean values of all variables in each of the given ranges.
        function STd = summary(obj, varargin)
            ST = obj.subtable(varargin{:});
            STd = ST.data;
            % TODO: Transfer over each dynamicprop (.get, .set)
            disp('Summary Table:');
            disp(STd);
        end
        
        % Produces a Stylized Plot of the Two Variables with the Given
        % Short Names Subject to the Given Range. Returns the plot handle.
        function ph = plot(obj, nameX, nameY, range, format)
            if nargin < 5
                format = 'o-';
            end
            % Obtain Data:
            xs = obj.get(char(nameX));
            ys = obj.get(char(nameY));
            
            % Determine Range:
            if nargin < 4
                range = true(size(xs));
            end
            
            % Plot Data:
            ph = plot(xs(range), ys(range), format);
            obj.label(nameX, nameY);
        end
        
        % Produces a Stylized Plot of the All the Variables with the Given
        % Short Names against the First Variable.
        % Returns the plot handles.
        function phs = multiplot(obj, nameX, varargin)
            disp("MULTIPLOT");
            phs = [];
            leg = {}; % legend entries
            hold on
                for i = 1:(nargin-2)
                    nameY = varargin{i};
                    phs(end+1) = obj.plot(nameX, nameY);
                    fullName = obj.cosmeticFullName(nameY); % Fetch full names
                    fullName(regexp(fullName,'[\n\r]')) = []; % Remove linebreaks
                    leg{i} = fullName;
                end
            hold off
            % Label Axes:
            fullNameX = obj.cosmeticFullName(nameX); % Fetch full names
            fullNameX(regexp(fullNameX,'[\n\r]')) = []; % Remove linebreaks
            xlabel(fullNameX, 'Interpreter', 'latex');
            ylabel('Output', 'Interpreter', 'latex');
            % Add Legend:
            legend(leg, 'Interpreter', 'latex');
        end
        
        % Produces a Stylized Plot of the Two Variables with the Given
        % Short Names Subject to the Given Range with Vertical Error Bars 
        % from the Variable with the Short Name nameE. Errorbars will only 
        % show up every n datapoints. Returns the plot handle.
        function eph = errorplot(obj, nameX, nameY, nameE, n, range, format)
            if nargin < 7
                format = 'o-';
            end
            % Obtain Data:
            xs = obj.get(char(nameX));
            ys = obj.get(char(nameY));
            es = obj.get(char(nameE));
            
            ebars = NaN(size(es));
            ebars(1:n:length(es)) = es(1:n:length(es));
            
            % Determine Range:
            if nargin < 6
                range = true(size(xs));
            end
            
            % Plot Data:
            eph = errorbar(xs(range), ys(range), ebars(range), format);
            
            obj.label(nameX, nameY);
        end
        
        % Creates a Plot with Error Bars for the Given X and Y Data Subject
        % to the Given Conditionals Range. Only plots points which are the 
        % average X and Y data for each value of varargin for the given 
        % variable, var.
        % Ex.
        % errorAvgAtplot('X','Y','dqc', ETable.is(V,9), 0.1, 'u', 1,2,3);
        % Plots a one point with errorbars for each value of u within 0.1 of 
        % (1,2,3) on a graph of Y vs X where V is 9.
        function eph = errorAvgAtplot(obj, nameX, nameY, nameE, range, window, var, varargin)
            % Obtain Data:
            xs = obj.get(char(nameX));
            ys = obj.get(char(nameY));
            es = obj.get(char(nameE));
            
            xs = xs(range);
            ys = ys(range);
            es = es(range);
            
            % Compute Points:
            vals = [varargin{:}];
            xps = [];%nan(size(vals));
            yps = [];%nan(size(vals));
            eyps = [];%nan(size(vals));
            exps = [];%nan(size(vals));
            for i=1:length(vals)
                rawVals = obj.get(char(var));
                cond = ETable.inrange(rawVals(range), vals(i)-window, vals(i)+window);
                if sum(cond)
                    xps(end+1) = mean(xs(cond));
                    yps(end+1) = mean(ys(cond));
                    eyps(end+1) = mean(es(cond));
                    exps(end+1) = 2*std(xs(cond));
                end
            end
            
            % Plot Data:
            eph = errorbar(xps, yps, eyps/2, eyps/2, exps/2, exps/2, 'o-');
            eph.MarkerSize = eph.MarkerSize / 2;
            
            obj.label(nameX, nameY);
        end
        
        % Helper Function which labels a plot, given the short names of the
        % x and y axes
        function label(obj, nameX, nameY)
            fullNameX = obj.cosmeticFullName(nameX); % Fetch full names
            fullNameY = obj.cosmeticFullName(nameY);
            fullNameX(regexp(fullNameX,'[\n\r]')) = []; % Remove linebreaks
            fullNameY(regexp(fullNameY,'[\n\r]')) = [];
            xlabel(fullNameX, 'Interpreter', 'latex');
            ylabel(fullNameY, 'Interpreter', 'latex');
        end
        
        % Convenience function that marks the last data point meeting the
        % where the variables in the varargin list are within 5% of their
        % associated values in the current plot of nameY vs nameX. Each 
        % datapoint is labeled with the conditionals then the coordinates 
        % of the point. The arrow to each datapoint has length l, angle a 
        % in radians, and horizontal alignment given by horizAlign
        % Lengths are referenced in terms of x-axis units.
        % Ex:
        % ETable.mark('t','T', 35,pi/2, 'V',9, 'Ua',1)
        % This will mark the last datapoint where V is 9, and Ua is 1 with
        % something like: {'9V, 1m/s', '10min, 300K'} with an arrow that is
        % 35minutes long (if units of 't' are minutes) at an angle of pi/2.
        function m = mark(obj, nameX,nameY, l,a, horizAlign, varargin)
            if ~mod(length(varargin),2) % ensure length of varargin is even
                cond = true(size(obj.data{:,1})); % select all datapoints
                label =  {'', ''};
                
                if length(varargin) > 1
                    vars = string(varargin(1:2:end));
                    args = [varargin{2:2:end}];
                    for i = 1:length(vars)
                        cond = cond & ETable.is(obj.get(char(vars(i))), args(i));
                        if i>1
                            label{1} = strcat(label{1}, {', '});
                        end
                        label{1} = strcat(label{1}, string(args(i)), obj.units(vars(i)));
                    end
                end
                
                xs = obj.get(char(nameX)); xs = xs(cond); 
                ys = obj.get(char(nameY)); ys = ys(cond);
                
                % Prune Outliers
                out = isoutlier(xs);
                xs(out) = []; ys(out) = [];
                
                if ~isempty(xs)
                    x = xs(end); y = ys(end);
                    label{2} = strcat(string(floor(x)), obj.units(nameX), {', '}, string(floor(y)), obj.units(nameY));
                    m = ETable.arrow(x,y, l,a, label, 'HorizontalAlignment', horizAlign);
                end
            else
                error('#ETable::mark requires an even number of pairs of variables and values');
            end
        end
        
        % Convenience function that puts an annotation (arrow pointing to) 
        % the final point that meets a given conditionals list in the  
        % current plot of nameY vs nameX.
        % l is the length of the arrow, a is angle, and t is the text, 
        % along with a vararginlist of parameters.
        % Lengths are referenced in terms of x-axis units.
        function a = annotate(obj, nameX,nameY, cond, l,a, t, varargin)
            xs = obj.get(char(nameX)); xs = xs(cond); 
            ys = obj.get(char(nameY)); ys = ys(cond);
            if ~isempty(xs)
                x = xs(end); y = ys(end);
                a = ETable.arrow(x,y, l,a, t,varargin);
            end
        end
        
        function tab2 = binCompressTable(tab, namesX, nameB, bins, window, range)
            tab2 = ETable(array2table([]), []);
            for nx = namesX
                if nx == "X"
                end
                tab.add(char("Std. of " + tab.cosmeticFullName(char(nx))), char("s"+nx), zeros(size(tab.get(char(nx)))));
                [~,~,X,S] = aggressiveBin(tab, nx, nameB, char("s"+nx), bins, window, range);
                tab2.add(tab.cosmeticFullName(char(nx)), char(nx), X);
                tab2.add(char("Std. of " + tab.cosmeticFullName(char(nx))), char("s"+nx), S);
                tab2.add(char("Uncertainty in " + tab.cosmeticFullName(char(nx))), char("d"+nx), 2.*S);
            end
        end

        function [X,S,x_sm,s_sm] = aggressiveBin(tab, nameX, nameB, nameSTD, bins, window, range)
            if nargin < 5
                window = 0.15;
            end
            x_sm = nan(numel(bins),1); % Small x range (on entry per bin)
            s_sm = nan(numel(bins),1);

            xdat = tab.get(char(nameX));
            X = xdat;
            xinrange = xdat(range);
            bdat = tab.get(char(nameB));
            binrange = bdat(range);
            S = tab.get(nameSTD);
            for i = 1:numel(bins)
                b = bins(i);

                brange = ETable.inrange(bdat, b-window, b+window) .* range;
                s = std(xinrange(ETable.inrange(binrange, b-window, b+window)));
                if isnan(s)
                    s = 0;
                end
                S = S.*~brange + s .* brange;
                m = mean(xinrange(ETable.inrange(binrange, b-window, b+window)));
                if isnan(m)
                    m = 0;
                end
                X = X.*~brange + m .* brange;
                x_sm(i) = m;
                s_sm(i) = s;
            end
        end
        function [X,S] = ab(T,x,b,s,bs,w,r)
            [X,S] = aggressiveBin(T,x,b,s,bs,w,r);
        end
        
        % Exports All the Columns Given in 'cols' (by Shortname) to an 
        % Excel File with the Given 'filename'. Number of sigfigs for
        % numbers can be given with sigfigs (4 by default).
        % If no columns are given, all columns will be exported.
        function export2Excel(obj, filename, cols, sigfigs)
            if nargin < 3
                cols = obj.shortNames;
            end
            if nargin < 4
                sigfigs = 4;
            end

            table = array2table(string(zeros(height(obj.data), length(cols))));
            for i = 1:numel(cols)
                table.Properties.VariableNames{i} = char(cols(i));
                table{:,i} = string(num2str(obj.get(char(cols(i))), sigfigs));
            end
            writetable(table, char(string(filename)+".xlsx"));
        end
    end
    
    methods(Static)
        % Loads a Table from a Single-Line Column-wise Text File where 
        % New Entries are Delimited by Spaces with 'n_cols' entries per row.
        % As an example, this can be useful for copying a table from a pdf.
        % Note: All entries must be numbers; leave column headers out of
        % file.
        % Column Headers Must be Given in String Array 'headers'.
        % Short Names (variable ids) must be given in shortNames
        function obj = loadFromLineFile(file, n_cols, headers, shortNames)
            fID = fopen(file, 'r');
            mat = fscanf(fID, '%f', [n_cols Inf])';
            tab = cell2table(cellfun(@num2str, num2cell(mat), 'un',0));
            
            % Scoop up Unaltered Full Names into Variable Descriptions, for
            % plotting labels:
            tab.Properties.VariableDescriptions = cellstr(headers);
            tab.Properties.VariableNames = matlab.lang.makeValidName(cellstr(headers));
            
            obj = ETable(tab, shortNames);
        end
        
        % Convenience function that adds the given text as a caption to the
        % figure.
        function c = caption(t)
            dim = [0.1, 0.07, 0, 0];
            c = annotation('textbox', dim, 'String', t, 'FitBoxToText', 'on', 'LineStyle', 'none', 'Interpreter', 'latex');
        end
        
        % Draws a grey verical dashed line at the given X-axis value on the
        % current plot, with a label of the given text at the bottom (or
        % top).
        % side: 'left','right','center','auto'
        % valign: 'top','bottom'
        function vline(x, txt, side, valign, color)
            if nargin < 3
                side = 'auto';
            end
            if nargin < 4
                valign = 'bottom';
            end
            if nargin < 5
                color = [0.5 0.4 0.4]; % grey
            end
            
            if strcmp(side, 'auto')
                if x > mean(xlim)
                    side = 'right';
                else
                    side = 'left';
                end
            end
            
            hold on
                plot([x x], ylim, ':', 'Color', color);
                size = ylim;
                if strcmp(valign, 'bottom')
                    fact = 0.05;
                else
                    fact = 0.95;
                end
                text(x, fact*diff(size) + size(1), char(txt), 'Color', color, 'HorizontalAlignment', side, 'Interpreter', 'latex');
            hold off
        end
        
        % Draws a grey horizontal dashed line at the given Y-axis value on 
        % the current plot, with a label of the given text at the left.
        % pos: 'left','center','right'
        % valign: 'top','middle','bottom','cap','baseline'
        function hline(y, txt, pos, valign, color)
            if nargin < 3
                hfact = 1; % Horizontal Positioning Factor
            else
                hfact = (find(pos==["left" "center" "right"],1) - 1) / 2;
                if isempty(hfact)
                    hfact = 1;
                end
            end
            if nargin < 4
                if y > mean(ylim)
                    valign = 'top';
                else
                    valign = 'bottom';
                end
            end
            if nargin < 5
                color = [0.5 0.4 0.4]; % grey
            end
            
            hold on
                plot(xlim, [y y], ':', 'Color', color);
                size = xlim;
                text(hfact*diff(size) + size(1), y, char(txt), 'Color', color, 'HorizontalAlignment', pos, 'VerticalAlignment', valign, 'Interpreter', 'latex');
            hold off
        end
        
        % Convenience function that draws an arrow to point x,y with length
        % l, angle a, and optional text, t along with a list of parameters.
        % Lengths are referenced in terms of x-axis units.
        function a = arrow(x,y, l,a, t, varargin)
            p = [x,y];

            axs = gca; % Get current axes
            sx = diff(axs.XLim); % Get size of each axis
            sy = diff(axs.YLim);
            o = p - l * [cos(a), sin(a)*sy/sx];

            d = p-o;
            a = quiver(o(1),o(2), d(1),d(2), 0, 'MaxHeadSize', 0.05*sx/norm(l * [cos(a), sin(a)*sy/sx]), 'HandleVisibility','off'); % don't show in legend
            if nargin > 4
                args = [varargin, {'Interpreter','latex'}];
                text(o(1),o(2),t, args{:});
            end
        end
        
        % Convenience function that returns whether the given value is 
        % within the given fractional range of the given target:
        function w = within(val, range, target)
            w = val < (target + range.*target) & val > (target - range.*target);
        end
        % Convenience function that returns whether the given value is
        % within 5% of the given value:
        function i = is(val, target)
            if target ~= 0
                i = ETable.within(val, 0.12, target);
            else
                i = ETable.inrange(val, -0.1, 0.1);
            end
        end
        % Convenience Function that returns whether the given value is 
        % within the given range:
        function w = inrange(val, lb,ub)
            w = val <= ub & val >= lb;
        end
    end
end