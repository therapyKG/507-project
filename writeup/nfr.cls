\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nfr}[2024/03/14 NFR template class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% Page setup
\LoadClass[11pt,a4paper]{article}
\RequirePackage[left=2cm,top=2cm,bottom=2cm,right=2cm]{geometry}

\renewcommand{\maketitle}{
   {\centering\bfseries\LARGE
   \vspace{.25cm}
   \@title\par
   \vspace{.25cm}
   }
}

% Footer
\RequirePackage{fancyhdr}
\RequirePackage[table]{xcolor}

\fancyhf{} % clear all header and footer fields

\pagestyle{fancy}

\fancyfoot[C]{\textcolor{gray}{\thepage}}

\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Hyperlinks and TOC for PDF
\RequirePackage[
        plainpages=false,%
        breaklinks,%
        colorlinks,%
        citecolor=black,%
        filecolor=black,%
        linkcolor=black,%
        urlcolor=black,%
        bookmarksopenlevel=2,%
        pdfpagemode=UseOutlines]{hyperref}
\AtBeginDocument{
    \hypersetup{
        pdftitle={\@title},%
        pdfauthor={\@author},%
        %pdfsubject={\@granttypeyear},%
        addtopdfcreator={using \csname ver@nfr.cls\endcsname~(https://github.com/einola/NFR\_template)}
    }
}

% Environments and counters for WPs, tasks, objectives, hypotheses, and research questions
% counters

% We can do Gantt charts
\RequirePackage{pgfgantt}

% Command to build a list of WPs, tasks, etc for a Gantt chart
\def\ganttlist{}
\newcommand*{\appendlist}[2]{%
    \ifx#1\empty%
        \xdef#1{#2}%
    \else%
        \xdef#1{#1, #2}%
    \fi}

% Work packages. Use \workpackage* to not include it in the Gantt chart
\newcounter{wpCntr}

\newcommand*{\@wpstar}[1]%
    {\refstepcounter{wpCntr}%
        \belowpdfbookmark{WP\thewpCntr:~#1}{WP\thewpCntr}%
        \subsubsection*{WP\thewpCntr:~#1}}

\newcommand*{\@wpnostar}[2]%
    {\@wpstar{#1}%
    \appendlist{\ganttlist}{1/\thewpCntr/wp\arabic{wpCntr}/#2}}

\newcommand*{\workpackage}{\@ifstar{\@wpstar}{\@wpnostar}}

% Tasks - Use \task* to not include it in the Gantt chart
\newcounter{taskCntr}[wpCntr]
\counterwithin{taskCntr}{wpCntr}

\newcommand*{\@taskstar}[1]%
    {\refstepcounter{taskCntr}%
    \paragraph{Task~\thetaskCntr:~#1}}

\newcommand*{\@tasknostar}[2]%
    {\@taskstar{#1}%
    \appendlist{\ganttlist}{2/\thetaskCntr/task\arabic{wpCntr}p\arabic{taskCntr}/#2}}

\newcommand*{\task}{\@ifstar{\@taskstar}{\@tasknostar}}

% Deliverables - Use \deliverable* to not include it in the Gantt chart
\newcounter{delivCntr}[wpCntr]
\counterwithin{delivCntr}{wpCntr}

\newcommand*{\@delivstar}[1]
    {\refstepcounter{delivCntr}%
    \emph{Deliverable~\thedelivCntr: #1}}

\newcommand*{\@delivnostar}[2]%
    {\@delivstar{#1}%
    \appendlist{\ganttlist}{3/\thedelivCntr/deliverable\arabic{wpCntr}p\arabic{delivCntr}/#2}}

\newcommand*{\deliverable}{\@ifstar{\@delivstar}{\@delivnostar}}

% Define a new Gantt chart element for deliverables using a star and some anchors from ganttmilestone
\pgfdeclareshape{ganttdeliverable}{
    \inheritsavedanchors[from=star]
    \inheritanchor[from=star]{text}
    \inheritanchor[from=star]{center}
    \inheritanchor[from=star]{mid}
    \inheritanchor[from=star]{base}
    \inheritanchor[from=star]{north}
    \inheritanchor[from=star]{south}
    \inheritanchor[from=star]{west}
    \inheritanchor[from=star]{north west}
    \inheritanchor[from=star]{south west}
    \inheritanchor[from=star]{east}
    \inheritanchor[from=star]{north east}
    \inheritanchor[from=star]{south east}
    \inheritanchorborder[from=star]
    \inheritbackgroundpath[from=star]
    \inheritanchor[from=ganttmilestone]{on top}
    \inheritanchor[from=ganttmilestone]{on bottom}
    \inheritanchor[from=ganttmilestone]{on right}
    \inheritanchor[from=ganttmilestone]{on left}
    }

% The ganttdeliverable chart element is defined exactly the same way as ganttmilestone, except for
    % shape=ganttdeliverable - for a different shape(!)
    % inner sep=4pt    - see below in the milestone section
    % left shift=1.0   - see below in the milestone section
    % right shift=-1.0 - see below in the milestone section
\newganttchartelement*{deliverable}{%
    deliverable/.style={%
        shape=ganttdeliverable, inner sep=4pt, draw, fill=black%
    },%
    deliverable incomplete/.style={/pgfgantt/deliverable, fill=black!25},%
    deliverable label text=\strut#1,%
    deliverable label font=\itshape,%
    deliverable label node/.style={%
        anchor=east, font=\ganttvalueof{deliverable label font}%
    },%
    deliverable inline label anchor=center,%
    deliverable inline label node/.style={%
        anchor=south, font=\ganttvalueof{deliverable label font}%
    },%
    deliverable progress label anchor=center,%
    deliverable progress label font=\scriptsize,%
    deliverable progress label node/.style={%
        anchor=west, font=\ganttvalueof{deliverable progress label font}%
    },%
    deliverable left shift=1.0,%
    deliverable right shift=-1.0,%
    deliverable top shift=.3,%
    deliverable height=.4%
}

% Milestones - Use \milestone* to not include it in the Gantt chart
\newcounter{mlstnCntr}[wpCntr]
\counterwithin{mlstnCntr}{wpCntr}

\newcommand*{\@mlstnstar}[1]
    {\refstepcounter{mlstnCntr}%
    \emph{Milestone~\themlstnCntr: #1}}

\newcommand*{\@mlstnnostar}[2]%
    {\@mlstnstar{#1}%
    \appendlist{\ganttlist}{4/\themlstnCntr/milestone\arabic{wpCntr}p\arabic{mlstnCntr}/#2}}

\newcommand*{\milestone}{\@ifstar{\@mlstnstar}{\@mlstnnostar}}

% Fix the predefined Gantt chart element called milestone, so that the diamond doesn't stretch. This is poorly explained dark magick and hacks, partly based on https://tex.stackexchange.com/questions/209252/milestone-aspect-ratio-in-pgfgantt-adjusted-for-units
\ganttset{
    milestone left shift=1,
    milestone right shift=-1,
    milestone/.style={shape=ganttmilestone, fill=black, inner sep=4pt}
    }

% Draw up the Gantt chart by looping over \ganttlist and calling \ganttgroup, \ganttbar, \ganttdeliverable, or \ganttmilestone, depending on where the information comes from. To be called within a suitably formatted ganttchart environment.
\newcommand*{\@drawgantt}[1]{
    \foreach \rank / \cntr / \name / \startdate / \enddate in \ganttlist {
        \ifnum \rank = 1
            \ganttnewline
            \ganttgroup[name=\name]{WP\cntr}{\startdate}{\enddate}
        \fi
        \ifnum \rank = 2
            \ganttnewline
            \ganttbar[name=\name]{Task \cntr}{\startdate}{\enddate}
        \fi
        \ifnum \rank = 3
            \ifnum #1 = 0
                \ganttdeliverable[name=\name]{}{\startdate}
            \else
                \ganttnewline
                \ganttdeliverable[name=\name]{Deliverable \cntr}{\startdate}
            \fi
        \fi
        \ifnum \rank = 4
            \ifnum #1 = 0
                \ganttmilestone[name=\name]{}{\startdate}
            \else
                \ganttnewline
                \ganttmilestone[name=\name]{Milestone \cntr}{\startdate}
            \fi
        \fi
    }
}

% \drawgantt* putts the milestone and deliverable triangle and star on top of the task bar, not in it's own line, like \dragantt does.
\newcommand*{\drawgantt}{\@ifstar{\@drawgantt{0}}{\@drawgantt{1}}}

% Objectives (numbered and not)
\newcounter{objCntr}
\newenvironment{objective*}%
    {\begin{quote}\begin{bfseries}\begin{flushleft}}%
    {\end{flushleft}\end{bfseries}\end{quote}}

\newenvironment{objective}[1][Objective~]%
    {\refstepcounter{objCntr}\begin{objective*}#1\theobjCntr:}%
    {\end{objective*}}

% Hypothesis (numbered and not)
\newcounter{hypCntr}
\newenvironment{hypothesis*}%
    {\begin{quote}\begin{bfseries}\begin{flushleft}}%
    {\end{flushleft}\end{bfseries}\end{quote}}

\newenvironment{hypothesis}[1][Hypothesis~]%
    {\refstepcounter{hypCntr}\begin{hypothesis*}#1\thehypCntr:}%
    {\end{hypothesis*}\ignorespacesafterend}

% Research questions (numbered and not)
\newcounter{resqCntr}
\newenvironment{resq*}%
    {\begin{quote}\begin{bfseries}\begin{flushleft}}%
    {\end{flushleft}\end{bfseries}\end{quote}}

\newenvironment{resq}[1][Research~question~]%
    {\refstepcounter{resqCntr}\begin{resq*}#1\theresqCntr:}%
    {\end{resq*}}

% References
\newcommand*{\wpref}[1]{WP\ref{#1}}
\newcommand*{\tskref}[1]{T\ref{#1}}
\newcommand*{\dref}[1]{D\ref{#1}}
\newcommand*{\mref}[1]{M\ref{#1}}

% Bibliography with natbib
\RequirePackage[sort&compress]{natbib}
% The bibliography can use a 9pt font, which equates to \footnotesize when the
% main document font size is 11pts
\renewcommand{\bibfont}{\footnotesize}

\endinput
